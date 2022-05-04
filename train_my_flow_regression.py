#! /usr/bin/python3.7

import json
import argparse
import os
import datetime
import shutil
from tqdm import tqdm
from math import log, pi

import torch
import wandb
import numpy as np
from torch.utils import data
from torch import optim

from dun_datasets.utils import position_encode
from dun_datasets.UCI_gap_loader import *
from module.flow import cnf
from condition_sampler.flow_sampler import FlowSampler


def standard_normal_logprob(z):
    dim = z.size(-1)
    log_z = -0.5 * dim * log(2 * pi)
    return log_z - z.pow(2) / 2


class MyDataset(data.Dataset):
    def __init__(self, condition, generated_output, transform=None):
        self.condition = condition
        self.generated_output = generated_output
        self.transform = transform

    def __getitem__(self, index):
        x = self.condition[index]
        y = self.generated_output[index]


        if self.transform is not None:
            x = self.transform(x)
            y = self.transform(y)

        return x, y

    def __len__(self):
        return len(self.generated_output)


if __name__ == "__main__":
    timestamp = '-'.join(str(datetime.datetime.now()).split())

    ''' Read commad line arguments '''
    parser = argparse.ArgumentParser(description="Flow trainer")
    
    # config-files
    parser.add_argument("--cfg", required=True, type=str, help="path to the configs")
    # wandb
    parser.add_argument("--wandb", default=False, type=bool, help="wheather record infomation to wandb")
    parser.add_argument("--group", default=None, type=str, help="what group the wandb record belongs to")
    # model save
    parser.add_argument("--save", default=None, type=str, help="where the model to be saved, if not specify model won't be saved")
    parser.add_argument("--save_per", default=1, type=int, help="save model per n EPOCH")
    # model load
    parser.add_argument("--load", default=None, type=str, help="load model from")
    # rand seed
    parser.add_argument("--seed", default=None, type=int, help="set torch random seed")
    # gpus
    parser.add_argument("--gpu", default=0, type=int, help="specify the gpu to be used")

    args = parser.parse_args()
    
    ''' Initialize all setting '''
    # set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
    # read config
    with open(args.cfg) as fp:
        configs = json.load(fp) # assume all key:value are ok, didn't check
    # create model
    prior = cnf(configs['model-structure']['input-dim'], 
                configs['model-structure']['flow-modules'], 
                configs['model-structure']['condition-size'],
                configs['model-structure']['num-block']).cuda(args.gpu)
    # load model
    if args.load is not None:
        prior.load_state_dict(torch.load(args.load, map_location=torch.device('cuda:'+str(args.gpu))))
    # load data
    x, _, _, _, y, _, _, _ = load_gap_UCI(configs['dataset'], gap=False)
    x, y = x[:, None, :], y[:, None, :]

    x_shape, y_shape = x.shape, y.shape
    x_min, x_max, x_var = np.min(x), np.max(x), np.var(x)
    y_min, y_max, y_var = np.min(y), np.max(y), np.var(y)

    # condition sampler
    condition_sampler = FlowSampler(configs['sampler-setting']['input-shape'], 
                                    configs['sampler-setting']['flow-modules'], 
                                    configs['sampler-setting']['num-block'], gpu=args.gpu)
    if configs['sampler-setting']['load'] is not None:
        condition_sampler.load(configs['sampler-setting']['load'])
    else:
        condition_sampler.fit(x,
                              batch=configs['sampler-setting']['hyper-parameters']['batch'], 
                              lr   =configs['sampler-setting']['hyper-parameters']['lr'],
                              epoch=configs['sampler-setting']['hyper-parameters']['epoch'])

    # position encode
    if configs['position_encoding']:
        x = position_encode(x, axis=2)
    # create dataset and dataloader
    my_dataset = MyDataset(condition=torch.Tensor(x).cuda(args.gpu), generated_output=torch.tensor(y).float().cuda(args.gpu))
    train_loader = data.DataLoader(my_dataset, shuffle=True, batch_size=configs['hyper-parameters']['batch'])
    # optimizer
    optimizer = optim.Adam(prior.parameters(), lr=configs['hyper-parameters']['lr'])
    # create save folder
    if args.save is not None:
        save_dir = os.path.join(args.save, configs['name'], timestamp)
        sampler_path = os.path.abspath(os.path.join(save_dir, 'sampler.pt'))        
        os.makedirs(save_dir, 0o0755)
        configs['sampler-setting']['load'] = sampler_path
        with open(os.path.join(save_dir, 'cfg.json'), 'w') as fp:
            json.dump(configs, fp)
        condition_sampler.save(sampler_path)
    # wandb
    if args.wandb == True:
        wandb.init(
            project="uncertain-flow",
            entity="suyihao1999",
            config=configs,
            name=os.path.join(configs["name"], timestamp),
            group=args.group)


    def fit(prior, condition, generated_output, weight):
        # compute from y to z, and probabilty delta
        approx21, delta_log_p2 = prior(generated_output, condition, torch.zeros(generated_output.shape[0], generated_output.shape[1], 1).to(generated_output))
        # compute z log probabilty
        approx2 = standard_normal_logprob(approx21).view(generated_output.shape[0], -1).sum(1, keepdim=True)
        # compute y log probabilty
        delta_log_p2 = delta_log_p2.view(generated_output.shape[0], generated_output.shape[1], 1).sum(1)
        log_p2 = (approx2 - delta_log_p2)
        # loss = - loglikelihood
        loss = -(log_p2 * weight).mean()
        # update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()


    ''' Training Loop '''
    ite = 0
    for epoch in tqdm(range(configs['hyper-parameters']['epoch'])):
        for condition, generated_output in train_loader:
            true_data_weight = torch.tensor([[1.]] * condition.size()[0]).cuda(args.gpu)

            # genarete noise
            x_u = np.random.uniform(x_min - x_var, x_max + x_var, (configs["hyper-parameters"]["train-noise"], 1, x_shape[2]))
            y_u = np.random.uniform(y_min - y_var, y_max + y_var, (configs["hyper-parameters"]["train-noise"], 1, y_shape[2]))
            # compute noise weight
            noise_logp = condition_sampler.logprob(torch.tensor(x_u).float().cuda(args.gpu))
            
            # noise_weight = torch.clamp(1 - torch.exp(noise_logp), 0, 1)  # (1-p)
            # noise_weight = 1 / (1 + torch.exp(noise_logp))               # 1 / (1+p)
            noise_weight = torch.pow(10, -torch.exp(noise_logp))         # 10 ** -p

            # add into batch
            if configs['position_encoding']:
                x_u = position_encode(x_u, axis=2)
            condition = torch.cat([condition, torch.tensor(x_u).float().cuda(args.gpu)])
            generated_output = torch.cat([generated_output, torch.tensor(y_u).float().cuda(args.gpu)])

            weight = torch.cat([true_data_weight, noise_weight])
            # fit
            loss = fit(prior, condition, generated_output, weight)
            # record
            if args.wandb == True:
                wandb.log({
                    'epoch': epoch,
                    'iteration': ite,

                    'loss': loss
                })
            ite += 1

        # save model
        if args.save is not None and (epoch+1) % args.save_per == 0:
            torch.save(
                prior.state_dict(), os.path.join(save_dir, 'epoch'+str(epoch+1)+'.pt')
            )
