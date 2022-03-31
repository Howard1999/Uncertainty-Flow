#! /usr/bin/python3.7

from distutils.command.config import config
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

from dataset.toy_1d_data import load_wiggle
from module.flow import cnf


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
        prior.load_state_dict(torch.load(args.load, map_location=args.gpu))
    # load data
    if configs['dataset'] == 'wiggle':
        x, y = load_wiggle
    else:
        x = np.load(configs['dataset']['x'])
        y = np.load(configs['dataset']['y'])
    # create dataset and dataloader
    my_dataset = MyDataset(condition=torch.Tensor(x).cuda(args.gpu), generated_output=torch.tensor(y).float().cuda(args.gpu))
    train_loader = data.DataLoader(my_dataset, shuffle=True, batch_size=configs['hyper-parameters']['batch'])
    # optimizer
    optimizer = optim.Adam(prior.parameters(), lr=configs['hyper-parameters']['lr'])
    # create save folder
    if args.save is not None:
        save_dir = os.path.join(args.save, configs['name'], timestamp)
        os.makedirs(save_dir, 0o0755)
        shutil.copyfile(args.cfg, os.path.join(save_dir, 'cfg.json'))
    # wandb
    if args.wandb == True:
        wandb.init(
            project="uncertain-flow",
            entity="suyihao1999",
            config=configs,
            name=os.path.join(configs["name"], timestamp),
            group=args.group)

    ''' Training Loop '''
    ite = 0
    for epoch in tqdm(range(configs['hyper-parameters']['epoch'])):
        for condition, generated_output in train_loader:
            # compute from y to z, and probabilty delta
            approx21, delta_log_p2 = prior(generated_output.squeeze(1), condition, torch.zeros(configs['hyper-parameters']['batch'], generated_output.shape[2], 1).to(generated_output))
            # compute z log probabilty
            approx2 = standard_normal_logprob(approx21).view(configs['hyper-parameters']['batch'], -1).sum(1, keepdim=True)
            # compute y log probabilty
            delta_log_p2 = delta_log_p2.view(configs['hyper-parameters']['batch'], generated_output.shape[2], 1).sum(1)
            log_p2 = (approx2 - delta_log_p2)
            # loss = - loglikelihood
            loss = -log_p2.mean()
            # update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # record
            if args.wandb == True:
                wandb.log({
                    'epoch': epoch,
                    'iteration': ite,

                    'loss': loss.item()
                })

            ite += 1

        # save model
        if args.save is not None and (epoch+1) % args.save_per == 0:
            torch.save(
                prior.state_dict(), os.path.join(save_dir, 'epoch'+str(epoch+1)+'.pt')
            )
