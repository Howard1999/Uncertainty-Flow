#! /usr/bin/python3.7

from matplotlib.image import imread
import torch
from torch.utils import data

import json

import numpy as np
import argparse
import matplotlib.pyplot as plt
from math import log, pi
from tqdm import tqdm

from module.flow import cnf
from dun_datasets.utils import position_encode
from dun_datasets.additional_gap_loader import *
from my_datasets.toy_1d_data import *
from condition_sampler.flow_sampler import FlowSampler


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


def visualize_uncertainty(savePath, gt_x, gt_y, xdata, mean, var, weight):
    dyfit = np.sqrt(var)
    plt.plot(gt_x, gt_y, 'ok', ms=1)
    plt.plot(xdata, mean, '-', color='g')
    plt.plot(xdata, dyfit, '-', color='r')
    plt.plot(xdata, weight, '-', color='b')
    plt.fill_between(xdata, mean - dyfit, mean + dyfit, color='g', alpha=0.2)
    plt.savefig(savePath)

def sortData(x, y):
    x_sorted, y_sorted = zip(*sorted(zip(x, y)))
    return np.asarray(x_sorted), np.asarray(y_sorted)

def main(configs, device, model_path):
    torch.manual_seed(0)

    # create model
    prior = cnf(configs['model-structure']['input-dim'], 
                configs['model-structure']['flow-modules'], 
                configs['model-structure']['condition-size'],
                configs['model-structure']['num-block']).cuda(device)
    prior.load_state_dict(torch.load(model_path))
    prior.eval()
    # load data
    if configs['dataset'] == 'wiggle':
        x, y = load_wiggle_1d()
        x, y = x[:, None, :], y[:, None, :]
    elif configs['dataset'] == 'matern':
        x, y = load_matern_1d()
        x, y = x[:, None, :], y[:, None, :]
    elif configs['dataset'] == 'agw':
        x, y = load_agw_1d()
        x, y = x[:, None, :], y[:, None, :]
    elif configs['dataset'] == 'dun':
        x, y, _, _ = load_my_1d()
        x, y = x[:, None, :], y[:, None, :]
    elif configs['dataset'] == 'ring':
        x, y = load_ring_1d()
        x, y = x[:, None, :], y[:, None, :]
    elif configs['dataset'] == 'compose':
        x, y = load_compose_1d()
        x, y = x[:, None, :], y[:, None, :]
    else:
        x = np.load(configs['dataset']['x'])
        y = np.load(configs['dataset']['y'])

    # condition sampler
    condition_sampler = FlowSampler(configs['sampler-setting']['input-shape'], 
                                    configs['sampler-setting']['flow-modules'], 
                                    configs['sampler-setting']['num-block'], gpu=args.gpu)

    if configs['sampler-setting']['load'] is not None:
        condition_sampler.load(configs['sampler-setting']['load'])
    else:
        print('WARNING: sampler is not specify, weight evaluate result might not work')
        condition_sampler.fit(x, 32, 5e-3, 50)

    gt_X, gt_y = sortData(x, y)

    X_eval = np.linspace(-5, 5, 100).reshape(-1, 1)
    y_eval = np.linspace(0, 0, 100).reshape(-1, 1)

    evalset = MyDataset(torch.Tensor(X_eval).to(device), torch.Tensor(y_eval).to(device))
    # print("shape (gtX, gtY, evalX, eval Y) = ", gt_X.shape, gt_y.shape, X_eval.shape, y_eval.shape)

    mean_list = []
    var_list = []
    x_list = []
    weight_list = []

    for _, x in tqdm(enumerate(evalset)):
        input_x = torch.from_numpy(np.random.normal(loc=0, scale=1, size=(100, 1, 1))).float().to(device)

        condition_y = x[0].expand(100, 1, configs['model-structure']['input-dim'])
        logp = condition_sampler.logprob(condition_y)
        weight = torch.pow(10, -torch.exp(logp))
        
        if configs['position_encoding']:
            condition_y = condition_y.cpu()
            condition_y = position_encode(condition_y, axis=2)
            condition_y = torch.tensor(condition_y).to(device)

        approx21 = prior(input_x, condition_y, None, reverse=True)

        np_x = float(x[0].detach().cpu().numpy()[0])
        np_var = float(torch.var(approx21).detach().cpu().numpy())
        np_mean = float(torch.mean(approx21).detach().cpu().numpy())
        np_weight = weight.mean().cpu().item()

        x_list.append(np_x)
        var_list.append(np_var)
        mean_list.append(np_mean)
        weight_list.append(np_weight)

    savePath = "var.png"
    visualize_uncertainty(savePath, gt_X.reshape(-1), gt_y.reshape(-1), x_list, mean_list, var_list, weight_list)

if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser(description="Uncertainty trainer")
    parser.add_argument("--load", type=str)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cfg", type=str, default="")
    
    args = parser.parse_args()

    # config
    with open(args.cfg) as fp:
        configs = json.load(fp) # assume all key:value are ok, didn't check
    # gpu
    device = torch.device("cuda:"+str(args.gpu))
    # main
    main(configs, device, args.load)