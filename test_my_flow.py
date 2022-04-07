#! /usr/bin/python3.7

import torch
from torch.utils import data

import json

import numpy as np
import argparse
import matplotlib.pyplot as plt
from math import log, pi
from tqdm import tqdm

from module.flow import cnf
from dataset.toy_1d_data import load_wiggle


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


def visualize_uncertainty(savePath, gt_x, gt_y, xdata, mean, var):
    dyfit = 2 * np.sqrt(var)
    plt.plot(gt_x, gt_y, 'ok', ms=1)
    plt.plot(xdata, mean, '-', color='g')
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
    # load dataset
    if configs['dataset'] == 'wiggle':
        x, y = load_wiggle()
    else:
        x = np.load(configs['dataset']['x'])
        y = np.load(configs['dataset']['y'])

    gt_X, gt_y = sortData(x, y)

    X_eval = np.linspace(-5, 5, 100).reshape(-1, 1)
    y_eval = np.linspace(0, 0, 100).reshape(-1, 1)

    evalset = MyDataset(torch.Tensor(X_eval).to(device), torch.Tensor(y_eval).to(device))
    # print("shape (gtX, gtY, evalX, eval Y) = ", gt_X.shape, gt_y.shape, X_eval.shape, y_eval.shape)

    mean_list = []
    var_list = []
    x_list = []

    for i, x in tqdm(enumerate(evalset)):
        input_x = torch.from_numpy(np.random.normal(loc=0, scale=1, size=(100, 1, 1))).float().to(device)

        position_encode = False
        m = 3
        # position encode
        if position_encode:
            condition_y = x[0].expand(100, 1, configs['model-structure']['input-dim'])
            x_p_list = [condition_y]
            for i in range(m):
                x_p_list.append(torch.sin((2**(i+1)) * condition_y))
                x_p_list.append(torch.cos((2**(i+1)) * condition_y))
            condition_y = torch.cat(x_p_list, dim=2)
        else:
            condition_y = x[0].expand(100, 1, configs['model-structure']['input-dim'])
        
        delta_p = torch.zeros(100, 1, configs['model-structure']['input-dim']).to(x[0])

        approx21, delta_log_p2 = prior(input_x, condition_y, delta_p, reverse=True)

        np_x = float(x[0].detach().cpu().numpy()[0])
        np_var = float(torch.var(approx21).detach().cpu().numpy())
        np_mean = float(torch.mean(approx21).detach().cpu().numpy())

        x_list.append(np_x)
        var_list.append(np_var)
        mean_list.append(np_mean)

    savePath = "var.png"
    visualize_uncertainty(savePath, gt_X.reshape(-1), gt_y.reshape(-1), x_list, mean_list, var_list)

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