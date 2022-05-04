#! /usr/bin/python3.7

import json
import argparse
import datetime
import time
from math import log, pi

import torch
import numpy as np
from torch.utils import data

from dun_datasets.utils import position_encode
from dun_datasets.UCI_gap_loader import *
from module.flow import cnf
from dun_metrics.regression_metrics import *


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
    parser = argparse.ArgumentParser(description="Flow tester")
    
    # config-files
    parser.add_argument("--cfg", required=True, type=str, help="path to the configs")
    # model load
    parser.add_argument("--load", default=None, type=str, help="load model from")
    # gpus
    parser.add_argument("--gpu", default=0, type=int, help="specify the gpu to be used")
    # save
    parser.add_argument("--save", default='./regression_result.csv', type=str, help="result file name")

    args = parser.parse_args()
    
    ''' Initialize all setting '''
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
    _, x, _, _, _, y, y_means, y_stds = load_gap_UCI(configs['dataset'], gap=False)
    x, y = x[:, None, :], y[:, None, :]
    # position encode
    if configs['position_encoding']:
        x = position_encode(x, axis=2)
    # create dataset and dataloader
    my_dataset = MyDataset(condition=torch.Tensor(x).cuda(args.gpu), generated_output=torch.tensor(y).float().cuda(args.gpu))
    test_loader = data.DataLoader(my_dataset, batch_size=128)

    
    def pred(prior, condition, sample_n=5):
        # get noise
        input_x = torch.from_numpy(np.random.normal(loc=0, scale=1, size=(sample_n*condition.size()[0], 1, 1))).float().cuda(args.gpu)
        # repeat n
        condition = condition.repeat(sample_n, 1, 1)
        # inference
        approx21 = prior(input_x, condition, None, reverse=True)
        # reshape
        approx21 = approx21.view(sample_n, -1)
        std = torch.sqrt(torch.var(approx21, dim=0, keepdim=True).transpose(0,1)) 
        mean = torch.mean(approx21, dim=0, keepdim=True).transpose(0,1)
        return mean, std

    print('***split and valprop should be specify***')
    output_str_list = ['method,epochs,dataset,split,n_samples,valprop,network,num,width,batch_size,ll,err,ece,tail_ece,batch_time']
    ''' Test Loop '''
    prior.eval()
    with torch.no_grad():
        for sample_n in [5, 10, 15, 20, 25, 30]:
            for num in range(5):
                # evaluate
                means, pred_stds, ys, times = [], [], [], []
                for condition, generated_output in test_loader:
                    tic = time.time()
                    mean, std = pred(prior, condition, sample_n)
                    toc = time.time()
                    
                    times.append(toc - tic)
                    means.append(mean)
                    pred_stds.append(std)
                    ys.append(generated_output.squeeze(dim=1))

                means = torch.cat(means, dim=0)
                pred_stds = torch.cat(pred_stds, dim=0)
                ys = torch.cat(ys, dim=0)

                batch_time = np.mean(times)
                
                means = means.cpu()
                pred_stds = pred_stds.cpu()
                ys = ys.cpu()
                try:
                    rms = get_rms(means, ys, y_means, y_stds)
                except Exception as e:
                    print('rms error')
                    print(e)
                    exit(0)

                ll = get_gauss_loglike(means, pred_stds, ys, y_means, y_stds)
                bin_probs, _, _, bin_counts, reference =\
                    gauss_callibration(means, pred_stds, ys, 10, cummulative=False, two_sided=False)
                ece = expected_callibration_error(bin_probs, reference, bin_counts)
                tail_ece = expected_callibration_error(bin_probs, reference, bin_counts, tail=True)

                # network&width just set to default for DUN plot program
                # 'method,epochs,dataset,split,n_samples,valprop,network,num,width,batch_size,ll,err,ece,tail_ece,batch_time'
                csv_row = 'Flow,120,'+configs['dataset']+',0,'+str(sample_n)+',0.15,ResNet,'+str(num)+',100,128,'\
                        +str(ll)+','+str(rms.item())+','+str(ece)+','+str(tail_ece)+','+str(batch_time)
                output_str_list.append(csv_row)

    csv_str = '\n'.join(output_str_list)

    with open(args.save, 'w') as fp:
        fp.write(csv_str)
