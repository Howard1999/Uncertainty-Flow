import torch
import matplotlib.pyplot as plt

from dun_datasets.additional_gap_loader import *
from my_datasets.toy_1d_data import *
from condition_sampler.flow_sampler import FlowSampler


# load data
x, y = load_compose_1d()

'''
# draw data dis
plt.xlim([0, 6])
plt.ylim([0, 3])
plt.plot(x, y, 'ok', ms=1)
plt.savefig('data_dis.png')
plt.clf()
exit()
'''

x, y = x[:, None, :], y[:, None, :]

x_sampler = FlowSampler((1, 1), '128-128', 1)
loss = x_sampler.fit(x, batch=128, lr=5e-3, epoch=200)

# draw loss
plt.plot([i for i in range(len(loss))], loss)
plt.savefig('loss.png')
plt.clf()

# sample test
sampled_x = x_sampler.sample(x.shape[0]*3).squeeze().cpu().numpy()
plt.hist(sampled_x, bins=32*2, color='g')
plt.hist(x.squeeze(), bins=32, color='r')
plt.savefig('sample_result.png')

x_sampler.save('project/toy-v1/trained_sampler/compose_sampler.pt')
