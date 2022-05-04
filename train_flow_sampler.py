import matplotlib.pyplot as plt

from dun_datasets.UCI_gap_loader import *
from dun_datasets.additional_gap_loader import *
from condition_sampler.flow_sampler import FlowSampler


# load data
x, _, _, _, _, _, _, _ = load_gap_UCI('concrete', gap=False)

'''
# draw data dis
plt.xlim([0, 6])
plt.ylim([0, 3])
plt.plot(x, y, 'ok', ms=1)
plt.savefig('data_dis.png')
plt.clf()
exit()
'''

x = x[:, None, :]
x_sampler = FlowSampler(x.shape[1:], '256-256-256', 1, gpu=3)
loss = x_sampler.fit(x, batch=64, lr=5e-3, epoch=800)

# draw loss
plt.plot([i for i in range(len(loss))], loss)
plt.savefig('loss.png')
plt.clf()

'''
# sample test
sampled_x = x_sampler.sample(x.shape[0]*3).squeeze().cpu().numpy()
plt.hist(sampled_x, bins=32*2, color='g')
plt.hist(x.squeeze(), bins=32, color='r')
plt.savefig('sample_result.png')
'''
x_sampler.save('project/regression/trained_sampler/concrete_sampler.pt')
