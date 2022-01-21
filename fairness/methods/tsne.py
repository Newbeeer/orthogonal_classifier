import torch
from adult import VAE,loss_BCE
from adult_data import create_torch_dataloader
from sklearn import manifold, datasets
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
plt.switch_backend('agg')



def test():
    #Since H(u) is a constant, we can calculate it at the end of all experiments
    best_F = 100
    fig = plt.figure(figsize=(8, 8))
    vae = VAE(z_dim=10)
    model_path = 'vaex_model_adult_ori2.pth.tar'
    vae.load_state_dict(torch.load(model_path))
    z_collect = []
    x_collect = []
    u_collect = []
    train_loader, test_loader = create_torch_dataloader(batch=64)
    for epoch in range(1):

        for iteration, (x, u, y) in enumerate(test_loader):
            x, u, y = x.cuda(), u.cuda(), y.cuda()
            recon_x, mean, log_var, z = vae(x, u)
            z_collect.append(mean.detach().cpu())
            u_collect.append(u.detach().cpu())
            x_collect.append(x.detach().cpu())

    x_ = torch.cat(x_collect, dim=0).numpy()
    z_ = torch.cat(z_collect,dim=0).numpy()
    u_ = torch.cat(u_collect,dim=0).numpy().squeeze(1)
    print("x shape: {},z shape:{}, u shape:{}".format(x_.shape,z_.shape,u_.shape))

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    Y = tsne.fit_transform(z_)
    np.save("tsne_ori_Y.npy",Y)
    ax = fig.add_subplot(2, 1, 2)

    plt.scatter(Y[:, 0], Y[:, 1], c=u_, cmap=plt.cm.Spectral)
    ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
    ax.yaxis.set_major_formatter(NullFormatter())
    # plt.axis('tight')

    #plt.savefig('tsne_x_kernel2.png')
    plt.savefig('tsne_x_ori2.png')
if __name__ == '__main__':
    test()