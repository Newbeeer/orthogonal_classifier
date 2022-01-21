import torch
import numpy as np
from adult_data import create_torch_dataloader
from adult import VAE
from scipy.stats import gaussian_kde


def mutual_information_q_u(args):

    z_collect = [[],[]]
    u_collect = []
    z_all = []
    best_F = 100
    vae = VAE(z_dim=10)
    model_path = 'vae_model_adult_latent_'+str(args.latent_size)+'.pth.tar'
    #model_path = 'vae_model_adult_kernel.pth.tar'
    vae.load_state_dict(torch.load(model_path))


    train_loader, test_loader = create_torch_dataloader(batch=64)
    for iteration, (x, u, y) in enumerate(train_loader):
        x, u, y = x.cuda(), u.cuda(), y.cuda()
        recon_x, mean, log_var, z = vae(x, u)
        u_collect.append(u.cpu())
        u = u.detach().cpu().numpy()
        mean = mean.detach().cpu().numpy()
        z_collect[0].append(mean[np.nonzero(1 - u)[0]])
        z_collect[1].append(mean[np.nonzero(u)[0]])
        z_all.append(mean)


    z_collect[0] = np.concatenate(z_collect[0], axis=0)

    z_collect[1] = np.concatenate(z_collect[1], axis=0)
    z_all = np.concatenate(z_all,axis=0)
    print(z_collect[0].shape,z_collect[1].shape,z_all.shape)
    u_ = torch.cat(u_collect,dim=0).numpy().squeeze(1)
    kde = [gaussian_kde(z_collect[0].transpose()), gaussian_kde(z_collect[1].transpose())]
    kde[0].set_bandwidth('silverman')
    kde[1].set_bandwidth('silverman')
    kde_all = gaussian_kde(z_all.transpose())
    kde_all.set_bandwidth('silverman')
    mi_zu = 0.0
    mi_z = 0.0
    cnt = 0.0
    for iteration, (x, u, y) in enumerate(test_loader):
        x, u, y = x.cuda(), u.cuda(), y.cuda()
        recon_x, mean, log_var, z = vae(x, u)
        u = u.detach().cpu().numpy()
        idx = [np.nonzero(1.0 - u)[0], np.nonzero(u)[0]]
        mean = mean.detach().cpu().numpy()
        mi_zu += kde[0].logpdf(mean[idx[0]].transpose()).sum()
        mi_zu += kde[1].logpdf(mean[idx[1]].transpose()).sum()
        mi_z += kde_all.logpdf(mean.transpose()).sum()
        cnt += x.size(0)

    print(mi_z/cnt,mi_zu/cnt)

    mi = (mi_zu - mi_z)/cnt
    print("I:",mi)