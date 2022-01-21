import torch
import torch.nn as nn
from german_data import create_torch_dataloader
import argparse
from sklearn.metrics import roc_auc_score
from kernel_regression import KernelRegression
import numpy as np
from utils import *

class Encoder(nn.Module):

    def __init__(self,z_dim=10):

        super().__init__()


        self.MLP = nn.Sequential(
            nn.Linear(58,50),
            nn.Softplus()
        )
        self.linear_means = nn.Linear(50, z_dim)
        self.linear_log_var = nn.Linear(50, z_dim)

    def forward(self, x, u=None):

        #x = torch.cat((x,u),dim=1)
        x = self.MLP(x) # q(z | x, u)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars

class Decoder(nn.Module):

    def __init__(self, z_dim =10):

        super().__init__()

        self.MLP = nn.Sequential(nn.Linear(z_dim,50),
        nn.Softplus(),
        nn.Linear(50,58),
        nn.Sigmoid())

    def forward(self, z, u=None):

        #z = torch.cat((z,u),dim=1)
        x = self.MLP(z)  # p(x | z, u)

        return x

class Logistic(nn.Module):

    def __init__(self):

        super().__init__()

        self.MLP = nn.Sequential(nn.Linear(10,1),
                                 nn.Sigmoid())


    def forward(self, z):

        x = self.MLP(z)  # p(u | z)
        return x
class Discriminator(nn.Module):

    def __init__(self, z_dim =10, latent_size = 50, relu = False):

        super().__init__()
        if relu:
            self.MLP = nn.Sequential(nn.Linear(z_dim,latent_size),
            nn.ReLU(),
            nn.Linear(latent_size,1),
            nn.Sigmoid())
        else:
            self.MLP = nn.Sequential(nn.Linear(z_dim, latent_size),
                                     nn.Softplus(),
                                     nn.Linear(latent_size, 1),
                                     nn.Sigmoid())

    def forward(self, z):

        x = self.MLP(z)  # p(u | z)
        return x

class Discriminator_depth(nn.Module):

    def __init__(self, z_dim =10, latent_size = 100,relu=False):

        super().__init__()
        if relu:
            self.MLP = nn.Sequential(nn.Linear(z_dim, latent_size),
                                     nn.ReLU(),
                                     nn.Linear(latent_size, 50),
                                     nn.ReLU(),
                                     nn.Linear(50, 1),
                                     nn.Sigmoid())
        else:
            self.MLP = nn.Sequential(nn.Linear(z_dim, latent_size),
                                     nn.LeakyReLU(0.2),
                                     nn.Linear(latent_size, 50),
                                     nn.LeakyReLU(0.2),
                                     nn.Linear(50, 1),
                                     nn.Sigmoid())


    def forward(self, z):

        x = self.MLP(z)  # p(u | z)
        return x

class Classifier(nn.Module):

    def __init__(self, z_dim =10):

        super().__init__()

        self.MLP = nn.Sequential(nn.Linear(z_dim, 50),
                                 nn.Softplus(),
                                 nn.Linear(50, 1),
                                 nn.Sigmoid())

    def forward(self, z):
        x = self.MLP(z)  # p(y | z)
        return x

class classifier_ori(nn.Module):
    def __init__(self):
        super().__init__()
        self.MLP = nn.Sequential(nn.Linear(58,50),
                                 nn.Softplus(),
                                 nn.Linear(50, 10),
                                 nn.Softplus(),
                                 nn.Linear(10, 1),
                                 nn.Sigmoid())
    def forward(self, z):
        x = self.MLP(z)  # p(y | z)
        return x

class VAE_x(nn.Module):

    def __init__(self,z_dim=10):

        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(z_dim=z_dim).cuda()
        self.decoder = Decoder(z_dim=z_dim).cuda()

    def forward(self, x,u, classifier=False):


        batch_size = x.size(0)
        means, log_var = self.encoder(x)
        if classifier:
            return means
        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.z_dim]).cuda()
        z = eps * std + means

        recon_x = self.decoder(z)

        return recon_x, means, log_var, z

    def inference(self, n=1, c=None):

        batch_size = n
        z = torch.randn([batch_size, self.latent_size]).cuda()
        recon_x = self.decoder(z, c)

        return recon_x
class VAE(nn.Module):

    def __init__(self,z_dim=10):

        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(z_dim=z_dim).cuda()
        self.decoder = Decoder(z_dim=z_dim).cuda()

    def forward(self, x,u, classifier=False):


        batch_size = x.size(0)
        means, log_var = self.encoder(x, u)
        if classifier:
            return means
        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.z_dim]).cuda()
        z = eps * std + means

        recon_x = self.decoder(z,u)

        return recon_x, means, log_var, z

    def inference(self, n=1, c=None):

        batch_size = n
        z = torch.randn([batch_size, self.latent_size]).cuda()
        recon_x = self.decoder(z, c)

        return recon_x

def loss_BCE(recon_x, x):

    BCE = torch.nn.functional.binary_cross_entropy(
            recon_x.view(-1, x.size(1)), x.view(-1, x.size(1)), size_average = False)

    return (BCE) / x.size(0)

def loss_KLD(mean, log_var):

    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return KLD / mean.size(0)

def train_vae(vae_model,F,train_loader,test_loader,args,latent_size):


    e1 = 1
    e2 = 1
    e3 = 10
    optimizer_vae = torch.optim.Adam(vae_model.parameters(), lr=args.learning_rate,betas=(0.5, 0.999))
    optimizer_F = torch.optim.Adam(F.parameters(), lr=args.learning_rate,betas=(0.5, 0.999))
    for epoch in range(args.epochs):
        train_loss_v = 0.0
        train_loss_F = 0.0
        for iteration, (x, u, y) in enumerate(train_loader):

            x, u, y = x.cuda(), u.cuda(), y.cuda()
            recon_x, mean, log_var, z = vae_model(x,u)
            recon_u = F(mean)

            loss = e1 * loss_BCE(recon_x, x) + e2 * loss_KLD(mean, log_var) - e3 * loss_BCE(recon_u,u)
            train_loss_v += loss.item()
            optimizer_vae.zero_grad()
            loss.backward()
            optimizer_vae.step()

            for i in range(1):
                recon_x, mean, log_var, z = vae_model(x, u)
                recon_u = F(mean)
                loss_F = loss_BCE(recon_u,u)
                train_loss_F += loss_F.item() * x.size(0)
                optimizer_F.zero_grad()
                loss_F.backward()
                optimizer_F.step()

        print("latent size : {}, epoch: {},  F loss : {}".format(latent_size,epoch,train_loss_F/len(train_loader.dataset)))

        if epoch % 50 == 0 and epoch!=0:
            torch.save(vae_model.state_dict(),
                       'vaex_model_depth_german_latent_' + str(latent_size) + str(epoch) + '.pth.tar')

        train_loss_F = 0.0
        correct = 0.0
        u_collect = []
        recon_u_collect = []
        for iteration, (x, u, y) in enumerate(test_loader):
            x, u, y = x.cuda(), u.cuda(), y.cuda()
            recon_x, mean, log_var, z = vae_model(x, u)
            recon_u = F(mean)

            loss_F = loss_BCE(recon_u, u)
            train_loss_F += loss_F.item() * x.size(0)
            pred = (recon_u > 0.5).float()
            correct += (pred == u).float().sum()
            u_collect.append(u.detach().cpu())
            recon_u_collect.append(recon_u.detach().cpu())

        u = torch.cat(u_collect, dim=0).numpy()
        recon_u = torch.cat(recon_u_collect, dim=0).numpy()
        test_auc = roc_auc_score(u, recon_u)
        print("Test: latent size : {}, F information : {}, acc:{}, auc:{}".format(latent_size, 0.631475 - train_loss_F / len(test_loader.dataset), correct/len(test_loader.dataset),test_auc))

    #torch.save(vae_model.state_dict(),'vaex_model_relu_german_latent_'+str(latent_size)+'.pth.tar')
    torch.save(vae_model.state_dict(), 'vaex_model_depth_german_latent_' + str(latent_size) + '.pth.tar')



def main(args):

    train_loader, test_loader = create_torch_dataloader(batch=64)
    #vae = VAE_x(z_dim=10)
    #F = Discriminator_depth(z_dim=10,latent_size=args.latent_size,relu=False).cuda()
    #3F = Discriminator(z_dim=10,latent_size=args.latent_size,relu=True).cuda()
    #train_vae(vae,F,train_loader,test_loader,args,args.latent_size)
    train_classifier_ori(args)
def train_classifier(args):
    train_loader, test_loader = create_torch_dataloader(batch=64)
    vae = VAE(z_dim=10)
    model_path = 'vae_model_german.pth.tar'
    vae.load_state_dict(torch.load(model_path))
    #F = Discriminator(z_dim=10).cuda()
    classifier = Classifier(z_dim=10).cuda()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate,betas=(0.5, 0.999))

    for epoch in range(args.epochs):
        train_loss = 0.0
        tcorrect = 0.0
        correct = 0.0
        for iteration, (x, u, y) in enumerate(train_loader):

            x, u, y = x.cuda(), u.cuda(), y.cuda().long()
            mean = vae(x,u,classifier=True)
            output = classifier(mean)
            pre = (output > 0.5).detach().long()
            tcorrect += pre.eq(y).sum().item()
            loss = loss_BCE(output, y.float())
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        for iteration, (x, u, y) in enumerate(test_loader):

            x, u, y = x.cuda(), u.cuda(), y.cuda().long()
            mean = vae(x,u,classifier=True)
            output = classifier(mean)

            pre = (output > 0.5).detach().long()
            correct += pre.eq(y).sum().item()

        print("Epoch:{}, train acc : {}, test acc : {}".format(epoch,tcorrect/len(train_loader.dataset),correct/len(test_loader.dataset)))
    torch.save(classifier.state_dict(), 'classifier_german.pth.tar')


def train_classifier_ori(args):

    train_loader, test_loader = create_torch_dataloader(batch=64)
    classifier = classifier_ori().cuda()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    p_y_u = np.zeros((2, 2))
    p_y = np.zeros(2)
    # calculate the conditional distribution
    for iteration, (x, u, y) in enumerate(train_loader):
        for u_, y_ in zip(u,y):
            p_y_u[int(u_), int(y_)] += 1
            p_y[int(y_)] += 1
    p_y_u /= p_y_u.sum(0, keepdims=True)
    p_y /= p_y.sum()
    print(p_y, p_y_u)

    for epoch in range(args.epochs):
        train_loss = 0.0
        tcorrect = 0.0
        correct = 0.0
        y_list_t = []
        y_hat_list_t = []
        u_list_t = []
        for iteration, (x, u, y) in enumerate(train_loader):

            x, u, y = x.cuda(), u.cuda(), y.cuda().long()

            output = classifier(x)
            pre = (output > 0.5).detach().long()
            loss = loss_BCE(output, y.float())
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.orth:
                output = output.detach().cpu().numpy()
                output = np.concatenate([1 - output, output], axis=1)
                w_1 = p_y_u[u.squeeze().cpu().numpy().astype(np.int32)]
                y_prior = p_y[y.squeeze().cpu().numpy().astype(np.int32)][:, None]
                output = y_prior * output / w_1
                output /= output.sum(1, keepdims=True)
                output = output[:, 1][:, None]
                pre = (output > 0.5).astype(np.float32)
                tcorrect += (pre == y.detach().cpu().numpy()).astype(np.float32).sum()
                #pre = np.random.randint(0, 2, pre.shape)
                y_hat_list_t.append(pre)
            else:
                tcorrect += pre.eq(y).sum().item()
                y_hat_list_t.append(pre.cpu().numpy())
            train_loss += loss.item()

            y_list_t.append(y.detach().cpu().numpy())
            u_list_t.append(u.detach().cpu().numpy())
        y_l_t = np.reshape(np.concatenate(y_list_t, axis=0), [-1])
        y_h_l_t = np.reshape(np.concatenate(y_hat_list_t, axis=0), [-1])
        u_l_t = np.reshape(np.concatenate(u_list_t, axis=0), [-1])

        y_list = []
        y_hat_list = []
        u_list = []
        for iteration, (x, u, y) in enumerate(test_loader):

            x, u, y = x.cuda(), u.cuda(), y.cuda().long()

            output = classifier(x)
            output = output.detach().cpu().numpy()
            if args.orth:
                output = np.concatenate([1 - output, output], axis=1)
                w_1 = p_y_u[u.squeeze().cpu().numpy().astype(np.int32)]
                y_prior = p_y[y.squeeze().cpu().numpy().astype(np.int32)][:, None]
                output = y_prior * output / w_1
                output /= output.sum(1, keepdims=True)
                output = output[:, 1][:, None]

            pre = (output > 0.5).astype(np.float32)
            correct += (pre == y.detach().cpu().numpy()).astype(np.float32).sum()

            y_list.append(y.detach().cpu().numpy())
            y_hat_list.append(pre)
            u_list.append(u.detach().cpu().numpy())

        y_l = np.reshape(np.concatenate(y_list, axis=0), [-1])
        y_h_l = np.reshape(np.concatenate(y_hat_list, axis=0), [-1])
        u_l = np.reshape(np.concatenate(u_list, axis=0), [-1])
        print("Epoch:{}, train acc : {}, test acc : {}".format(epoch,tcorrect/len(train_loader.dataset),correct/len(test_loader.dataset)))
        print("Train:  Equalized odds:", equalized_odds(y_l_t, y_h_l_t, u_l_t), "delta DP:", demographic_parity(y_h_l_t, u_l_t))
        print("Test:  Equalized odds:", equalized_odds(y_l, y_h_l, u_l), "delta DP:", demographic_parity(y_h_l, u_l))

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--latent_size", type=int)
    parser.add_argument("--ms", type=int)
    parser.add_argument("--orth", action='store_true', default=False)
    args = parser.parse_args()

    main(args)
