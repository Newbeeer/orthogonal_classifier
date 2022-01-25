import torch
import torch.nn as nn
import argparse
import numpy as np
from utils import *

class lambda_mi(nn.Module):

    def __init__(self):

        super().__init__()

        self.lambda_mi = nn.Parameter(torch.ones(3, requires_grad=True))

class Encoder(nn.Module):

    def __init__(self,z_dim=10):

        super().__init__()
        if args.data == 'adult':
            in_dim = 102
        elif args.data == 'german':
            in_dim = 58
        if args.mifr:
            in_dim += 1
        self.MLP = nn.Sequential(
            nn.Linear(in_dim, 50),
            nn.Softplus(),
            # nn.Linear(50, z_dim),
            # nn.Softplus(),
        )
        self.linear_means = nn.Linear(50, z_dim)
        self.linear_log_var = nn.Linear(50, z_dim)
    def forward(self, x, u=None):
        if u is not None:
            x = torch.cat((x, u), dim=1)
            x = self.MLP(x)
            means = self.linear_means(x)
            log_vars = self.linear_log_var(x)
            return means, log_vars

        x = self.MLP(x)
        z = self.linear_means(x)

        return z

class Decoder(nn.Module):

    def __init__(self, z_dim=10):

        super().__init__()
        if args.data == 'adult':
            out_dim = 102
        elif args.data == 'german':
            out_dim = 58
        self.MLP = nn.Sequential(nn.Linear(z_dim+1, 50),
        nn.Softplus(),
        nn.Linear(50, out_dim),
        nn.Sigmoid())

    def forward(self, z, u):

        z = torch.cat((z, u), dim=1)
        x = self.MLP(z)  # k(x | z, u)

        return x

class classifier_fc(nn.Module):

    def __init__(self, z_dim=10):

        super().__init__()

        self.MLP = nn.Sequential(nn.Linear(z_dim, 50),
                                 nn.Softplus(),
                                 nn.Linear(50, 1),
                                 nn.Sigmoid())

    def forward(self, z):

        x = self.MLP(z)
        return x

class classifier_ori(nn.Module):
    def __init__(self):
        super().__init__()
        if args.data == 'adult':
            z_dim = 102
        elif args.data == 'german':
            z_dim = 58
        self.MLP = nn.Sequential(nn.Linear(z_dim, 50),
                                 nn.Softplus(),
                                 nn.Linear(50, 10),
                                 nn.Linear(10, 50),
                                 nn.Softplus(),
                                 nn.Linear(50, 1),
                                 # nn.Softplus(),
                                 # nn.Linear(10, 1),
                                 nn.Sigmoid())
    def forward(self, z):
        x = self.MLP(z)  # p(y | z)
        return x

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

def train_vae(args):
    best_test_acc = 0.
    eo = 0.
    dp = 0.
    if args.data == 'adult':
        from adult_data import create_torch_dataloader
    elif args.data == 'german':
        from german_data import create_torch_dataloader
    train_loader, test_loader = create_torch_dataloader(batch=64)
    e1 = 10
    e2 = args.gamma
    print(f"e1:{e1}, e2:{e2}")
    vae_model = VAE(z_dim=10)
    adv = classifier_fc(z_dim=10).cuda()
    classifier = classifier_fc(z_dim=10).cuda()
    l_mi = lambda_mi().cuda()
    optimizer_vae = torch.optim.Adam(vae_model.parameters(), lr=args.learning_rate,betas=(0.5, 0.999))
    optimizer_adv = torch.optim.Adam(adv.parameters(), lr=args.learning_rate,betas=(0.5, 0.999))
    optimizer_c = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate,betas=(0.5, 0.999))
    optimizer_l = torch.optim.Adam(l_mi.parameters(), lr=args.learning_rate,betas=(0.5, 0.999))
    for epoch in range(args.epochs):
        train_loss_v = 0.0
        tcorrect = 0.0
        correct = 0.0
        train_loss = 0.0
        for iteration, (x, u, y) in enumerate(train_loader):

            x, u, y = x.cuda(), u.cuda(), y.cuda()

            # update feature
            recon_x, mean, log_var, z = vae_model(x,u)
            recon_u = adv(mean)
            loss = loss_BCE(recon_x, x) + l_mi.lambda_mi[1] * (loss_KLD(mean, log_var) - e1) - l_mi.lambda_mi[2] * (loss_BCE(recon_u,u) - e2)
            train_loss_v += loss.item()
            optimizer_vae.zero_grad()
            loss.backward()
            optimizer_vae.step()

            # update lambda
            recon_x, mean, log_var, z = vae_model(x, u)
            recon_u = adv(mean)
            loss = -(l_mi.lambda_mi[1] * (loss_KLD(mean, log_var) - e1) + l_mi.lambda_mi[2] * (
                        loss_BCE(recon_u, u) - e2))
            optimizer_l.zero_grad()
            loss.backward()
            optimizer_l.step()

            # update adversary
            for i in range(1):
                recon_x, mean, log_var, z = vae_model(x, u)
                recon_u = adv(mean)
                loss_adv = loss_BCE(recon_u, u)
                optimizer_adv.zero_grad()
                loss_adv.backward()
                optimizer_adv.step()

            # update fc
            z = vae_model(x, u, classifier=True)
            y_output = classifier(z)

            loss = loss_BCE(y_output, y.float())
            optimizer_c.zero_grad()
            loss.backward()
            optimizer_c.step()

            l_mi.lambda_mi.data = torch.clamp(l_mi.lambda_mi.data, 0)
            pre = (y_output > 0.5).detach().long()
            tcorrect += pre.eq(y).sum().item()
            train_loss += loss.item()

        y_list = []
        y_hat_list = []
        u_list = []
        for iteration, (x, u, y) in enumerate(test_loader):

            x, u, y = x.cuda(), u.cuda(), y.cuda().long()
            output = classifier(vae_model(x, u, classifier=True))
            output = output.detach().cpu().numpy()

            pre = (output > 0.5).astype(np.float32)
            correct += (pre == y.detach().cpu().numpy()).astype(np.float32).sum()

            y_list.append(y.detach().cpu().numpy())
            y_hat_list.append(pre)
            u_list.append(u.detach().cpu().numpy())

        y_l = np.reshape(np.concatenate(y_list, axis=0), [-1])
        y_h_l = np.reshape(np.concatenate(y_hat_list, axis=0), [-1])
        u_l = np.reshape(np.concatenate(u_list, axis=0), [-1])
        print("Epoch:{}, train acc : {}, test acc : {}".format(epoch, tcorrect / len(train_loader.dataset),correct / len(test_loader.dataset)))
        print("Test:  Equalized odds:", equalized_odds(y_l, y_h_l, u_l), "delta DP:", demographic_parity(y_h_l, u_l))
        print("lambdas:", l_mi.lambda_mi.data)

        if correct / len(test_loader.dataset) > best_test_acc:
            best_test_acc = correct/len(test_loader.dataset)
            eo = equalized_odds(y_l, y_h_l, u_l)
            dp = demographic_parity(y_h_l, u_l)
            print("Best ----- ")
            print(f"test acc:{best_test_acc}, eo :{eo}, dp:{dp}")
    print("Best ----- ")
    print(f"test acc:{best_test_acc}, eo :{eo}, dp:{dp}")

def main(args):

    if args.laftr:
        train_classifier_laftr(args)
    elif args.mifr:
        train_vae(args)
    elif args.hsic:
        train_hsic(args)
    else:
        train_classifier_ori(args)

def train_hsic(args):

    from hsic import RbfHSIC
    best_test_acc = 0.
    eo = 0.
    dp = 0.
    # b
    print("HSIC gamma:{}. sigma:{}".format(args.gamma, args.sigma))
    if args.data == 'adult':
        from adult_data import create_torch_dataloader
    elif args.data == 'german':
        from german_data import create_torch_dataloader
    train_loader, test_loader = create_torch_dataloader(batch=64)
    hsic = RbfHSIC(args.sigma)
    encoder = Encoder(z_dim=10).cuda()
    decoder = Decoder(z_dim=10).cuda()
    classifier = classifier_fc(z_dim=10).cuda()

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()) + list(decoder.parameters()), lr=args.learning_rate,betas=(0.5, 0.999))

    for epoch in range(args.epochs):
        train_loss = 0.0
        tcorrect = 0.0
        correct = 0.0
        for iteration, (x, u, y) in enumerate(train_loader):

            x, u, y = x.cuda(), u.cuda(), y.cuda().long()

            # update backbone
            z = encoder(x)
            y_output = classifier(z)
            x_output = decoder(z, u)

            loss = loss_BCE(y_output, y.float()) + args.gamma * hsic(z, u)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pre = (y_output > 0.5).detach().long()
            tcorrect += pre.eq(y).sum().item()
            train_loss += loss.item()

        y_list = []
        y_hat_list = []
        u_list = []
        for iteration, (x, u, y) in enumerate(test_loader):

            x, u, y = x.cuda(), u.cuda(), y.cuda().long()
            output = classifier(encoder(x))
            output = output.detach().cpu().numpy()

            pre = (output > 0.5).astype(np.float32)
            correct += (pre == y.detach().cpu().numpy()).astype(np.float32).sum()

            y_list.append(y.detach().cpu().numpy())
            y_hat_list.append(pre)
            u_list.append(u.detach().cpu().numpy())

        y_l = np.reshape(np.concatenate(y_list, axis=0), [-1])
        y_h_l = np.reshape(np.concatenate(y_hat_list, axis=0), [-1])
        u_l = np.reshape(np.concatenate(u_list, axis=0), [-1])
        print("Epoch:{}, train acc : {}, test acc : {}".format(epoch, tcorrect / len(train_loader.dataset),correct / len(test_loader.dataset)))
        print("Test:  Equalized odds:", equalized_odds(y_l, y_h_l, u_l), "delta DP:", demographic_parity(y_h_l, u_l))

        if correct/len(test_loader.dataset) > best_test_acc:
            best_test_acc = correct/len(test_loader.dataset)
            eo = equalized_odds(y_l, y_h_l, u_l)
            dp = demographic_parity(y_h_l, u_l)
            print("Best ----- ")
            print(f"test acc:{best_test_acc}, eo :{eo}, dp:{dp}")
    print("Final Best ----- ")
    print(f"test acc:{best_test_acc}, eo :{eo}, dp:{dp}")

def train_classifier_laftr(args):
    best_test_acc = 0.
    eo = 0.
    dp = 0.
    print("LAFTR gamma:", args.gamma)
    if args.data == 'adult':
        from adult_data import create_torch_dataloader
    elif args.data == 'german':
        from german_data import create_torch_dataloader
    train_loader, test_loader = create_torch_dataloader(batch=64)
    encoder = Encoder(z_dim=10).cuda()
    decoder = Decoder(z_dim=10).cuda()
    adv = classifier_fc(z_dim=10).cuda()
    classifier = classifier_fc(z_dim=10).cuda()

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()) + list(decoder.parameters()), lr=args.learning_rate,betas=(0.5, 0.999))
    optimizer_adv = torch.optim.Adam(adv.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

    for epoch in range(args.epochs):
        train_loss = 0.0
        tcorrect = 0.0
        correct = 0.0
        for iteration, (x, u, y) in enumerate(train_loader):

            x, u, y = x.cuda(), u.cuda(), y.cuda().long()

            # update adv
            z = encoder(x)
            u_output = adv(z)

            loss_adv = loss_eo(u_output, u, y)
            optimizer_adv.zero_grad()
            loss_adv.backward()
            optimizer_adv.step()

            # update backbone
            z = encoder(x)
            y_output = classifier(z)
            u_output = adv(z)
            x_output = decoder(z, u)

            loss = loss_BCE(y_output, y.float()) - args.gamma * loss_eo(u_output, u, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pre = (y_output > 0.5).detach().long()
            tcorrect += pre.eq(y).sum().item()
            train_loss += loss.item()


        y_list = []
        y_hat_list = []
        u_list = []
        for iteration, (x, u, y) in enumerate(test_loader):

            x, u, y = x.cuda(), u.cuda(), y.cuda().long()
            output = classifier(encoder(x))
            output = output.detach().cpu().numpy()

            pre = (output > 0.5).astype(np.float32)
            correct += (pre == y.detach().cpu().numpy()).astype(np.float32).sum()

            y_list.append(y.detach().cpu().numpy())
            y_hat_list.append(pre)
            u_list.append(u.detach().cpu().numpy())

        y_l = np.reshape(np.concatenate(y_list, axis=0), [-1])
        y_h_l = np.reshape(np.concatenate(y_hat_list, axis=0), [-1])
        u_l = np.reshape(np.concatenate(u_list, axis=0), [-1])
        print("Epoch:{}, train acc : {}, test acc : {}".format(epoch, tcorrect / len(train_loader.dataset),correct / len(test_loader.dataset)))
        print("Test:  Equalized odds:", equalized_odds(y_l, y_h_l, u_l), "delta DP:", demographic_parity(y_h_l, u_l))

        if correct/len(test_loader.dataset) > best_test_acc:
            best_test_acc = correct/len(test_loader.dataset)
            eo = equalized_odds(y_l, y_h_l, u_l)
            dp = demographic_parity(y_h_l, u_l)
            print("Best ----- ")
            print(f"test acc:{best_test_acc}, eo :{eo}, dp:{dp}")
    print("Best ----- ")
    print(f"test acc:{best_test_acc}, eo :{eo}, dp:{dp}")

def train_classifier_ori(args):
    best_test_acc = 0.
    eo = 0.
    dp = 0.
    if args.data == 'adult':
        from adult_data import create_torch_dataloader
    elif args.data == 'german':
        from german_data import create_torch_dataloader
    train_loader, test_loader = create_torch_dataloader(batch=64)
    encoder = Encoder(z_dim=10).cuda()
    classifier = classifier_fc(z_dim=10).cuda()
    optimizer = torch.optim.Adam(list(classifier.parameters())+list(encoder.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
    p_y_u = np.zeros((2, 2))
    p_y = np.zeros(2)
    # calculate the conditional distribution
    for iteration, (x, u, y) in enumerate(train_loader):
        for u_, y_ in zip(u,y):
            p_y_u[int(u_), int(y_)] += 1
            p_y[int(y_)] += 1
    p_y_u /= p_y_u.sum(1, keepdims=True)
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

            output = classifier(encoder(x))
            pre = (output > 0.5).detach().long()
            loss = loss_BCE(output, y.float())
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            y_hat_list_t.append(pre.cpu().numpy())
            tcorrect += pre.eq(y).sum().item()
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
            output = classifier(encoder(x))

            output = output.detach().cpu().numpy()
            if args.orthogonal:
                output = np.concatenate([1-output, output], axis=1)
                w_1 = p_y_u[u.squeeze().cpu().numpy().astype(np.int32)]
                y_prior = p_y[y.squeeze().cpu().numpy().astype(np.int32)][:, None]
                output = y_prior * output / w_1
                output /= output.sum(1, keepdims=True)
                output = output[:, 1][:, None]

            pre = (output > 0.5).astype(np.float32)
            # pre = np.random.randint(0, 2, pre.shape)
            correct += (pre == y.detach().cpu().numpy()).astype(np.float32).sum()

            y_list.append(y.detach().cpu().numpy())
            y_hat_list.append(pre)
            u_list.append(u.detach().cpu().numpy())

        y_l = np.reshape(np.concatenate(y_list, axis=0), [-1])
        y_h_l = np.reshape(np.concatenate(y_hat_list, axis=0), [-1])
        u_l = np.reshape(np.concatenate(u_list, axis=0), [-1])
        print("Epoch:{}, train acc : {}, test acc : {}".format(epoch, tcorrect/len(train_loader.dataset),correct/len(test_loader.dataset)))
        print("Test:  Equalized odds:", equalized_odds(y_l, y_h_l, u_l), "delta DP:", demographic_parity(y_h_l, u_l))
        if correct/len(test_loader.dataset) > best_test_acc:
            best_test_acc = correct/len(test_loader.dataset)
            eo = equalized_odds(y_l, y_h_l, u_l)
            dp = demographic_parity(y_h_l, u_l)
            print("Best ----- ")
            print(f"test acc:{best_test_acc}, eo :{eo}, dp:{dp}")
    print("Best ----- ")
    print(f"test acc:{best_test_acc}, eo :{eo}, dp:{dp}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data", type=str, default='adult', choices=['adult', 'german'], help='name of dataset')
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--orthogonal", action='store_true', default=False, help='classifier orthogonalization')
    parser.add_argument("--laftr", action='store_true', default=False)
    parser.add_argument("--mifr", action='store_true', default=False)
    parser.add_argument("--hsic", action='store_true', default=False)
    parser.add_argument("--gamma", type=float, default=5)
    parser.add_argument("--sigma", type=float, default=1)
    args = parser.parse_args()
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))
    main(args)