# # requirements
import torch
import torch.nn as nn
import torchvision
import numpy as np
import logging
import os
# colab requirements
# from IPython.display import clear_output
from time import sleep

if not os.path.exists('Variational_GON_Loggers'):
    os.mkdir('Variational_GON_Loggers')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s | | %(levelname)s | | %(message)s')

# logger_file_name = 'WSMBSS_Logger_Sparse1'
# logger_file_name = input('Enter the Logger File Name: ')
logger_file_name = 'MNIST_Train_Logger'
logger_file_name = os.path.join('Variational_GON_Loggers', logger_file_name)
file_handler = logging.FileHandler(logger_file_name,'w')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

logger.info('Code started \n')

# image data
dataset_name = 'mnist' # ['mnist', 'fashion']
img_size = 32
n_channels = 1
img_coords = 2

# training info
n_epochs = 1000
lr = 2*1e-4
batch_size = 64
nz = 256
ngf = 16
nc = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info("Using device: {}".format(str(device)))
logger.info("Training on {} is starting".format(dataset_name))
logger.info("We now give the hyperparameters")
logger.info("Learning rate : {}".format(lr))
logger.info("nz : {}".format(nz))
logger.info("ngf : {}".format(ngf))
logger.info("nc (Number of Channel)".format(nc))
logger.info("Number of Epochs : {}".format(n_epochs))

# create GON network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc21 = nn.Linear(nz, nz)
        self.fc22 = nn.Linear(nz, nz)
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=True),
            nn.BatchNorm2d(ngf * 4),
            nn.ELU(),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 2),
            nn.ELU(),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf),
            nn.ELU(),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=True),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, input):
        mu = self.fc21(input)
        logvar = self.fc22(input)
        z = self.reparameterize(mu, logvar)
        return self.main(z.unsqueeze(-1).unsqueeze(-1)), mu, logvar
    
    def sample(self, batch_size):
        z = torch.randn(batch_size, nz, 1, 1).cuda()
        return self.main(z)

def vae_loss(p, x, mu, logvar, weight=0.0):
    BCE = torch.nn.functional.binary_cross_entropy(p.view(-1, 32 * 32 * nc), x.view(-1, 32 * 32 * nc), reduction='none').sum(1).mean()
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    return BCE + (KLD * weight), BCE, KLD

# helper functions
def cycle(iterable):
    """
    You can use this function with a Pytorch DataLoader as the following.
    
    train_loader = torch.utils.data.DataLoader(dataset, sampler=None, shuffle=True, batch_size=batch_size, drop_last=True)
    train_iterator = iter(cycle(train_loader))
    x, t = next(train_iterator)
    """
    while True:
        for x in iterable:
            yield x
            
# load datasets
if dataset_name == 'mnist':
    dataset_trn = torchvision.datasets.MNIST('data', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(32), torchvision.transforms.ToTensor()
    ]))
    dataset_tst = torchvision.datasets.MNIST('data', train=False, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(32), torchvision.transforms.ToTensor()
    ]))

if dataset_name == 'fashion':
    dataset_trn = torchvision.datasets.FashionMNIST('data', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(32), torchvision.transforms.ToTensor()
    ]))
    dataset_tst = torchvision.datasets.FashionMNIST('data', train=False, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(32), torchvision.transforms.ToTensor()
    ]))

trn_loader = torch.utils.data.DataLoader(dataset_trn, sampler=None, shuffle=True, batch_size=batch_size, drop_last=True)
tst_loader = torch.utils.data.DataLoader(dataset_tst, sampler=None, shuffle=True, batch_size=batch_size, drop_last=True)

F = Generator().to(device)

optim = torch.optim.Adam(lr=lr, params=F.parameters())
logger.info(f'> Number of parameters {len(torch.nn.utils.parameters_to_vector(F.parameters()))}')
# print(f'> Number of parameters {len(torch.nn.utils.parameters_to_vector(F.parameters()))}')

logger.info("\n")
epoch_mse_loss_trn = 0
epoch_sse_loss_trn = 0
epoch_vae_loss_trn = 0
epoch = 0
for i,(x,t) in enumerate(trn_loader):
    x = x.to(device)
    batch_size_ = x.shape[0]
    z = torch.zeros(batch_size_, nz).to(device).requires_grad_()
    g, mu, logvar = F(z)
    L_inner, BCE, KLD = vae_loss(g, x, mu, logvar, 1.0)
    grad = torch.autograd.grad(L_inner, [z], create_graph=True, retain_graph=True)[0]
    z = (-grad)#.detach()

    # now with z as our new latent points, optimise the data fitting loss
    g, mu, logvar = F(z)
    L_outer, BCE, KLD = vae_loss(g, x, mu, logvar, 1.0)
    mse = ((g - x)**2).sum(1).mean()
    sse = ((g - x)**2).sum() / x.shape[0]
    epoch_mse_loss_trn += mse.item()
    epoch_sse_loss_trn += sse.item() 
    epoch_vae_loss_trn += L_outer.item()
#     for w in F.parameters():
#         w.grad.data.zero_()
        
# print(f"Epoch: {epoch}   Train Mean Squared Loss: {epoch_mse_loss_trn/i:.7f}")
# print(f"Epoch: {epoch}   Train Sum Squared Loss: {epoch_sse_loss_trn/i:.7f}")
# print(f"Epoch: {epoch}   Train VAE Loss: {epoch_vae_loss_trn/i:.7f}")

logger.info(f"Epoch: {epoch}   Train Mean Squared Loss: {epoch_mse_loss_trn/i:.7f}")
logger.info(f"Epoch: {epoch}   Train Sum Squared Loss: {epoch_sse_loss_trn/i:.7f}")
logger.info(f"Epoch: {epoch}   Train VAE Loss: {epoch_vae_loss_trn/i:.7f}")

epoch_mse_loss_tst = 0
epoch_sse_loss_tst = 0
epoch_vae_loss_tst = 0
for j,(x,t) in enumerate(tst_loader):
    x = x.to(device)
    batch_size_ = x.shape[0]
    z = torch.zeros(batch_size_, nz).to(device).requires_grad_()
    g, mu, logvar = F(z)
    L_inner, BCE, KLD = vae_loss(g, x, mu, logvar, 1.0)
    grad = torch.autograd.grad(L_inner, [z], create_graph=True, retain_graph=True)[0]
    z = (-grad)#.detach()

    # now with z as our new latent points, optimise the data fitting loss
    g, mu, logvar = F(z)
    L_outer, BCE, KLD = vae_loss(g, x, mu, logvar, 1.0)
    mse = ((g - x)**2).sum(1).mean()
    sse = ((g - x)**2).sum() / x.shape[0]
    epoch_mse_loss_tst += mse.item()
    epoch_sse_loss_tst += sse.item() 
    epoch_vae_loss_tst += L_outer.item()
#     for w in F.parameters():
#         w.grad.data.zero_()
        
# print(f"Epoch: {epoch}   Test Mean Squared Loss: {epoch_mse_loss_tst/j:.7f}")
# print(f"Epoch: {epoch}   Test Sum Squared Loss: {epoch_sse_loss_tst/j:.7f}")
# print(f"Epoch: {epoch}   Test VAE Loss: {epoch_vae_loss_tst/j:.7f}")

logger.info(f"Epoch: {epoch}   Test Mean Squared Loss: {epoch_mse_loss_tst/j:.7f}")
logger.info(f"Epoch: {epoch}   Test Sum Squared Loss: {epoch_sse_loss_tst/j:.7f}")
logger.info(f"Epoch: {epoch}   Test VAE Loss: {epoch_vae_loss_tst/j:.7f}")

for epoch_ in range(n_epochs):
    step = 0
    for (x,t) in trn_loader:
        # sample a batch of data
        # x, t = next(train_iterator)
        x = x.to(device)
        batch_size_ = x.shape[0]
        # compute the gradients of the inner loss with respect to zeros (gradient origin)
        z = torch.zeros(batch_size_, nz).to(device).requires_grad_()
        g, mu, logvar = F(z)
        L_inner, BCE, KLD = vae_loss(g, x, mu, logvar, 1.0)
        grad = torch.autograd.grad(L_inner, [z], create_graph=True, retain_graph=True)[0]
        z = (-grad)#.detach()

        # now with z as our new latent points, optimise the data fitting loss
        g, mu, logvar = F(z)
        L_outer, BCE, KLD = vae_loss(g, x, mu, logvar, 1.0)
        optim.zero_grad()
        L_outer.backward()
        optim.step()
        step += 1

    epoch_mse_loss_trn = 0
    epoch_sse_loss_trn = 0
    epoch_vae_loss_trn = 0
    for i,(x,t) in enumerate(trn_loader):
        # sample a batch of data
        # x, t = next(train_iterator)
        x = x.to(device)
        batch_size_ = x.shape[0]
        z = torch.zeros(batch_size_, nz).to(device).requires_grad_()
        g, mu, logvar = F(z)
        L_inner, BCE, KLD = vae_loss(g, x, mu, logvar, 1.0)
        grad = torch.autograd.grad(L_inner, [z], create_graph=True, retain_graph=True)[0]
        z = (-grad)#.detach()
        
        # now with z as our new latent points, optimise the data fitting loss
        g, mu, logvar = F(z)
        L_outer, BCE, KLD = vae_loss(g, x, mu, logvar, 1.0)
        mse = ((g - x)**2).sum(1).mean()
        sse = ((g - x)**2).sum() / x.shape[0]
        epoch_mse_loss_trn += mse.item()
        epoch_sse_loss_trn += sse.item() 
        epoch_vae_loss_trn += L_outer.item()
        for w in F.parameters():
            w.grad.data.zero_()

#     torch.cuda.empty_cache()

    epoch_mse_loss_tst = 0 # Mean Squared Error, initially zero
    epoch_sse_loss_tst = 0 # Sum Squared Error, initially zero
    epoch_vae_loss_tst = 0
    for j,(x,t) in enumerate(tst_loader): # Run over the test set
        x = x.to(device)
        batch_size_ = x.shape[0]
        z = torch.zeros(batch_size_, nz).to(device).requires_grad_()
        g, mu, logvar = F(z)
        L_inner, BCE, KLD = vae_loss(g, x, mu, logvar, 1.0)
        grad = torch.autograd.grad(L_inner, [z], create_graph=True, retain_graph=True)[0]
        z = (-grad)#.detach()
        
        # now with z as our new latent points, optimise the data fitting loss
        g, mu, logvar = F(z)
        L_outer, BCE, KLD = vae_loss(g, x, mu, logvar, 1.0)
        mse = ((g - x)**2).sum(1).mean()
        sse = ((g - x)**2).sum() / x.shape[0]
        epoch_mse_loss_tst += mse.item()
        epoch_sse_loss_tst += sse.item() 
        epoch_vae_loss_tst += L_outer.item()
        for w in F.parameters(): # Make all the grads after inference (since they do not contribute to training)
            w.grad.data.zero_()

    # print(f"\nEpoch: {epoch_ + 1}   Train Mean Squared Loss: {epoch_mse_loss_trn/i:.7f}")
    # print(f"Epoch: {epoch_ + 1}   Train Sum Squared Loss: {epoch_sse_loss_trn/i:.7f}")
    # print(f"Epoch: {epoch_ + 1}   Train VAE Loss: {epoch_vae_loss_trn/i:.7f}")
    # print(f"Epoch: {epoch_ + 1}   Test Mean Squared Loss: {epoch_mse_loss_tst/j:.7f}")
    # print(f"Epoch: {epoch_ + 1}   Test Sum Squared Loss: {epoch_sse_loss_tst/j:.7f}")
    # print(f"Epoch: {epoch_ + 1}   Test VAE Loss: {epoch_vae_loss_tst/j:.7f}")
    
    logger.info("\n")
    logger.info(f"Epoch: {epoch_ + 1}   Train Mean Squared Loss: {epoch_mse_loss_trn/i:.7f}")
    logger.info(f"Epoch: {epoch_ + 1}   Train Sum Squared Loss: {epoch_sse_loss_trn/i:.7f}")
    logger.info(f"Epoch: {epoch_ + 1}   Train VAE Loss: {epoch_vae_loss_trn/i:.7f}")
    logger.info(f"Epoch: {epoch_ + 1}   Test Mean Squared Loss: {epoch_mse_loss_tst/j:.7f}")
    logger.info(f"Epoch: {epoch_ + 1}   Test Sum Squared Loss: {epoch_sse_loss_tst/j:.7f}")
    logger.info(f"Epoch: {epoch_ + 1}   Test VAE Loss: {epoch_vae_loss_tst/j:.7f}")