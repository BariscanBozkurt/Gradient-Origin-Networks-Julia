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


if not os.path.exists('GON_Loggers'):
    os.mkdir('GON_Loggers')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s | | %(levelname)s | | %(message)s')

# logger_file_name = 'WSMBSS_Logger_Sparse1'
# logger_file_name = input('Enter the Logger File Name: ')
logger_file_name = 'CIFAR_Train_Logger'
logger_file_name = os.path.join('GON_Loggers', logger_file_name)
file_handler = logging.FileHandler(logger_file_name,'w')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

logger.info('Code started \n')

# image data
dataset_name = 'cifar' # ['mnist', 'fashion']
img_size = 32
n_channels = 3

# training info
lr = 2*1e-4
batch_size = 64
nz = 256
ngf = 16
nc = n_channels
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info("Using device: {}".format(str(device)))
logger.info("Training on {} is starting".format(dataset_name))
logger.info("We now give the hyperparameters")
logger.info("Learning rate : {}".format(lr))
logger.info("nz : {}".format(nz))
logger.info("ngf : {}".format(ngf))
logger.info("nc (Number of Channel)".format(nc))

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
                                             # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset_trn = torchvision.datasets.CIFAR10(root='./data', train=True,
                                    download=True, transform=transform)

dataset_tst = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

trn_loader = torch.utils.data.DataLoader(dataset_trn, sampler=None, shuffle=True, batch_size=batch_size, drop_last=True)
tst_loader = torch.utils.data.DataLoader(dataset_tst, sampler=None, shuffle=True, batch_size=batch_size, drop_last=True)

# create the GON network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
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
            nn.ConvTranspose2d(ngf, n_channels, 4, 2, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

F = Generator().to(device)

optim = torch.optim.Adam(lr=lr, params=F.parameters())
logger.info(f'> Number of parameters {len(torch.nn.utils.parameters_to_vector(F.parameters()))}')
# print(f'> Number of parameters {len(torch.nn.utils.parameters_to_vector(F.parameters()))}')

logger.info("\n")
epoch_mse_loss_trn = 0
epoch_sse_loss_trn = 0
epoch = 0
for i,(x,t) in enumerate(trn_loader):
    x = x.to(device)
    batch_size_ = x.shape[0]
    z = torch.zeros(batch_size_, nz, 1, 1).to(device).requires_grad_()
    g = F(z)
    L_inner = ((g - x)**2).sum(1).mean()
    grad = torch.autograd.grad(L_inner, [z], create_graph=True, retain_graph=True)[0]
    z = (-grad)#.detach()

    # now with z as our new latent points, optimise the data fitting loss
    g = F(z)
    L_outer = ((g - x)**2).sum(1).mean()
    sse = ((g - x)**2).sum() / x.shape[0]
    epoch_mse_loss_trn += L_outer.item()
    epoch_sse_loss_trn += sse.item() 
#     for w in F.parameters():
#         w.grad.data.zero_()
        
# print(f"Epoch: {epoch}   Train Mean Squared Loss: {epoch_mse_loss_trn/i:.7f}")
# print(f"Epoch: {epoch}   Train Sum Squared Loss: {epoch_sse_loss_trn/i:.7f}")

logger.info(f"Epoch: {epoch}   Train Mean Squared Loss: {epoch_mse_loss_trn/i:.7f}")
logger.info(f"Epoch: {epoch}   Train Sum Squared Loss: {epoch_sse_loss_trn/i:.7f}")

epoch_mse_loss_tst = 0
epoch_sse_loss_tst = 0
for j,(x,t) in enumerate(tst_loader):
    x = x.to(device)
    batch_size_ = x.shape[0]
    z = torch.zeros(batch_size_, nz, 1, 1).to(device).requires_grad_()
    g = F(z)
    L_inner = ((g - x)**2).sum(1).mean()
    grad = torch.autograd.grad(L_inner, [z], create_graph=True, retain_graph=True)[0]
    z = (-grad)#.detach()

    # now with z as our new latent points, optimise the data fitting loss
    g = F(z)
    L_outer = ((g - x)**2).sum(1).mean()
    sse = ((g - x)**2).sum() / x.shape[0]
    epoch_mse_loss_tst += L_outer.item()
    epoch_sse_loss_tst += sse.item()
#     for w in F.parameters():
#         w.grad.data.zero_()
        
# print(f"Epoch: {epoch}   Test Mean Squared Loss: {epoch_mse_loss_tst/j:.7f}")
# print(f"Epoch: {epoch}   Test Sum Squared Loss: {epoch_sse_loss_tst/j:.7f}")
logger.info(f"Epoch: {epoch}   Test Mean Squared Loss: {epoch_mse_loss_tst/j:.7f}")
logger.info(f"Epoch: {epoch}   Test Sum Squared Loss: {epoch_sse_loss_tst/j:.7f}")

n_epochs = 500
for epoch_ in range(n_epochs):
    step = 0
    for (x,t) in trn_loader:
        # sample a batch of data
        # x, t = next(train_iterator)
        x = x.to(device)
        batch_size_ = x.shape[0]
        # compute the gradients of the inner loss with respect to zeros (gradient origin)
        z = torch.zeros(batch_size_, nz, 1, 1).to(device).requires_grad_()
        g = F(z)
        L_inner = ((g - x)**2).sum(1).mean()
        grad = torch.autograd.grad(L_inner, [z], create_graph=True, retain_graph=True)[0]
        z = (-grad)

        # now with z as our new latent points, optimise the data fitting loss
        g = F(z)
        L_outer = ((g - x)**2).sum(1).mean()
        optim.zero_grad()
        L_outer.backward()
        optim.step()
        step += 1

    epoch_mse_loss_trn = 0
    epoch_sse_loss_trn = 0
    for i,(x,t) in enumerate(trn_loader):
        # sample a batch of data
        # x, t = next(train_iterator)
        x = x.to(device)
        batch_size_ = x.shape[0]
        z = torch.zeros(batch_size_, nz, 1, 1).to(device).requires_grad_()
        g = F(z)
        L_inner = ((g - x)**2).sum(1).mean()
        grad = torch.autograd.grad(L_inner, [z], create_graph=True, retain_graph=True)[0]
        z = (-grad)#.detach()

        # now with z as our new latent points, optimise the data fitting loss
        g = F(z)
        L_outer = ((g - x)**2).sum(1).mean()
        sse = ((g - x)**2).sum() / x.shape[0]
        epoch_mse_loss_trn += L_outer.item()
        epoch_sse_loss_trn += sse.item() 
        for w in F.parameters():
            w.grad.data.zero_()

#     torch.cuda.empty_cache()

    epoch_mse_loss_tst = 0 # Mean Squared Error, initially zero
    epoch_sse_loss_tst = 0 # Sum Squared Error, initially zero
    for j,(x,t) in enumerate(tst_loader): # Run over the test set
        x = x.to(device)
        batch_size_ = x.shape[0]
        z = torch.zeros(batch_size_, nz, 1, 1).to(device).requires_grad_()
        g = F(z)
        L_inner = ((g - x)**2).sum(1).mean()
#         grad = torch.autograd.grad(L_inner, [z], create_graph=True, retain_graph=False)[0]
#         z = (-grad).detach() # Detaching is not important during inference
        grad = torch.autograd.grad(L_inner, [z], create_graph=True, retain_graph=True)[0]
        z = (-grad) # Detaching is not important during inference

        g = F(z)
        L_outer = ((g - x)**2).sum(1).mean() # Calculate batch mean squared error
        sse = ((g - x)**2).sum() / x.shape[0] # Calculate batch summed squared error
        epoch_mse_loss_tst += L_outer.item() # Update epoch MSE Loss
        epoch_sse_loss_tst += sse.item() # Update epoch SSE Loss
        for w in F.parameters(): # Make all the grads after inference (since they do not contribute to training)
            w.grad.data.zero_()

    # print(f"\nEpoch: {epoch_ + 1}   Train Mean Squared Loss: {epoch_mse_loss_trn/i:.7f}")
    # print(f"Epoch: {epoch_ + 1}   Train Sum Squared Loss: {epoch_sse_loss_trn/i:.7f}")
    # print(f"Epoch: {epoch_ + 1}   Test Mean Squared Loss: {epoch_mse_loss_tst/j:.7f}")
    # print(f"Epoch: {epoch_ + 1}   Test Sum Squared Loss: {epoch_sse_loss_tst/j:.7f}")
    logger.info("\n")
    logger.info(f"Epoch: {epoch_ + 1}   Train Mean Squared Loss: {epoch_mse_loss_trn/i:.7f}")
    logger.info(f"Epoch: {epoch_ + 1}   Train Sum Squared Loss: {epoch_sse_loss_trn/i:.7f}")
    logger.info(f"Epoch: {epoch_ + 1}   Test Mean Squared Loss: {epoch_mse_loss_tst/j:.7f}")
    logger.info(f"Epoch: {epoch_ + 1}   Test Sum Squared Loss: {epoch_sse_loss_tst/j:.7f}")