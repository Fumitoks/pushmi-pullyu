import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
import pytorch_lightning as pl
import os
from collections import Counter
import numpy as np
from tqdm import tqdm_notebook as tqdm
from .utils import numpify_list, numpify

class VGG(nn.Module):
    def __init__(self, channels, fc_dims, kernel_size=[], activation_type='relu'):
        super(VGG, self).__init__()
        self.vgg_blocks = nn.ModuleList()
        self.fcs = nn.ModuleList()
        for i in range(len(channels) - 1):
            if len(kernel_size) == 0:
                k_size = 3
            elif len(kernel_size) == 1:
                k_size = kernel_size[0]
            else:
                k_size = kernel_size[i]
            self.vgg_blocks.append(VGGBlock(in_channels = channels[i], out_channels = channels[i+1],
                                            kernel_size=k_size, activation_type = activation_type))
        for i in range(len(fc_dims) - 1):
            self.fcs.append(nn.Linear(fc_dims[i], fc_dims[i+1]))

    def forward(self, x):
        for block in self.vgg_blocks:
            x = block.forward(x)
        x = x.reshape(x.size(0), -1)
        for i in range(len(self.fcs)):
            x = self.fcs[i](x)
            if i!=len(self.fcs)-1:
                x = nn.ReLU()(x)
        return x

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation_type='relu'):
        super(VGGBlock, self).__init__()
        if activation_type == 'relu':
            self.activation = nn.ReLU()
        self.sequence = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
                #nn.BatchNorm2d(out_channels),
                self.activation,
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
                #nn.BatchNorm2d(out_channels),
                self.activation,
                nn.MaxPool2d(kernel_size=2, stride=2))
        
    def forward(self, x):
        x = self.sequence(x)
        return x

class SimpleConv(nn.Module):
    def __init__(self, channels, fc_dims, kernel_size=[], activation_type='relu', **kwargs):
        super(SimpleConv, self).__init__()
        self.vgg_blocks = nn.ModuleList()
        self.fcs = nn.ModuleList()
        for i in range(len(channels) - 1):
            if len(kernel_size) == 0:
                k_size = 3
            elif len(kernel_size) == 1:
                k_size = kernel_size[0]
            else:
                k_size = kernel_size[i]
            self.vgg_blocks.append(ConvBlock(in_channels = channels[i], out_channels = channels[i+1],
                                            kernel_size=k_size, activation_type = activation_type))
        for i in range(len(fc_dims) - 1):
            self.fcs.append(nn.Linear(fc_dims[i], fc_dims[i+1]))

    def forward(self, x):
        for block in self.vgg_blocks:
            x = block.forward(x)
        x = x.reshape(x.size(0), -1)
        for i in range(len(self.fcs)):
            x = self.fcs[i](x)
            if i!=len(self.fcs)-1:
                x = nn.ReLU()(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation_type='relu'):
        super(ConvBlock, self).__init__()
        if activation_type == 'relu':
            self.activation = nn.ReLU()
        self.sequence = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
                #nn.BatchNorm2d(out_channels),
                self.activation,
                nn.MaxPool2d(kernel_size=2, stride=2))
        
    def forward(self, x):
        x = self.sequence(x)
        return x

def get_model_name(config):
    return '_'.join([
            config.model.loss_type,
            str(config.model.batch_size),
            str(f'{config.model.lr:.0e}'),
            config.arch.name,
            config.model.signature
            ])

class TestMnist(pl.LightningModule):
    def __init__(self, config):
        super(TestMnist, self).__init__()
        # mnist images are (1, 28, 28) (channels, width, height) 
        self.batch_size = config.model.batch_size
        self.lr = config.model.lr
        self.loss_type = config.model.loss_type
        self.signature = config.model.signature
        self.arch = config.arch.__dict__
        self.constant_split_val = False

        self.num_classes = 10
        self.custom_step = 0
        self.val_results = list()
        self.train_transform = transforms.Compose([
                    transforms.RandomAffine(degrees=20, translate=(0.1,0.1), scale=(0.9, 1.1))
                    #transforms.ToPILImage(),                   
                    #transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    #transforms.ToTensor(),
                    #transforms.Normalize((0.1307,), (0.3081,))
                    ])        
        self.name = get_model_name(config)

        self.net = self.init_layers()

    def init_layers(self):
        if self.arch['mode'] == 'SimpleConv':
            return SimpleConv(**self.arch)

    def forward(self, x):
        out = self.net.forward(x)
        out = torch.softmax(out, dim=1)
        return out

    def full_loss(self, logits, prefix='train'):
        s = logits.shape[0]
        logits, transformed_logits = logits[:s//2], logits[s//2:]
        mean_logits = torch.mean(logits, axis=0)

        if self.loss_type == 'distance':
            loss1 = torch.linalg.norm(mean_logits-torch.ones_like(mean_logits)/self.num_classes)
            simplex_vertex = torch.zeros_like(mean_logits)
            simplex_vertex[0] = simplex_vertex[0] + 1
            max_dist = torch.linalg.norm(simplex_vertex-torch.ones_like(mean_logits)/self.num_classes)
            loss2 = max_dist - torch.mean(torch.linalg.norm(logits-torch.ones_like(logits)/self.num_classes, dim=1))
        
        if self.loss_type == 'entropy':
            loss1 = self.entropy_loss(mean_logits)
            loss2 = torch.mean(self.entropy_loss(logits))

        loss3 = torch.mean(torch.linalg.norm(logits - transformed_logits, dim=1))

        #TODO: experiment with coefficients 
        loss = 1.*loss1 + 1.*loss2 + 1.*loss3
        return {prefix + '_loss' : loss, prefix + '_loss1' : loss1, prefix + '_loss2' : loss2, prefix + '_loss3' : loss3}

    def entropy_loss(self, logits):
        ent = logits * torch.log(logits+0.0001)
        return ent.sum(axis=-1)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        transformed = self.train_transform(x)
        x = torch.cat([x, transformed])
        logits = self.forward(x)
        loss_dict = self.full_loss(logits, 'train')
        self.log_dict(loss_dict, prog_bar=True, logger=False)
        loss = loss_dict['train_loss']
        
        # wandb logs
        self.custom_step += batch[0].shape[0]
        self.logger.experiment.log({
            'train_loss' : loss,
            'epoch' : self.current_epoch
            }, step = self.custom_step, commit=False)

        return loss

    def get_predictions(self, batch):
        '''
        Returns arrays (start probabilities, end probabilities) on given batch 
        '''    
        x, _ = batch
        with torch.no_grad():
            logits = self.forward(x)
        return logits

    def full_eval(self, eval_batch_size = 1000, mode='val'):
        old_bs = self.batch_size
        self.batch_size = eval_batch_size
        if mode == 'val':
            dataloader = self.val_dataloader()
        elif mode == 'train':
            dataloader = self.train_dataloader()
        else:
            print('Wrong mode selected')
            return None
        logits_list = []
        labels_list = []
        for batch in tqdm(dataloader):
            _, labels = batch
            batch_logits = self.get_predictions(batch)
            logits_list.append(batch_logits)
            labels_list.append(labels)
        logits = torch.cat(logits_list)
        true_labels = torch.cat(labels_list)
        _, predicted_labels = torch.max(logits, dim=1)
        logits, true_labels, predicted_labels = numpify_list([logits, true_labels, predicted_labels])
        self.batch_size = old_bs
        return {'logits' : logits, 'true_labels' : true_labels, 'predicted_labels' : predicted_labels}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        transformed = self.train_transform(x)
        x = torch.cat([x, transformed])
        logits = self.forward(x)
        loss_dict = self.full_loss(logits, 'val')

        _, predicted_labels = torch.max(logits[:logits.shape[0]//2], dim=1)

        d = {'val_loss' : loss_dict['val_loss'], 'true_labels' : y, 'predicted_labels' : predicted_labels, 'logits' : logits[:logits.shape[0]//2]}
        return d

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        true_labels = torch.cat([x['true_labels'] for x in outputs])
        predicted_labels = torch.cat([x['predicted_labels'] for x in outputs])
        logits = torch.cat([x['logits'] for x in outputs])
        acc = get_cluster_acc(true_labels.cpu().detach().numpy(), predicted_labels.cpu().detach().numpy())
        self.val_results.append({'true_labels' : true_labels, 'predicted_labels' : predicted_labels, 'logits' : logits, 'acc' : acc})
        # log to progess bar
        log_dict = {'val_loss' : avg_loss, 'val_acc' : acc}
        self.log_dict(log_dict, prog_bar=True, logger=False)
        # wandb logger
        self.logger.experiment.log(log_dict, step = self.custom_step, commit=False)
        
    def prepare_data(self):
        # transforms for images
        transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.1307,), (0.3081,))])       
        # prepare transforms standard to MNIST
        mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)       
        #self.mnist_train, _ = random_split(mnist_train, [60000, 0])
        #_, self.mnist_val = random_split(mnist_train, [55000, 5000])
        if self.constant_split_val:
            self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000], generator=torch.Generator().manual_seed(666))
        else:
            self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self,mnist_test, batch_size=self.batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

def get_cluster_acc(true_labels, predicted_labels):
    counters = []
    for i in np.unique(predicted_labels).tolist():
        counters.append(Counter(true_labels[predicted_labels==i]))
    return sum([max(counter.values()) for counter in counters]) / true_labels.shape[0]
