import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

import numpy as np
import time
from tqdm import tqdm


class MLP(nn.Module):
    """
    Multilayer perceptrons
    """

    def __init__(self, n_feat, nn_dim, n_class, dropout, bias=True):
        super().__init__()

        self.nn1 = nn.Linear(n_feat, nn_dim, bias=bias)
        self.nn2 = nn.Linear(nn_dim, n_class, bias=bias)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.nn1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.nn2(x)
        return F.log_softmax(x, dim=1)
    

class MyDataset(Dataset):
    """
    Make torch.Dataset object from .npy file.
    """
    def __init__(self, data_fn):
        data = np.load(data_fn)
        self.features = torch.tensor(data[:, :-1], dtype=torch.float)
        self.labels = torch.tensor(data[:, -1], dtype=torch.long)
        
    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        features = self.features[idx]
        labels = self.labels[idx]
        return features, labels


def train(model, optimizer, loss, train_loader, valid_lodaer, epoch, mlp_fn, opt_fn):
    t = time.time()
    correct = 0
    train_loss = 0.0
    
    for batch_idx, (feature, label) in enumerate(tqdm(train_loader)):
        model.train()  # set the mode to "train".

        optimizer.zero_grad()

        output_train = model(feature)

        loss_train = loss(output_train, label)
        
        loss_train.backward()
        optimizer.step()
        
        torch.save(model.state_dict(), mlp_fn)
        torch.save(optimizer.state_dict(), opt_fn)
        
        pred = output_train.data.max(1, keepdim=True)[1]
        correct += pred.eq(label.data.view_as(pred)).sum()
        train_loss += loss_train.item()
    
    train_loss /= len(train_loader.dataset)
    print(f'Epoch: {epoch + 1}',
          f'loss_train: {train_loss}',
          f'acc_train: {correct}/{len(train_loader.dataset)} ({100. * correct/ len(train_loader.dataset)}%)\n'
          f'time: {time.time() - t}s'
    )
    val_loss = validation(model, loss, valid_lodaer)
    return train_loss, val_loss

        
def validation(model, loss, valid_loader):
    model.eval()
    valid_loss = 0
    correct = 0
    
    # Test validation data
    with torch.no_grad():
        for feature, label in valid_loader:
            output = model(feature)
            valid_loss += loss(output, label)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(label.data.view_as(pred)).sum()
    
    valid_loss /= len(valid_loader.dataset)
    print(f'\nValidation set: Avg. loss: {valid_loss}, '
          f'Accuracy: {correct}/{len(valid_loader.dataset)} ({100. * correct/ len(valid_loader.dataset)}%)\n')
    
    return valid_loss