from custom_dataloaders import json_loader as dl
from training_models import autoencoder_640 as model

import torch
import torch.nn as nn
from torchvision import *
from torch.utils.data import DataLoader

import os
import json
import numpy as np
import matplotlib.pyplot as plt


def accuracy(out, labels):
    _, pred = torch.max(out, dim=1)
    return torch.sum(pred == labels).item()


def train(batch_size, train, test, num_epochs, learning_rate, weight_decay, print_every):
    """
    Preprocessing as Needed
    """
    """
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((width, height)),
        torchvision.transforms.ToTensor()
    ])
    """

    loaded_data = dl.get_dataset(train, test)
    train_dataset = loaded_data[0]
    test_dataset = loaded_data[1]

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    net = model.autoencoder()
    net = net.cuda() if device else net
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    use_cuda = torch.cuda.is_available()

    list_of_accs = []
    valid_loss_min = np.Inf
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []
    total_step = len(trainloader)

    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0
        print(f'Epoch {epoch}\n')
        for batch_idx, (data_, target_) in enumerate(trainloader):

            # print(data_, target_)
            data_, target_ = data_.to(device), target_.to(device)
            optimizer.zero_grad()

            outputs = net(data_)
            loss = criterion(outputs, target_)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, pred = torch.max(outputs, dim=1)
            correct += torch.sum(pred == target_).item()
            total += target_.size(0)
            if (batch_idx) % print_every == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, num_epochs, batch_idx, total_step, loss.item()))
        train_acc.append(100 * correct / total)
        train_loss.append(running_loss / total_step)
        print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct / total):.4f}')
        batch_loss = 0
        total_t = 0
        correct_t = 0

        confusion_matrix = torch.zeros(6, 6)
        with torch.no_grad():
            net.eval()

            for data_t, target_t in (testloader):
                data_t, target_t = data_t.to(device), target_t.to(device)
                outputs_t = net(data_t)
                loss_t = criterion(outputs_t, target_t)
                batch_loss += loss_t.item()
                _, pred_t = torch.max(outputs_t, dim=1)
                correct_t += torch.sum(pred_t == target_t).item()
                total_t += target_t.size(0)
                for t, p in zip(target_t.view(-1), pred_t.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

            val_acc.append(100 * correct_t / total_t)
            val_loss.append(batch_loss / len(testloader))
            network_learned = batch_loss < valid_loss_min
            print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')

            if network_learned:
                valid_loss_min = batch_loss
                torch.save(net.state_dict(), 'resnet.pt')
                print('Improvement-Detected, save-model')

        print(confusion_matrix)
        acc_per_class = confusion_matrix.diag() / confusion_matrix.sum(1)
        print(acc_per_class)
        list_of_accs.append(acc_per_class)
        net.train()

    fig = plt.figure(figsize=(20, 10))
    plt.title("Train-Validation Accuracy")
    plt.plot(train_acc, label='train')
    plt.plot(val_acc, label='validation')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('accuracy', fontsize=12)
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':

    """
    Hyperparameters
    """
    num_epochs = 20
    batch_size = 32
    learning_rate = 0.001
    weight_decay = 0.0001


    update_every = 100

    """
    Data Directories
    """
    train_data_loc = "data/Train"
    test_data_loc = "data/Test"

    """All the below is just because my dataset is scuffed af
    Normally, you'd skip to train() and that'd be the end of it"""


    train(batch_size,
          train_data_loc,
          test_data_loc,
          num_epochs,
          learning_rate,
          weight_decay,
          update_every)
