# -*- coding: utf-8 -*-
"""Classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1l5FSUpmVLfoOu9oUogn55e_3gWVYe_XE
"""

#If you would run nn on TPU
flag = 1
#check access to a TPU
import os
assert os.environ['COLAB_TPU_ADDR'], 'Make sure to select TPU from Edit > Notebook settings > Hardware accelerator'

#Installing Pytorch/XLA
!pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.8-cp37-cp37m-linux_x86_64.whl

#Import lb

import torch_xla
import torch_xla.core.xla_model as xm

from google.colab import drive
drive.mount('/content/drive')

import torch
import numpy as np
import os
from sklearn.model_selection import train_test_split

class CNN_Classifier(torch.nn.Module):
    def __init__(self):
        super(CNN_Classifier, self).__init__()
        
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5)
        self.act1 = torch.nn.ReLU()
        
        self.conv2 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
        self.act2 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.act3 = torch.nn.ReLU()
        
        self.conv4 = torch.nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3)
        self.act4 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.fc1 = torch.nn.Linear(128*71, 256)
        self.act5 = torch.nn.ReLU()
        
        self.fc2 = torch.nn.Linear(256, 128)
        self.act6 = torch.nn.ReLU()

        self.dropout_f = torch.nn.Dropout(p = 0.4)
        self.dropout_c2 = torch.nn.Dropout(p = 0.1)
        self.dropout_c4 = torch.nn.Dropout(p = 0.2)
        
        self.fc3 = torch.nn.Linear(128, 5)
        
    def forward(self, x):
        x = x.view(256, 1, 300)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.dropout_c2(x)
        
        x = self.conv2(x)
        x = self.act2(x)

        x = self.pool1(x)
        
        x = self.conv3(x)
        x = self.act3(x)
        x = self.dropout_c4(x)
        
        x = self.conv4(x)
        x = self.act4(x)
        x = self.pool2(x)
        
        x = x.view(x.size(0), x.size(1) * x.size(2))
        
        x = self.dropout_f(x)
        x = self.fc1(x)
        x = self.act5(x)
        x = self.dropout_f(x)    
        x = self.fc2(x)
        x = self.act6(x)
        x = self.fc3(x)
        
        return x

## Load CNN param
'''
CNN = CNN_Classifier()
CNN.load_state_dict(torch.load(PATH))
CNN.eval()
'''
CNN = CNN_Classifier()

#If you would run nn on GPU/CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#If you would run nn on TPU
device = xm.xla_device()

CNN = CNN.to(device)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(CNN.parameters(), lr=1.0e-3, eps=0.001)

class Custom_Subset(torch.utils.data.Dataset):
    
    def __init__(self, data, labels, indices, transform=None):
        """
        Args:
            data (torch.Tensor): Датасет в формате тензора
            labels (torch.Tensor): Метки датасета в формате тензора
            indices (list): Список индексов из целого датасета
            transform (callable, optional): Необязательный transform 
                который будет применен к экземпляру.
        """
        
        self.labels = labels
        self.data = data
        self.indices = indices
        self.transform = transform
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
            signal = self.data[self.indices[idx]]
            label = self.labels[self.indices[idx]]
            sample = {'signal': signal, 'label': label}
            
            if self.transform:
                sample = self.transform(sample)
                
            return sample

train_label_path = '/content/drive/MyDrive/MIT-BIH_Pre-Processed/train_label.pt'
train_data_path = '/content/drive/MyDrive/MIT-BIH_Pre-Processed/train_data.pt'

data = torch.load(train_data_path)
labels = torch.load(train_label_path)

#Splitting train_valid 
train_idx, valid_idx = train_test_split(np.arange(labels.size(0)), test_size=0.1, shuffle=True, 
                                       stratify = labels)
train_data = Custom_Subset(data, labels, train_idx)
valid_data = Custom_Subset(data, labels, valid_idx)

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=256, drop_last=True)
valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=256, drop_last=True)

import itertools
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

from sklearn import metrics
import time

num_epoch = 50
classes = ['N', 'SVEB', 'VEB', 'F', 'Q']
y_test = []
predicts = []
test_loss_history = []

for epoch in range(num_epoch):
    t_start = time.monotonic()
    for batch in train_dataloader:
        optimizer.zero_grad()
        
        X_batch = batch['signal'].to(device)
        y_batch = batch['label'].to(device)
        
        preds = CNN.forward(X_batch)
        
        loss_value = loss(preds, y_batch)
        loss_value.backward()
        
        optimizer.step()
    
    #perform a prediction on the validation  data
    for batch_test in valid_dataloader:
        X_test_batch = batch_test['signal'].to(device)
        y_test_batch = batch_test['label'].to(device)
        test_preds = CNN.forward(X_test_batch)
        _, labels = torch.max(test_preds, 1)
        y_test.extend(y_test_batch.tolist())
        predicts.extend(labels.tolist())

    t_stop = time.monotonic()
    t_run = t_stop - t_start
    print('Training time of epoch {} is {} seconds'.format(epoch, t_run))
    print(metrics.classification_report(y_test, predicts, target_names=classes))

model_path = '/content/drive/MyDrive/state_dict.pt'
torch.save(CNN.state_dict(), model_path)

## Dataset for test
class Custom_Dataset(torch.utils.data.Dataset):
    
    def __init__(self, data, labels, transform=None):
        """
        Args:
            data (torch.Tensor): Датасет в формате тензора
            labels (torch.Tensor): Метки датасета в формате тензора
            transform (callable, optional): Необязательный transform 
                который будет применен к экземпляру.
        """
        
        self.labels = labels
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
            signal = self.data[idx]
            label = self.labels[idx]
            sample = {'signal': signal, 'label': label}
            
            if self.transform:
                sample = self.transform(sample)
                
            return sample

#Create test_dataloader
test_label_path = '/content/drive/MyDrive/MIT-BIH_Pre-Processed/test_label.pt'
test_data_path = '/content/drive/MyDrive/MIT-BIH_Pre-Processed/test_data.pt'
test_data = torch.load(test_data_path)
test_labels = torch.load(test_label_path)
k = Custom_Dataset(test_data, test_labels)
test_dataloader = torch.utils.data.DataLoader(k, batch_size=256, drop_last=True)

classes = ['N', 'SVEB', 'VEB', 'F', 'Q']
y_test = []
predicts = []
for batch_test in test_dataloader:
    X_test_batch = batch_test['signal'].to(device)
    y_test_batch = batch_test['label'].to(device)
    yest_preds = CNN.forward(X_test_batch)
    _, labels = torch.max(test_preds, 1)
    y_test.extend(y_test_batch.tolist())
    predicts.extend(labels.tolist())

cmt = metrics.confusion_matrix(y_test, predicts)
plot_confusion_matrix(cmt, classes)
print(metrics.classification_report(y_test, predicts, target_names=classes))