#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image


def get_off_accuracy(model, data_loader):
    p=0
    freq_pos = np.zeros(81)
    freq_neg = np.zeros(80)
    for img, label in data_loader:
        out = model(img)
        out = out.float()
        label = label.cuda()

        for i in range(0,len(label)):
            diff = label[i]+1 - out[i].long()
            if diff>=0:
                freq_pos[diff] +=1
            else:
                freq_neg[diff] +=1

    freq_total = np.concatenate((freq_neg, freq_pos))
    freq_total = freq_total/freq_total.sum()



    diffs = []
    for n in range(-80, 81):
        diffs.append(n)


    plt.bar(diffs[50:110], freq_total[50:110])
    plt.title("Distribution of Actual-Prediction")
    plt.xlabel("difference value")
    plt.ylabel("prob")
    print("+/- 1 years accuracy: {:.2f}%".format(freq_total[79:81].sum()*100))
    print("+/- 5 years accuracy: {:.2f}%".format(freq_total[75:86].sum()*100))
    print("+/- 10 years accuracy: {:.2f}%".format(freq_total[70:91].sum()*100))


# In[ ]:


def get_accuracy(model, data):
    c=0
    mean = 0.0
    for imgs, labels in data:
      mean += labels.sum()
      c+=32
    mean = (mean/c)
    
    #print(mean)
    
    correct = 0
    total = 0
    count = 0
    ss_reg = 0
    ss_total = 0
    
    for imgs, labels in data:
        labels = labels.float()
        output = model(imgs.cuda()) # We don't need to run F.softmax

        # print(output)
        # pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        # correct += pred.eq(labels.view_as(pred)).sum().item()
        output = torch.round(output)
        output = output.float()
        output = output.cpu().detach().numpy()
        output = torch.tensor(output)
        #correct += np.isclose(output.detach().numpy(), labels, 0.05).sum()
        total += imgs.shape[0]
        count+=1
        ss_reg += ((labels+1-output)**2).sum()
        ss_total += ((labels+1-mean.float())**2).sum()
    return 1-ss_reg/ss_total


# In[ ]:


def get_model_name(name, batch_size, learning_rate, epoch):
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(name,
                                                   batch_size,
                                                   learning_rate,
                                                   epoch)
    return path

