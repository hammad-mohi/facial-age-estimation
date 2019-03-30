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
    for img, label, _ in data_loader:
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
def get_subgroup_accuracy(model, data_loader):
    
    age_accuracy = np.zeros(96)
    age_five_off_accuracy = np.zeros(96)
    age_total_counts = np.zeros(96)
    
    gender_accuracy = np.zeros(2)
    gender_five_off_accuracy = np.zeros(2)
    gender_total_counts = np.zeros(2)
    
    race_accuracy = np.zeros(5)
    race_five_off_accuracy = np.zeros(5)
    race_total_counts = np.zeros(5)
    
    for image, age, path in data_loader:
#         print(path)
        if len(path[0].split('_'))<4:
            continue
        gender = int(path[0].split('_')[1])
        race = int(path[0].split('_')[2])
        
        age+=1
        pred = model(image)

        if age.cpu().float()-0.5 <= pred.cpu() <= age.cpu().float()+0.5:
            age_accuracy[age]+=1
            gender_accuracy[gender]+=1
            race_accuracy[race]+=1

        if age.cpu().float()-5.5 <= pred.cpu() <= age.cpu().float()+5.5:
            age_five_off_accuracy[age]+=1
            gender_five_off_accuracy[gender]+=1
            race_five_off_accuracy[race]+=1
            
        age_total_counts[age]+=1
        gender_total_counts[gender]+=1
        race_total_counts[race]+=1
        
    age_accuracy = age_accuracy/age_total_counts 
    age_five_off_accuracy = age_five_off_accuracy/age_total_counts 
    
    gender_accuracy = gender_accuracy/gender_total_counts 
    gender_five_off_accuracy = gender_five_off_accuracy/gender_total_counts 
    
    race_accuracy = race_accuracy/race_total_counts 
    race_five_off_accuracy = race_five_off_accuracy/race_total_counts 
    
    print(age_total_counts)
    print(gender_total_counts)
    print(race_total_counts)
    
    plt.bar(range(1,96), age_accuracy[1:])
    plt.title("Distribution of exact Actual-Prediction")
    plt.xlabel("age")
    plt.ylabel("acc prob")
    plt.show()
    
    plt.bar(range(1,96), age_five_off_accuracy[1:])
    plt.title("Distribution of exact Actual-Prediction")
    plt.xlabel("age")
    plt.ylabel("+/- 5acc prob")
    plt.show()
    
    plt.bar(range(0,2), gender_accuracy)
    plt.title("Distribution of exact Actual-Prediction")
    plt.xlabel("gender")
    plt.ylabel("acc prob")
    plt.show()
    
    plt.bar(range(0,2), gender_five_off_accuracy)
    plt.title("Distribution of exact Actual-Prediction")
    plt.xlabel("gender")
    plt.ylabel("+/- 5acc prob")
    plt.show()
    
    plt.bar(range(0,5), race_accuracy)
    plt.title("Distribution of exact Actual-Prediction")
    plt.xlabel("race")
    plt.ylabel("acc prob")
    plt.show()   
    
    plt.bar(range(0,5), race_five_off_accuracy)
    plt.title("Distribution of exact Actual-Prediction")
    plt.xlabel("race")
    plt.ylabel("+/- 5acc prob")
    plt.show()
    
    return age_accuracy, age_five_off_accuracy, gender_accuracy, gender_five_off_accuracy, race_accuracy, race_five_off_accuracy



def get_accuracy(model, data, batch_size):
    c=0
    mean = 0.0
    for imgs, labels, _ in data:
        mean += labels.sum()
        c+=batch_size
    mean = (mean/c)
    
    correct = 0
    total = 0
    count = 0
    ss_reg = 0
    ss_total = 0
    
    for imgs, labels, _ in data:
        labels = labels.float()
        output = model(imgs.cuda())
        
        output = torch.round(output)
        output = output.float()
        output = output.cpu().detach().numpy()
        output = torch.tensor(output)
        
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