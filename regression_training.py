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
from google.cloud import storage


def get_off_accuracy_regression(model, data_loader, GPU= 0):
    p=0
    freq_pos = np.zeros(81)
    freq_neg = np.zeros(80)
    for img, label in data_loader:
        out = model(img)
        #out = out.int()
        label = label.cuda(GPU)

        for i in range(0,len(label)):
            diff = label[i]+1 - out[i].int()
            if diff>=0:
                freq_pos[diff] +=1
            else:
                freq_neg[diff] +=1

    diffs = []
    for n in range(-80, 81):
        diffs.append(n)
    
    freq_total = np.concatenate((freq_neg, freq_pos))
    freq_total = freq_total/freq_total.sum()
    plt.bar(diffs[50:110], freq_total[50:110])
    plt.title("Distribution of Actual-Prediction")
    plt.xlabel("difference value")
    plt.ylabel("prob")
    print("+/- 1 year accuracy: {:.2f}%".format(freq_total[79:81].sum()*100))
    print("+/- 5 years accuracy: {:.2f}%".format(freq_total[75:86].sum()*100))
    print("+/- 10 years accuracy: {:.2f}%".format(freq_total[70:91].sum()*100))
    
    return
    
    

def evaluate_regression(data_eval, net, criterion, batch_size =32, GPU = 0):
    total_epoch = 0
    total_loss = 0
    for inputs, labels in data_eval:
        outputs = net(inputs.cuda(GPU))
        loss = criterion(outputs.cuda(GPU), labels.float().cuda(GPU))
        total_loss += loss.item()
        total_epoch += len(labels)
    return float(total_loss)/(total_epoch)
  
    
def train_regression(train_data, valid_data, net, pretrained = False, checkpoint = None, model_name = 'VGG', batch_size=32, learning_rate=5e-05, num_epochs=30, GPU= 0):
    
    
    ########################################################################
    # Fixed PyTorch random seed for reproducible result
    from torch.autograd import Variable
    torch.manual_seed(1000)
    ########################################################################
    # Obtain the PyTorch data loader objects to load batches of the datasets
    #train_loader, val_loader, test_loader = load_dataset(batch_size)
    ########################################################################
    # Define the Loss function and optimizer
    # The loss function will be 
    # Optimizer will be SGD with Momentum.
    
    criterion = nn.MSELoss() #.cuda(GPU)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    pretrained_epoch = 0
    train_losses, val_losses, train_acc, val_acc = [], [], [], []
    if pretrained == True:
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        pretrained_epoch = checkpoint['epoch']
        train_losses = checkpoint['train_loss']
        val_losses = checkpoint['valid_loss']
        net.train()
        print("resuming training after epoch: ", pretrained_epoch)
    
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    ########################################################################
    # Train the network
    # Loop over the data iterator and sample a new batch of training data
    # Get the output from the network, and optimize our loss function.
    start_time = time.time()
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        total_epoch = 0
        total_train_loss = 0
        #random.shuffle(train_data)
        for inputs, labels in train_data:
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass, backward pass, and optimize
            outputs = net(Variable(inputs.cuda(GPU)))
            #loss = criterion(outputs.cuda(GPU), labels.float().cuda(GPU))
            loss = criterion(outputs.cuda(GPU), labels.float().cuda(GPU))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            total_epoch += len(labels) #adding batch size
            
        # save the current training information
        train_losses.append(float(total_train_loss)/total_epoch)            # compute *average* loss
        val_losses.append(evaluate_regression(valid_data, net, criterion, batch_size = batch_size, GPU = GPU))
        #train_acc.append(get_accuracy(net, train_data, batch_size = batch_size)) # compute training accuracy 
        #val_acc.append(get_accuracy(net, valid_data, batch_size = batch_size))  # compute validation accuracy
            
        #print("Epoch: {}, Training Loss: {:.3f}, Validation Loss: {:.3f}, Training Accuracy: {:.3f}, Validation Accuracy: {:.3f}".format(epoch+pretrained_epoch+1, train_losses[-1], val_losses[-1], train_acc[-1],val_acc[-1]))
        print("Epoch: {}, Training Loss: {:.3f}, Validation Loss: {:.3f}".format(epoch+pretrained_epoch+1, train_losses[-1], val_losses[-1]))
        # Save the current model (checkpoint) to a file
        #torch.save({ 'epoch': epoch, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'train_loss': train_losses, 'train_accuracy': train_acc, 'valid_loss': val_losses, 'valid_accuracy': val_acc}, 
        #           "{}_features_bs{}_lr{}".format(pretrained, batch_size, learning_rate))
        
        torch.save({ 'epoch': epoch+pretrained_epoch+1, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'train_loss': train_losses, 'valid_loss': val_losses}, 
                   "{}_features_bs{}_lr{}".format(model_name, batch_size, learning_rate))
        
        #saving it in the google cloud storage
        client = storage.Client()
        bucket = client.get_bucket("aps360team12")
        blob_name = "{}_features_bs{}_lr{}_epoch{}".format(model_name, batch_size, learning_rate, epoch+pretrained_epoch+1)
        blob = bucket.blob(blob_name)

        source_file_name = "{}_features_bs{}_lr{}".format(model_name, batch_size, learning_rate)
        blob.upload_from_filename(source_file_name)
        print("File uploaded to {}.".format(bucket.name))
        
    
    
        
    # plotting
    plt.title("Loss curves w/ lr={}, batch size = {}".format(learning_rate, batch_size))
    plt.axis([0, num_epochs, 0, max(train_losses[0], val_losses[0])])
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    print("Final Training Loss: {}".format(train_losses[-1]))
    print("Final Validation Loss: {}".format(val_losses[-1]))

    print('Finished Training')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))
    
    return

def regression_demo(data, model):
    k = 0
    for img, label in data:
        #img = img / 2 + 0.5
        #plt.subplot(3, 5, k+1)
        #plt.axis('off')
        pred = model(img)
        for count, i in enumerate(img, 0):
            print(count)
            i = np.transpose(i, [1,2,0])
            plt.imshow(i)
            plt.show()
            print("Actual Age: ", label[count], " Prediction: ", pred[count])
            k += 1
        #print(label, vgg_regression(img))
            if k > 10:
                break

        break
    
    return