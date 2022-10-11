from torchvision import models, transforms
import torch.nn as nn
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import time
import torch
available_device = (torch.device('cuda') if torch.cuda.is_available() 
         else torch.device('cpu'))
print("Training on device ", available_device)


class MNISTdigits(torch.utils.data.Dataset):
  def __init__(self, file_name):
        column_names=['label']+['pixel%i'%i for i in range(784)]
        self.all_data=pd.read_csv(file_name,names=column_names)
        self.labels = torch.from_numpy(self.all_data["label"].to_numpy())
        data_points=self.all_data.iloc[:,1:].to_numpy()
        self.data_tensor=torch.from_numpy(data_points/255).float()
        self.transform=None
  def __len__(self):
        return len(self.labels)
  def __getitem__(self, index):
        X = self.data_tensor[index,:].view(1,28,28)
        if self.transform:
            X = self.transform(X)
        y = self.labels[index]

        return X, y
        

class MNISTdigitsAugmented(torch.utils.data.Dataset):
    def __init__(self, dataset, augmentation_ratio, transform=None): #the augmentation ratio is the ratio of the size of augmented data to original data, we assume that is an integer
        self.original_dataset=dataset
        self.augmentation_ratio=round(augmentation_ratio)
        self.augmentation_transform=transform
    def __len__(self):
        return (self.augmentation_ratio)*(len(self.original_dataset))
    def __getitem__(self, index):
        original_index=index//self.augmentation_ratio
        X,y = self.original_dataset[original_index]
        if index%self.augmentation_ratio:
            X = self.augmentation_transform(X)
        return X, y
    
    
class ModelTrainer:
    def __init__(self,model,loss_function,optimizer):
        self.model=model
        self.loss_fn=loss_function
        self.optimizer=optimizer
        self.epochs=[]
        self.losses=[]
    def load_data(self,file_name,split_ratio_list,augmentation_ratio=1,augmentation_transform=None):
        self.augmentation_ratio=augmentation_ratio
        self.augmentation_transform=augmentation_transform
        self.data_set=MNISTdigits(file_name)
        self.split_data(split_ratio_list)
        self.normalize_train_data()
    def split_data(self,split_ratio_list):
        split_ratio=torch.tensor(split_ratio_list)
        lengths=split_ratio*(len(self.data_set)//100)
        train_set,self.valid_set=torch.utils.data.random_split(self.data_set,lengths)
        self.train_set=MNISTdigitsAugmented(train_set,self.augmentation_ratio,self.augmentation_transform)
    def normalize_train_data(self):
        data_loader=torch.utils.data.DataLoader(self.train_set, batch_size=256, shuffle=False)
        means=[]
        stds=[]
        for imgs,labels in data_loader:
            means.append(torch.mean(imgs))
            stds.append(torch.std(imgs))
        self.mean = torch.mean(torch.tensor(means))
        self.std = torch.mean(torch.tensor(stds))
        self.data_set.transform = transforms.Normalize((self.mean),(self.std))
    def train(self,n_epochs, batch_size=64):
        self.model.train()
        self.batch_size=batch_size
        train_loader=torch.utils.data.DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        initial_n_epochs=len(self.epochs)
        start_time=time.time()
        for epoch in range(initial_n_epochs+1,initial_n_epochs+n_epochs+1): #We add len(epochs) because we would like to continue where we started after running train() the next time
            for imgs,labels in train_loader:
                imgs=imgs.to(device=available_device)
                labels=labels.to(device=available_device)
                self.loss=self.loss_fn(self.model(imgs),labels)
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
            self.epochs.append(epoch) 
            self.losses.append(self.loss.detach().item())
        end_time=time.time()
        self.training_time=end_time-start_time
    def get_training_time(self):
        return self.training_time
    def get_accuracy(self,dataset):
        self.model.eval()
        dataset.transform = self.data_set.transform
        data_loader=torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        correct_guesses=0
        all_samples=0
        with torch.no_grad():
            for imgs, labels in data_loader:
                imgs=imgs.to(device=available_device)
                labels=labels.to(device=available_device)
                _, predicted = torch.max(self.model(imgs), dim=1)
                correct_guesses+=(predicted==labels).sum()
                all_samples+=labels.shape[0]
        return correct_guesses/all_samples
    def get_train_valid_accuracy(self):
        return self.get_accuracy(self.train_set), self.get_accuracy(self.valid_set)
    def plot_loss_agains_n_epochs(self):
        plt.rcParams['figure.figsize'] = [20, 10]
        plt.plot(self.epochs,self.losses)
    def save_parameters(self,file_name):
        torch.save(self.model.state_dict(), file_name)

def analyze_model(model_trainer,batch_size,n_epochs):
    model_trainer.train(n_epochs,batch_size)
    model_trainer.plot_loss_agains_n_epochs()
    train_accuracy, valid_accuracy = model_trainer.get_train_valid_accuracy()
    return model_trainer.training_time,train_accuracy.item(),valid_accuracy.item()

class ResNetModelTrainer(ModelTrainer):
    def __init__(self,learning_rate):
        self.learning_rate=learning_rate
        self.model=models.resnet18().to(device=available_device)
        self.model.conv1=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(device=available_device)
        self.model.fc=nn.Linear(512,10).to(device=available_device)
        self.optimizer=torch.optim.SGD(self.model.parameters(),lr=learning_rate)
        self.loss_fn=nn.CrossEntropyLoss()
        self.epochs=[]
        self.losses=[]
    def save_parameters(self,file_suffix=''):
        self.parameters_file_name="./saved_models/resnet18_epochs_%d_batch_size_%d_lr_%f_val_accuracy_%f%s.pt"%(len(self.epochs),self.batch_size,self.learning_rate,self.get_accuracy(self.valid_set),file_suffix)
        super().save_parameters(self.parameters_file_name)
