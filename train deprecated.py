import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="true"
import cv2
import numpy as np
import torch
import torch.nn as nn
import dataManager
from torch.utils.data import DataLoader

from torchinfo   import summary

#import importlib
#importlib.reload(dataLoader)
from model import MyModel,FCDenseNet57

# Device configuration


class Trainer():
    def __init__(self, model, train_dataset, test_dataset, training_name, weight_folder,
                    num_epochs, batch_size, learning_rate, device ) :
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device

        self.criterion = nn.L1Loss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        
        
    def initialise_weight():
        pass

    def train_one_batch(self,in_tens, out_tens, ext_tens):
        in_tens = in_tens.to(self.device)
        out_tens = out_tens.to(self.device)
        # print(in_tens.device)
        
        # Forward pass
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
        outputs = self.model(in_tens)
        # print(outputs.size())
        # print(out_tens.size())
        loss = self.criterion(outputs, out_tens)

        # Backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return outputs

    def show_batch_results(self,epoch, i, in_tens, out_tens, ext_tens):
        print (f'Epoch [{epoch+1}/{self.num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
    
        dic = self.train_dataset.separate_into_images(outputs)
        dic_label = self.train_dataset.separate_into_images(out_tens)
        dataManager.saveImage("./Results\\"+str(epoch)+"_"+str(i)+"_Input.exr",in_tens.cpu())

        for key,value in dic.items():
            
            dataManager.saveImage("./Results\\"+str(epoch)+"_"+str(i)+"_"+ key +".png",value.cpu())
            dataManager.saveImage("./Results\\"+str(epoch)+"_"+str(i)+"_"+ key +"_label.png",dic_label[key].cpu())


    def train_from_scratch(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #device = "cpu"
        print("**** Starting Training on {} ****".format(device))

        print("Loading dataset")
        dataset = dataManager.MyDataset("./generated_images_512")
        dataset.check_data()
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print(device)
        model = FCDenseNet57(8,device).to(device) # We train for 8 channels because we do not predict the Environement yet
        summary(model, (1,3, 512,512))

        n_total_steps = len(dataset)
        for epoch in range(num_epochs):
            
            for i, (in_tens, out_tens, ext_tens) in enumerate(train_dataloader):
                outputs = self.train_one_batch(in_tens, out_tens, ext_tens)

                if (i) % 100 == 0:
                    print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
                
                    dic = dataset.separate_into_images(outputs)
                    dic_label = dataset.separate_into_images(out_tens)
                    dataManager.saveImage("./Results\\"+str(epoch)+"_"+str(i)+"_Input.exr",in_tens.cpu())

                    for key,value in dic.items():
                        
                        dataManager.saveImage("./Results\\"+str(epoch)+"_"+str(i)+"_"+ key +".png",value.cpu())
                        dataManager.saveImage("./Results\\"+str(epoch)+"_"+str(i)+"_"+ key +"_label.png",dic_label[key].cpu())






if __name__ =="__main__":

    num_epochs = 200
    batch_size = 1
    learning_rate = 0.01


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = "cpu"
    print("**** Starting Training on {} ****".format(device))

    print("Loading dataset")
    dataset = dataManager.MyDataset("./generated_images_512")
    dataset.check_data()
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(device)
    model = FCDenseNet57(8,device).to(device) # We train for 8 channels because we do not predict the Environement yet
    summary(model, (1,3, 512,512))


    criterion = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        n_total_steps = len(dataset)
        for i, (in_tens, out_tens, ext_tens) in enumerate(train_dataloader):

            in_tens = in_tens.to(device)
            out_tens = out_tens.to(device)
            # print(in_tens.device)
          
            # Forward pass
            # print(torch.cuda.memory_summary(device=None, abbreviated=False))
            outputs = model(in_tens)
            # print(outputs.size())
            # print(out_tens.size())
            loss = criterion(outputs, out_tens)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i) % 100 == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
                
                dic = dataset.separate_into_images(outputs)
                dic_label = dataset.separate_into_images(out_tens)
                dataManager.saveImage("./Results\\"+str(epoch)+"_"+str(i)+"_Input.exr",in_tens.cpu())

                for key,value in dic.items():
                    
                    dataManager.saveImage("./Results\\"+str(epoch)+"_"+str(i)+"_"+ key +".png",value.cpu())
                    dataManager.saveImage("./Results\\"+str(epoch)+"_"+str(i)+"_"+ key +"_label.png",dic_label[key].cpu())

                

    print('**** Finished Training ****')