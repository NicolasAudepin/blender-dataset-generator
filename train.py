import os
from time import time
from tkinter import E, N
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="true"
import cv2
import numpy as np
import torch
import torch.nn as nn
import dataManager
from torch.utils.data import DataLoader

from rich import print
from rich.panel import Panel
from rich.text import Text
from rich.console import Group
from rich.pretty import pprint
import logging
from training_display import Training_display_tread
from torchinfo import summary
from datetime import datetime
from random_pokemon import random_pokemon

from torch.utils.tensorboard import SummaryWriter

#import importlib
#importlib.reload(dataLoader)
from model import FCDenseNet57, FCDenseNet57

# Device configuration


class Trainer():
    def __init__(self, model, train_dataset, test_dataset, training_name, weight_folder,
                    num_epochs=10, batch_size = 0, learning_rate = 0.01, device="cuda", description = ""  ) :
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset 
        self.training_name = training_name
        self.weight_folder = weight_folder
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.description = description
        self.criterion = nn.L1Loss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        print(Panel(Group(
            f"                  Name : [yellow bold]{self.training_name}",
            f"            Num epochs : [cyan bold]{self.num_epochs}",
            f"            Batch size : [cyan bold]{self.batch_size}",
            f"         Learning rate : [cyan bold]{self.learning_rate}",
            f"                Device : [cyan bold]{self.device}",
            f"logs and weight folder : [cyan bold]{self.weight_folder}",
            ),title="Trainer's hyper-param"))

        
    def initialise_weight():
        pass

    def description_string(self,details =""):
        str = f"Name : {self.training_name}"+"\n \n"
        str +=f"logs and weight folder : {self.weight_folder} \n \n"
        str +=f"Train data: {self.train_dataset.image_folder} \n \n"
        str +=f"Test data: {self.test_dataset.image_folder} \n \n"
        str +=f"Learning rate: {self.learning_rate} \n \n"
        # str +=f"Model: {type(self.model)} \n \n"

        str += "\n"+details

        return str

    def evaluate_one_batch(self,in_tens, out_tens, ext_tens):
        in_tens = in_tens.to(self.device)
        out_tens = out_tens.to(self.device)
        # print(in_tens.device)
        # Forward pass
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
        outputs = self.model(in_tens)
        # print(outputs.size())
        # print(out_tens.size())
        loss = self.criterion(outputs, out_tens)
        return outputs, loss

    def train_one_batch(self,in_tens, out_tens, ext_tens):
        """Trains the model of self.model on one batch. returns the output and loss for this batch 
        \n in_tens : The input batch of the model
        \n out_tens : The target batch output of the model  
        """
        outputs, loss = self.evaluate_one_batch(in_tens, out_tens, ext_tens)

        # Backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return outputs, loss



    def show_batch_results(self,epoch, i, in_tens, out_tens, ext_tens ,outputs , loss):
        
        dic = self.train_dataset.separate_into_images(outputs)
        dic_label = self.train_dataset.separate_into_images(out_tens)
        dataManager.saveImage("./Results\\"+str(epoch)+"_"+str(i)+"_Input.exr",in_tens.cpu())

        for key,value in dic.items():        
            dataManager.saveImage("./Results\\"+str(epoch)+"_"+str(i)+"_"+ key +".png",value.cpu())
            dataManager.saveImage("./Results\\"+str(epoch)+"_"+str(i)+"_"+ key +"_label.png",dic_label[key].cpu())


    def simple_progress_bar(self, value, max, width=40, style = "[magenta bold]"):
        """return a string that represent the progress of the value toward the max as a growing arrow """
        adv = int(value/max*width)+1
        return style+(">".rjust(adv,"=").ljust(width))



    def train_from_scratch(self):
        """The whole training of the model with tensorboard and saving good epoch weights. Uses all of the args of the class """
        print(Panel(f"Training from scratch on {self.device}"))
        self.train_dataset.check_data()
        self.test_dataset.check_data()

        train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)
        
        train_display = Training_display_tread()
        train_display.start()
        displayed_data = dict()

        folder =  self.weight_folder
        writer = SummaryWriter(log_dir=folder)
        writer.add_text("Description",self.description_string(self.description))
                    
        best_loss = 123456789
        
        for epoch in range(self.num_epochs):
            
            time_epoch_start = datetime.now()
            #Train the model
            for i, (in_tens, out_tens, ext_tens) in enumerate(train_dataloader):
                try:
                    outputs,loss = self.train_one_batch(in_tens, out_tens, ext_tens) #the actual training
                    
                    #displaying stuff in the table
                    displayed_data["Epoch"]=f"{epoch+1}/{self.num_epochs}"
                    time = str(datetime.now() - time_epoch_start).split('.')[0] 
                    displayed_data["Time"]=f"[cyan bold]{time}"
                    displayed_data["Step"]=f"[cyan bold]{i+1}/{len(self.train_dataset)}"
                    displayed_data["Loss"]=f"[yellow bold]{loss.item():.4f}"
                    displayed_data["Train Progress"]= self.simple_progress_bar(i,len(self.train_dataset),width=50)
                    displayed_data["Test Step"]= None
                    displayed_data["Test Progress"]= None
                    displayed_data["Test Loss"]= None
                    displayed_data["Saved"]= None
                    train_display.queue.put(displayed_data)
                    
                    if (i) % 1000 == 0:
                        self.show_batch_results(epoch, i, in_tens, out_tens, ext_tens, outputs, loss) #saves images
                    if i == 0:
                        writer.add_scalar("Loss/train",loss.item(),epoch)
                except KeyboardInterrupt:
                    train_display.stop()
                    raise KeyboardInterrupt
                except Exception as ex:    
                    train_display.stop()
                    print(ex.args)
                    raise Exception

            #calculate test loss
            torch.cuda.empty_cache() 
            losses=[]
            with torch.no_grad(): #to avoid cuda OOM error
                for i, (in_tens, out_tens, ext_tens) in enumerate(test_dataloader):
                    _, loss = self.evaluate_one_batch(in_tens, out_tens, ext_tens)
                    losses.append(loss.item())
                    displayed_data["Test Step"]=f"[cyan bold]{i+1}/{len(self.test_dataset)}"
                    displayed_data["Test Progress"]= self.simple_progress_bar(i,len(self.test_dataset),width=20,style=f"[magenta bold]")
            test_loss = np.mean(losses)

            #save the model
            if best_loss > test_loss:   
                best_loss=test_loss
                torch.save(model.state_dict(), folder + f"\\Loss_{test_loss:.4f}_Epoch_{epoch}.pt")
                displayed_data["Saved"]= "[green bold]YES"
            else:
                displayed_data["Saved"]= "[red]NO"

            #log and print the test loss
            displayed_data["Test Loss"]= f"[yellow bold]{test_loss:.4f}"
            writer.add_scalar("Loss/test",test_loss,epoch)

            #show the result of previous epochs in the rich table
            train_display.previous_data.append(displayed_data.copy())

        #stop the display thread (or at least try)
        train_display.stop()
        train_display.join()
        




if __name__ =="__main__":
    logging.basicConfig(level=logging.DEBUG)
    # logging.debug("yo")

    print(Panel(Text("Script train.py Starting",justify = "center",style="bold magenta")))
    
    num_epochs = 20000
    batch_size = 1
    learning_rate = 0.01

    description  ="Using FCDenseNet57 on the Tiramisu architecture with RELU final activation. Does not includ the HDRI yet. Does not clamp the exr Input yet. Not Great yet."

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = "cpu"
    
    #generated_images_1_im

    train_dataset = dataManager.MyDataset("./generated_images_256")
    test_dataset = dataManager.MyDataset("./generated_images_256")
    train_dataset = dataManager.MyDataset("./generated_images_512_train")
    test_dataset = dataManager.MyDataset("./generated_images_512_test")
    train_dataset = dataManager.MyDataset("./generated_images_1_im")
    test_dataset = dataManager.MyDataset("./generated_images_1_im")
    

    model = FCDenseNet57(8,device).to(device) # We train for 8 channels because we do not predict the Environement yet
    # logging.info("Trainer")
    training_name ="FCDenseNet57"+random_pokemon()
    folder = "runs\\"+datetime.now().strftime("%Y%m%d_%H%M%S_")+training_name
    trainer = Trainer(model, train_dataset, test_dataset, training_name, folder, 
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device = device,
        description = description)

    trainer.train_from_scratch()                

    print(Panel(Text("Script train.py End",justify = "center",style="bold magenta")))
    