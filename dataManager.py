#create the data tensors from an image folder
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="true"
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset 

import logging
from rich import print
from rich.panel import Panel
from rich.text import Text
from rich.console import Group


def loadImage(filename):
    """
    Loads an image file and converts it to a usable tensor. This means permuting Heigth,Width,Channel to Channel,Heigth,Width 
    """
    img = cv2.imread(filename,cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    # print("*** LOADING "+ filename)
    # print(img.shape)
    # print(img.dtype)
    # print(img[0][0])
    if(img.dtype == "uint16"): 
        img = img.astype("uint8") #when a png file is read this is needed 
    
    # print("LOAD"+filename)
    x = torch.from_numpy(img) #do not likes uint16
    # tensor width,height,channels
    x=x.permute(2,0,1)  # tensor channels,width,height
    
    x = x / 255
    x = x - 0.5
    # print("as ")
    # print(x.size())
    # print(x.dtype)
    
    
    return x

def convert_to_im(tens,type = "uint16"):
    """
    Convert a tensor into a numpy array that can be saved or seen as an image.
    choose uint16 for png or float32 for exr.  
    """
    z= tens.detach()
    z = z + 0.5
    
    if len(z.size()) == 4:
        z=z[0]    
    if len(z.size()) == 3:
        z = z.permute(1,2,0)
    if len(z.size()) == 2:
        # z = np.reshape(z,(z.shape[0],z.shape[1],1))
        pass

    z =z.numpy()
    # print("CONVERTING from")
    # print(z.shape)
    # print(z.dtype)

    if type == "uint16":
        z = z *65535
        z = z.astype("uint16")
    elif type == "float32":
        z = z *255

    # print(z[0][0])
    
    return z


def showImage(image,type="uint16"):
    """
    Converts a tensor to a numpy array to then show it. 
    """
    print("Showing a tensor of size {}".format(image.size()))
    z = convert_to_im(image,type)
    cv2.imshow('img',z)
    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows()

def saveImage(filename, image):
    """
    Converts a tensor to an image to save it as an image
    """
    extension =  filename.split(".")[-1]
    if extension =="exr":
        z = convert_to_im(image, "float32")
    elif extension =="png":
        z = convert_to_im(image, "uint16")
    cv2.imwrite(filename,z)


class MyDataset(Dataset):
    """Loads the training and testing data from a given image_folder.
    It is made to load data created by my blender script containing groups of images as listed un self.data_list.
    This class contains functions to check the integrity of the image groups and functions to extract the data from the outputs of the model.
    """
    def __init__(self,image_folder):
        self.image_folder = image_folder
        filenames = os.listdir(self.image_folder)
        self.nb_files = len(filenames)
        self.data_list = ["0_Environment.exr","1_Renders.exr","2_Base_color.png","3_Normals.png","4_Roughness.png","5_Metallic.png"]
        
        self.nb_data = int(len(filenames)/len(self.data_list))
        
        #print(f"{self.nb_files} files in data folder")
        
    def __len__(self):
        """The length of the dataset is not the number of images. It is the number of groups"""
        return self.nb_data  

    def separate_into_images(self,tensor):
        """ takes a tensor with 8 channels in and transforms it into the 4 images base, normals, rougness, metal
            Each images hase 3 channels beceause this is intended to save the results of the model 
        """
        tens  = tensor[0].detach()
        reshaped = []
        for ten in tens:
            ten = ten.view(1,ten.size()[0],ten.size()[1])
            #showImage(ten)
            reshaped.append(ten)
        base = torch.cat((reshaped[0],reshaped[1],reshaped[2]),0) 
        normals = torch.cat((reshaped[3],reshaped[4],reshaped[5]),0) 
        roughness = torch.cat((reshaped[6],reshaped[6],reshaped[6]),0) 
        metal = torch.cat((reshaped[7],reshaped[7],reshaped[7]),0) 
        out = dict()
        out["base"] = base 
        out["normals"] = normals 
        out["roughness"] = roughness 
        out["metal"] = metal 
        return out


    def load_group(self,folder,index):
        """
        For a group of images from the folder creates an input tensor from the Render image and a output tensor from all the baked images.
        We only keep one channel from the roughness and metalic pictures as their data is simply a matrice of float.
        
        """
        
        file = str(index).zfill(6) + "_" + "1_Renders.exr"
        filepath = os.path.join(self.image_folder,file)
        input_tensor = loadImage(filepath) 

        file = str(index).zfill(6) + "_" + "0_Environment.exr"
        filepath = os.path.join(self.image_folder,file)
        env_tensor = loadImage(filepath) 

        file = str(index).zfill(6) + "_" + "2_Base_color.png"
        filepath = os.path.join(self.image_folder,file)
        #print(filepath)
        base_tensor = loadImage(filepath) 

        file = str(index).zfill(6) + "_" + "3_Normals.png"
        filepath = os.path.join(self.image_folder,file)
        normals_tensor = loadImage(filepath)


        file = str(index).zfill(6) + "_" + "4_Roughness.png"
        filepath = os.path.join(self.image_folder,file)
        roughness_tensor = loadImage(filepath)
        roughness_tensor = roughness_tensor[0]
        roughness_tensor = roughness_tensor.view(1,roughness_tensor.size()[0],roughness_tensor.size()[1])
        
        
        file = str(index).zfill(6) + "_" + "5_Metallic.png"
        filepath = os.path.join(self.image_folder,file)
        metal_tensor = loadImage(filepath)
        metal_tensor = metal_tensor[0]
        metal_tensor = metal_tensor.view(1,metal_tensor.size()[0],metal_tensor.size()[1])

        output_tensor = torch.cat((base_tensor,normals_tensor,roughness_tensor,metal_tensor),0) 
        return input_tensor , output_tensor ,  env_tensor

    def __getitem__(self, index) :
        """Returns 3 tensors: the input image tensor, the output channels and the environment tensor
        """
        return self.load_group(self.image_folder,index)



    def check_data(self):
        """Checks if the pictures in the folder follow the expected formats and names."""
        filenames = os.listdir(self.image_folder)
                
        nb_data = int(len(filenames)/len(self.data_list))
        nb_missing=0
        for i in range(nb_data):
            for name in self.data_list:
                
                file = str(i).zfill(6) + "_" + name
                if not os.path.isfile(os.path.join(self.image_folder,file)):
                    print(f"[red bold]{file}")
                    nb_missing +=1
        print(Panel(Group(
            f"Images at : {self.image_folder}",
            f"{len(filenames)} files in data folder",
            f"{'[red bold]' if nb_missing>0 else ''}There are {nb_missing} missing images",
            f"{self.nb_data} Data groups"
            ),title=f"Dataset Check at {self.image_folder}"))
        
        if nb_missing > 0:
            print("[red bold]Error : Missing Images in the dataset folder")
            raise Exception
                
            

          



if __name__ == "__main__":
    from torch.utils.data import DataLoader

    folder = "./generated_images"
    data = MyDataset(folder)
    data.check_data()

    # img = "./generated_images\\339_5_Metallic.png"
    # x = loadImage(img)
    # print(x)
    # showImage(x)
    for i in range(10):
        train_dataloader = DataLoader(data, batch_size=1, shuffle=True)
        in_tens, out_tens, ext_tens = next(iter(train_dataloader))
        print("****")
        in_tens = (in_tens +0.5) * 255 
        # showImage(in_tens /2 ,"float32")
        # showImage(in_tens * 1 ,"float32")
        # showImage(in_tens * 1.1 ,"float32")
        # showImage(ext_tens)

        print(in_tens.size())
        dic = data.separate_into_images(out_tens)

        print(torch.mean(in_tens))
        print(torch.max(in_tens))

        in_tens = in_tens / torch.max(in_tens)
        in_tens2 = in_tens / torch.mean(in_tens)
        
        in_tens = (in_tens /255) -0.5 
        in_tens2 = (in_tens2 /255) -0.5 
        showImage( torch.cat((in_tens,in_tens2),3)  ,"float32")
        
        # showImage(ext_tens,"float32")#exr images need to be seen in float32 for correct exposure
        # showImage(dic["base"])
    
