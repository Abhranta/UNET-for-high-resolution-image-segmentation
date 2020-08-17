import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image
import numpy as np
 
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

#Here we will be creating the dataloader clss
class data_set(Dataset):
    def __init__(self , csv_file  , root_dir , transform = None ):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        
        
    def __len__(self):
        return len(self.annotations) 
    
    
    def __getitem__(self , index):
        img_path = os.path.join(self.root_dir , self.annotations.iloc[index , 0])
        image = cv2.imread(img_path )
        
        if self.transform:
            image = self.transform(image)
           
            
        return image


#HERE WE WILL BE LOADING THE DATA
batch_size = 1
my_transforms = transforms.Compose([
    transforms.ToPILImage() , 
    transforms.Resize((1024 , 1024)) ,
    transforms.ToTensor()
])

my_transforms_grayscale = transforms.Compose([
    transforms.ToPILImage() , 
    transforms.Resize((1024 , 1024)) , 
    transforms.Grayscale(1) , 
    transforms.ToTensor()    
])

colored_dataset = data_set(csv_file = "image_annotation.csv" , root_dir = "...PATH_TO_TRAINING_IMAGES" , 
                          transform = my_transforms)

grayscale_dataset = data_set(csv_file = "label_annotation.csv" , root_dir = "...PATH_TO_GROUND_TRUTH_IMAGES" , 
                             transform = my_transforms_grayscale)

image_loader = DataLoader(dataset = colored_dataset , batch_size = batch_size , shuffle = False)
grayscale_loader = DataLoader(dataset = grayscale_dataset , batch_size = batch_size , shuffle = False)


#FROM HERE ON, I WILL BE CREATING THE MODEL
#FIRST I WILL DEFINE THE DOUBLE-CONVOLUTION LAYERS PRESENT BEFORE EACH MAXPOOL AND TRANSPOSE CONV LAYERS.
def double_conv(in_channels , out_channels):
  convoluted_data = nn.Sequential(
      
      nn.Conv2d(in_channels , out_channels , 3 , stride = 1 , padding = 1) ,
      nn.ReLU(inplace = True) , 
      nn.BatchNorm2d(out_channels) ,
      


      nn.Conv2d(out_channels , out_channels , 3 , stride = 1 , padding = 1) , 
      nn.ReLU(inplace = True) , 
      nn.BatchNorm2d(out_channels) 

    )

  return convoluted_data

  #HERE WILL BE THE MAIN UNET MODEL
class UNet(nn.Module):
    def __init__(self):
        super(UNet , self).__init__()

        self.maxpool = nn.MaxPool2d(2 , 2)

        self.down_conv1 = double_conv(3 , 16)
        self.down_conv2 = double_conv(3 , 32)
        self.down_conv3 = double_conv(32 , 64)
        self.down_conv4 = double_conv(64 , 128)
        self.down_conv5 = double_conv(128 , 256)
        self.down_conv6 = double_conv(256 , 512)
        self.down_conv7 = double_conv(512 , 1024)

        self.up_conv1 = nn.ConvTranspose2d(1024 , 512 , 2 , 2)
        self.up_conv1a = double_conv(1024 , 512)

        self.up_conv2 = nn.ConvTranspose2d(512 , 256 , 2 , 2)
        self.up_conv2a = double_conv(512 , 256)

        self.up_conv3 = nn.ConvTranspose2d(256 , 128 , 2 , 2)
        self.up_conv3a = double_conv(256 , 128)

        self.up_conv4 = nn.ConvTranspose2d(128 , 64 , 2 , 2)
        self.up_conv4a = double_conv(128 , 64)

        self.up_conv5 = nn.ConvTranspose2d(64 , 32 , 2 , 2)
        self.up_conv5a = double_conv(64 , 32)

        self.up_conv6 = nn.ConvTranspose2d(32 , 16 , 2 , 2)
        self.up_conv6a = double_conv(32 , 16)

        self.final_conv = nn.Conv2d(32 , 1 , 1 , 1)

    def forward(self , x):

        #ENCODER PASS
        x1 = self.down_conv1(x) #
        x2 = self.maxpool(x1)
        
        x3 = self.down_conv2(x) #
        x4 = self.maxpool(x3)
        
        x5 = self.down_conv3(x4) #
        x6 = self.maxpool(x5)
        
        x7 = self.down_conv4(x6) #
        x8 = self.maxpool(x7)
        
        x9 = self.down_conv5(x8) #
        x10 = self.maxpool(x9)
        
        x11 = self.down_conv6(x10) #
        x12 = self.maxpool(x11)
        
        x13 = self.down_conv7(x12)    
        

        #DECODER PASS
        x14 = self.up_conv1(x13)
        x14 = torch.cat([x14 , x11] , 1)
        x15 = self.up_conv1a(x14)

        x16 = self.up_conv2(x15)
        x16 = torch.cat([x16 , x9] , 1)
        x17 = self.up_conv2a(x16)

        x18 = self.up_conv3(x17)
        x18 = torch.cat([x18 , x7] , 1)
        x19 = self.up_conv3a(x18)

        x20 = self.up_conv4(x19)
        x20 = torch.cat([x20 , x5] , 1)
        x21 = self.up_conv4a(x20)

        x22 = self.up_conv5(x21)
        x22 = torch.cat([x22 , x3] , 1)
        x23 = self.up_conv5a(x22)

        x24 = self.up_conv6(x23)
        x24 = torch.cat([x24 , x1] , 1)
        x25 = self.up_conv6a(x24)

        output = self.final_conv(x25)

        return F.sigmoid(output )
  
  
unet = UNet()
unet.to("cuda")

#HERE I WILL BE DEFINING THE LOSS FUNCTION.
#FOR OUR IMAGE SEGMENTATION MODEL , I AM GONNA USE DICE-LOSS
def dice_loss(pred,target):
    smooth = 1
    numerator = 2 * torch.sum(pred * target)
    denominator = torch.sum(pred + target)
    loss =  1 - ((numerator + smooth) / (denominator + smooth))
    return loss

#I AM GONNA USE THE RMS_PROP OPTIMIZER WITH LEARNING RATE = 2e-4
optimizer = optim.RMSprop(unet.parameters(), lr=0.0002)


#HERE I WILL BE DEFINING MY TRAINING LOOP 
EPOCHS = 30
loss_list = []
epoch_list = []
num_training_images = 100

for epoch in range(EPOCHS):
    epoch_list.append(epoch)
    train_loss = 0
    for i in zip(image_loader , grayscale_loader):
        with torch.autograd.set_detect_anomaly(True):
            input_img = i[0].to("cuda")
            ground_truth = i[1].to("cuda") 

            optimizer.zero_grad()
            #FORWARD PASS
            output = unet(input_img)
            
            #CALCULATION OF LOSS
            loss  = dice_loss(output , ground_truth )
            
            #BACK-PROPAGATION
            loss.backward()

            #CHANGING THE WEIGHTS
            optimizer.step()
            train_loss += loss.item() * input_img.size(0)
            print(loss)
        
    
       
    train_loss = train_loss / num_training_images   
    loss_list.append(train_loss)
    print(epoch , train_loss)
torch.save(unet.state_dict() , r"...PATH.../weightFile.pt")
plt.plot(epoch_list , loss_list)
plt.show()
