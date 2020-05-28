from __future__ import print_function, division
import os, random, time, copy
from skimage import io, transform
import numpy as np
import os.path as path
import scipy.io as sio
from scipy import misc
from scipy import ndimage, signal
import scipy
import pickle

import math
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler 
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets, models, transforms

from libs_deblur.utils.metrics import *
from libs_deblur.utils.flow_functions import *
from libs_deblur.models.pixel_embedding_model import *
from libs_deblur.datasetMotionBlur import *
from libs_deblur.trainvalGaussBlur import *
import libs_deblur.pyblur
import warnings # ignore warnings
warnings.filterwarnings("ignore")



device ='cpu'
if torch.cuda.is_available(): device='cuda:0'
##
##initModel = PixelEmbedModelResNet18().to(device)

class SiamesePixelEmbed(nn.Module):
    def __init__(self, emb_dimension=64, filterSize=11, device='cpu', pretrained=False):
        super(SiamesePixelEmbed, self).__init__()
        self.device = device
        self.emb_dimension = emb_dimension  
        self.PEMbase = PixelEmbedModelResNet18(emb_dimension=self.emb_dimension, pretrained=pretrained)  
        self.rawEmbFeature1 = 0
        self.rawEmbFeature2 = 0        
        self.embFeature1_to_2 = 0
        self.embFeature1_to_2 = 0
        self.filterSize = filterSize
        self.filterSize2Channel = self.filterSize**2
                
        self.ordered_embedding = nn.Sequential(            
            nn.Conv2d(self.emb_dimension, self.filterSize2Channel, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(self.filterSize2Channel),     
            nn.Conv2d(self.filterSize2Channel, self.filterSize2Channel, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(self.filterSize2Channel),            
            nn.Conv2d(self.filterSize2Channel, self.filterSize2Channel, kernel_size=3, padding=1, bias=True)
        )
        
        
    def forward(self, inputs1, inputs2):        
        
        self.rawEmbFeature1 = self.PEMbase.forward(inputs1)
        
        self.embFeature1_to_2 = self.ordered_embedding(self.rawEmbFeature1)        
        self.embFeature1_to_2 = F.softmax(self.embFeature1_to_2, 1)

##        print("Self embed",self.embFeature1_to_2.size())        
        return self.embFeature1_to_2

initModel = SiamesePixelEmbed()

exp_dir = './libs_deblur' # experiment directory, used for reading the init model
project_name = 'demo' # name this project as "demo", used for saving files
path_to_root = './libs_deblur/dataset' # where to fetch data


batch_size = 16 # small batch size for demonstration; using larger batch size (like 56 and 64) for training

embedding_dim = 16 # dimension of the learned embedding space
kernel_size = 11 # the kernel size in the filter flow
cropSize = [64, 64] # patch size for training the model
sigmaMin=0.5
sigmaMax=2

lambda_norm = 0.1
total_epoch_num = 5 # total number of epoch in training
base_lr = 0.0005 # base learning rate

torch.cuda.device_count()
torch.cuda.empty_cache()

save_dir = os.path.join(exp_dir, project_name) # where to save the log file and trained models.
print(save_dir)    
if not os.path.exists(save_dir): os.makedirs(save_dir)
log_filename = os.path.join(save_dir, 'train.log')

class LossOrderedPairReconstruction(nn.Module):
    def __init__(self, device='cpu', filterSize=11):
        super(LossOrderedPairReconstruction, self).__init__()
        self.device = device
        self.filterSize = filterSize        
        self.filterSize2Channel = self.filterSize**2
        self.reconstructImage = 0
        
    def forward(self, image1, image2, filters_img1_to_img2):
        N,C,H,W = image1.size()
        self.reconstructImage = self.rgbImageFilterFlow(image1, filters_img1_to_img2)
        diff = self.reconstructImage - image2               
        diff = torch.abs(diff)       
        totloss = torch.sum(torch.sum(torch.sum(torch.sum(diff))))        
        return totloss/(N*C*H*W)
    
    
    def rgbImageFilterFlow(self, img, filters):                
        inputChannelSize = 1
        outputChannelSize = 1
        N = img.size(0)
        paddingFunc = nn.ZeroPad2d(int(self.filterSize/2))
        img = paddingFunc(img)        
        imgSize = [img.size(2),img.size(3)]
##        print("imgSize",img.size())
        out_R = F.unfold(img[:,0,:,:].unsqueeze(1), (self.filterSize, self.filterSize))
##        print("(N, out_R.size(1), imgSize[0]-self.filterSize+1, imgSize[1]-self.filterSize+1)",(N, out_R.size(1), imgSize[0]-self.filterSize+1, imgSize[1]-self.filterSize+1))
        out_R = out_R.view(N, out_R.size(1), imgSize[0]-self.filterSize+1, imgSize[1]-self.filterSize+1)
        
        #out_R = paddingFunc(out_R)
##        print("outR,filters",out_R.size(),filters.size())
        out_R = torch.mul(out_R, filters)
        out_R = torch.sum(out_R, dim=1).unsqueeze(1)

        out_G = F.unfold(img[:,1,:,:].unsqueeze(1), (self.filterSize, self.filterSize))
        out_G = out_G.view(N, out_G.size(1), imgSize[0]-self.filterSize+1, imgSize[1]-self.filterSize+1)    
        #out_G = paddingFunc(out_G)
        out_G = torch.mul(out_G, filters)
        out_G = torch.sum(out_G, dim=1).unsqueeze(1)

        out_B = F.unfold(img[:,2,:,:].unsqueeze(1), (self.filterSize, self.filterSize))
        out_B = out_B.view(N, out_B.size(1), imgSize[0]-self.filterSize+1, imgSize[1]-self.filterSize+1)    
        #out_B = paddingFunc(out_B)
        out_B = torch.mul(out_B, filters)
        out_B = torch.sum(out_B, dim=1).unsqueeze(1)
        return torch.cat([out_R, out_G, out_B], 1)


transform4Image = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((127.,127.,127.),(127.,127.,127.)) # (mean, std)
    ]) # (mean, std)

whole_datasets = {set_name: 
                  Dataset4MotionBlur(root_dir=path_to_root,
                                     size=cropSize, set_name=set_name, 
                                     transform=transform4Image, 
                                     sigmaMin=sigmaMin, sigmaMax=sigmaMax)
                  for set_name in ['train', 'val']}


dataloaders = {set_name: DataLoader(whole_datasets[set_name], 
                                    batch_size=batch_size,
                                    shuffle=set_name=='train', 
                                    num_workers=0) # num_work can be set to batch_size
               for set_name in ['train', 'val']}

dataset_sizes = {set_name: len(whole_datasets[set_name]) for set_name in ['train', 'val']}
print(dataset_sizes)



################## loss function ###################5
loss_1_to_2 = LossOrderedPairReconstruction(device=device, filterSize=kernel_size)

loss_l1norm = nn.L1Loss(size_average=True)

optimizer_ft = optim.Adam([{'params': initModel.PEMbase.parameters()},
                           {'params': initModel.ordered_embedding.parameters(), 'lr': base_lr},                           
                         ], lr=base_lr)


# Decay LR by a factor of 0.5 every int(total_epoch_num/5) epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=int(total_epoch_num/5), gamma=0.5)

fn = open(log_filename,'w')
fn.write(log_filename+'\t'+device+'\n\n')
#fn.write(path.basename(__file__)+'\n\n')
fn.close()
file_to_note_bestModel = os.path.join(save_dir,'note_bestModel.log')
fn = open(file_to_note_bestModel, 'w')
fn.write('Record of best models on the way.\n')
fn.close()

model_ft = train_model(initModel, dataloaders, dataset_sizes, 
                       loss_1_to_2, 
                       optimizer_ft, exp_lr_scheduler,
                       num_epochs=total_epoch_num, 
                       work_dir=save_dir, device=device)


