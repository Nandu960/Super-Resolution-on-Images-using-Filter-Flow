import os, random, time, copy
from skimage import io, transform
import numpy as np
import os.path as path
import scipy.io as sio
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler 
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets, models, transforms


##from torchsummary import summary

    
class Classifier(nn.Module):
    def __init__(self, inputDim=128, num_class=10):
        super(Classifier, self).__init__()
        self.num_class = num_class
        self.inputDim = inputDim        
        self.dropoutRate = 0.5
        
        if self.dropoutRate==0:
            self.feature_for_cls = nn.Sequential(
                nn.Conv2d(inputDim, num_class, 1)
            )
            
        else:
            self.feature_for_cls = nn.Sequential(
                nn.Dropout(self.dropoutRate),
                nn.Conv2d(inputDim, num_class, 1)
            )
        
    def forward(self, inputs):
        out = self.feature_for_cls(inputs)        
        return out        
    
    
    
    
    

    
    
##    
##    
class PixelEmbedModelResNet18(nn.Module):
    def __init__(self, emb_dimension=64, pretrained=True):
        super(PixelEmbedModelResNet18, self).__init__()
        self.emb_dimension = emb_dimension        
        
        
        self.layer0= nn.Sequential(
            nn.Conv2d(3, int(emb_dimension/2), 
                      kernel_size=(3,3), stride=1, padding=True, dilation=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2))
        )
        self.layer1= nn.Sequential(
            nn.Conv2d(32, int(emb_dimension/2), 
                      kernel_size=(3,3), stride=1, padding=True, dilation=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2))
        )
        self.layer2= nn.Sequential(
            nn.Conv2d(32, int(emb_dimension/2), 
                      kernel_size=(3,3), stride=1, padding=True, dilation=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2))
        )
        self.layer3= nn.Sequential(
            nn.Conv2d(32, int(emb_dimension/2), 
                      kernel_size=(3,3), stride=1, padding=True, dilation=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2))
        )
        self.layer4= nn.Sequential(
            nn.Conv2d(32, int(emb_dimension/2),
                      kernel_size=(3,3), stride=1, padding=True, dilation=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2))
        )
        
                
        
                
        
        self.streamTwo_feats = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
        )
        
                
        
        
        
        self.emb = nn.Sequential(
            nn.Conv2d(int(emb_dimension*2+32), emb_dimension, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(emb_dimension),
            
            nn.Conv2d(emb_dimension, emb_dimension, kernel_size=3, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(emb_dimension),

            nn.Conv2d(emb_dimension, emb_dimension, kernel_size=3, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(emb_dimension)
            
        )
        
        
    def forward(self, inputs):
        self.interp_x4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  
##        self.interp_x8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)  

        #print("Inputs",inputs)
        input_size = inputs.size()
##        print("Input",input_size)
        out = self.layer0(inputs)
##        print("Layer Zero output:",out.size())
        out_stream2 = self.streamTwo_feats(inputs)
        

        #print(out.size())
        out_layer1 = self.layer1(out)
        out = self.layer1(out)        
           
        
        
        out_layer2 = self.layer2(out)
        out = self.layer2(out)
        
        out_layer3 = self.layer3(out)
        out = self.layer3(out)
        out_layer4 = self.layer4(out)
        out = self.layer4(out)
        
        

        
        
       
        
        
##        self.interp = nn.Upsample(scale_factor=8, mode='bicubic', align_corners=True)        
##        out = self.interp(out)    
            
##        print("2nd Stream",out_stream2.size())
##        print(out_layer1.size())
##        print(out_layer2.size())
##        print(out_layer3.size())
        
        out = torch.cat([out_stream2, out_layer1, out_layer2, out_layer3, out], 1)
        out=self.emb(out)
        print("Out",out.size())
                
        return out      
    
##
##device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
##model =PixelEmbedModelResNet18().to(device)
##
##print(summary(model,(3,64,64)))
