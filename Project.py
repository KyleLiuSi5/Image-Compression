import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
from PIL import Image
import scipy.io
import scipy.misc
import pyae
from decimal import getcontext
import time

getcontext().prec = 10000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
block_size = 128
N = 128
M = 192
BATCH = 2
EPOCH = 1



class Kodim:
    def __init__(self):
        datasets = []
        batchs = []
        VALIDATION_SIZE = 1
        count = 0
        for filename in os.listdir("data/"):
          #if count >= 1000:
          #  break
            if filename.endswith(".png"):
                with Image.open(os.path.join("data/", filename)) as f: 

                    arr = np.array(f)
                    arr = arr.swapaxes(1,2)
                    arr = arr.swapaxes(0,1)
                    for i in range(int(arr.shape[1]/block_size)):
                        for j in range(int(arr.shape[2]/block_size)):
                            datasets.append(arr[:,block_size*i : block_size*(i+1),block_size*j : block_size*(j+1)])

        #self.validation_data = data[:int(VALIDATION_SIZE/batch_size)]
        datasets = np.array(datasets)
        self.train_data = datasets
#======================================================================================
#NonLinearTransform 
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(N, N, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(N, N, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        return out

class NonLinear(nn.Module):
    def __init__(self, ResidualBlock):
        super(NonLinear, self).__init__()
        self.inchannel = 3
        self.conv_52s = nn.Conv2d(3, N, kernel_size=5,padding=2)
        self.conv_52 = nn.Conv2d(N, N, kernel_size=5,padding=2)
        self.conv_52e = nn.Conv2d(N, M, kernel_size=5,padding=2)
        self.res = self.make_layer(ResidualBlock, 3, 2, stride=1)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_52s(x.float())
        out = self.res(out)
        out = self.conv_52(out)
        out = self.res(out)
        out = self.conv_52(out)
        out = self.res(out)
        out = self.conv_52e(out)
        return out
def NonLinearTransform():
    return NonLinear(ResidualBlock)

#======================================================================================
#inverse nonelinear
class InverseResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.ConvTranspose2d(N, N, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(N, N, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class InverseNonLinear(nn.Module):
    def __init__(self, ResidualBlock):
        super(InverseNonLinear, self).__init__()
        self.inchannel = 3
        self.conv_52s = nn.ConvTranspose2d(M, N, kernel_size=5,padding=2)
        self.conv_52 = nn.ConvTranspose2d(N, N, kernel_size=5,padding=2)
        self.conv_52e = nn.ConvTranspose2d(N, 3, kernel_size=5,padding=2)
        self.res = self.make_layer(ResidualBlock, 3, 2, stride=1)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_52s(x.float())
        out = self.res(out)
        out = self.conv_52(out)
        out = self.res(out)
        out = self.conv_52(out)
        out = self.res(out)
        out = self.conv_52e(out)
        return out
    
def InverseNonLinearTransform():
    return InverseNonLinear(ResidualBlock)
#===========================================================================
class MaskConv(nn.Module):
    def __init__(self, seq_module):
        super(MaskConv, self).__init__()
        self.seq_module = seq_module

    def forward(self, x):
        for module in self.seq_module:
            x = module(x)
            mask = torch.ByteTensor(x.size()).fill_(0)
            if x.is_cuda:
                mask = mask.cuda()
            #for i, length in enumerate(lengths):
                               
                #if (mask[i].shape[0] - length.shape[0]) > 0:
                    
                    #mask[i].narrow(2, length, mask[i].shape[0] - length).fill_(1) #這行有bug
                    
                    
            x = x.masked_fill(mask, 0)
        return x

#===========================================================================
#Entropy model

class HyperpriorModel(nn.Module):
    def __init__(self):
        super(HyperpriorModel, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(M, M, kernel_size=3,  padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(M, M, kernel_size=5,  padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(M, M, kernel_size=5,  padding=2, bias=False)
        )
        self.right = nn.Sequential(
            nn.ConvTranspose2d(M, int(3*M/2), kernel_size=3,  padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(int(3*M/2), 2*M, kernel_size=5,  padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(2*M, 2*M, kernel_size=3,  padding=1, bias=False)
        )
        
    def forward(self, x): 
        out = self.left(x.float())
        out = torch.round(out) #做Q #這邊可以輸出bitstream
        out = self.right(out)
        return out

class EntropyModel(nn.Module):
    def __init__(self):
        super(EntropyModel, self).__init__()
        self.inchannel = 3
        self.net = nn.Sequential(
            nn.Conv2d(4*M, 4*M, kernel_size=1,  padding=0, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(4*M, 4*M, kernel_size=1,  padding=0, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(4*M, 4*M, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(4*M, 4*M, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(4*M,9*M, kernel_size=1,  padding=0, bias=False)
        )

    def forward(self, x): 
        out = self.net(x.float())
        return out
    
net = NonLinearTransform().to(device)
net_inverse = InverseNonLinearTransform().to(device)
entropy_hyperprior = HyperpriorModel().to(device)
MaskConvLayer = MaskConv(nn.Sequential(nn.Conv2d(M, 2*M, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)))).to(device) #Autoregressive
entropy_model = EntropyModel().to(device)

try:
    net.load_state_dict(torch.load('Nonlinear.pth'))
    net_inverse.load_state_dict(torch.load('InverseNonlinear.pth'))
    entropy_hyperprior.load_state_dict(torch.load('HyperpriorModel.pth'))
    MaskConvLayer.load_state_dict(torch.load('MaskConvLayer.pth'))
    entropy_model.load_state_dict(torch.load('EntropyModel.pth'))

except:
    print("No .pth file!")
    

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(),lr=0.00005,weight_decay=0.5)
datasets = Kodim()

for epoch in range(EPOCH):
    print('\nEpoch: %d/%d' % (epoch + 1,EPOCH) )
    time_s = time.time()
    net.train()
    sum_loss = 0.0
    correct = 0.0
    total = 0.0
    
    for i in range(int(datasets.train_data.shape[0]/BATCH)): 
        
        #if i > 1:
            #break;
        
        inputs = torch.from_numpy(datasets.train_data[BATCH*i : BATCH*(i+1)])
        inputs = inputs.to(device)
        optimizer.zero_grad()

        outputs = net(inputs) ########################## Non-Linear Transform ##############################

        quantization = np.random.uniform(-0.5,0.5,(inputs.shape[0],M,128,128))  #Quantization 8 (9還需要做 後面才會用到)
        quantization = torch.from_numpy(quantization).to(device)
        outputs_quantizationed = outputs + quantization ####################### Quantization ##############################
        
        #這個部分是我們認為應該要加隨機分布的noise下去train
        #這樣test的時候做round也會等同於加上一個u(-0.5,0.5)的函數
        
        
        entropy_input1 = entropy_hyperprior(outputs_quantizationed) 
        entropy_input2 = MaskConvLayer(outputs) #Autoregressive
        
        entropy_input = torch.cat((entropy_input1, entropy_input2), 1)
        outputs_9M = entropy_model(entropy_input)
        
        
        #算術編碼
        
        im = outputs_9M.cpu().detach().numpy()
        msg = im.flatten()
        #print(msg.shape)
        min_ = msg.min()
        max_ = msg.max()
        msg = msg - min_

        hist, bin_edges = np.histogram(a = im, bins = range(0, int( -min_ +max_ +2)))
        frequency_table = {key: value for key, value in zip(bin_edges[0:int( -min_ +max_ +1)], hist)}
        AE = pyae.ArithmeticEncoding(frequency_table = frequency_table)
        #print("Output Bitstream")

        #編碼
        #encoded_msg, _ = AE.encode(msg = msg, probability_table = AE.probability_table) 
        #print(encoded_msg)
        
        #輸出BitStream############################
        
        #解碼
        #decoded_msg, _ = AE.decode(encoded_msg=encoded_msg, msg_length=len(msg), probability_table=AE.probability_table)
        #decoded_msg = np.reshape(decoded_msg, im.shape) 
        
        outputs = net_inverse(outputs_quantizationed) #Decoder

        loss = criterion(outputs.float(), inputs.float())
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
        
        print(BATCH*i+1,"/",datasets.train_data.shape[0],"Loss =",loss.item())
    time_e = time.time()
    print("Sum_Loss =",sum_loss,"   Cost",time_e-time_s,"s")
    
   
torch.save(net.state_dict(), "Nonlinear.pth")
torch.save(net_inverse.state_dict(), "InverseNonlinear.pth")
torch.save(entropy_hyperprior.state_dict(), "HyperpriorModel.pth")
torch.save(MaskConvLayer.state_dict(), "MaskConvLayer.pth")
torch.save(entropy_model.state_dict(), "EntropyModel.pth")

print("FINISH!!")
print("Model saved.")










