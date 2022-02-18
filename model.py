import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchinfo   import summary


class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate, device):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels,device = device))
        self.add_module('relu', nn.ReLU(True))
        self.add_module('conv', nn.Conv2d(in_channels, growth_rate, kernel_size=3,
                                          stride=1, padding=1, bias=True,device = device))
        self.add_module('drop', nn.Dropout2d(0.2))

    def forward(self, x):
        return super().forward(x)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers, device, upsample=False ):
        super().__init__()
        self.upsample = upsample
        self.layers = nn.ModuleList([DenseLayer(
            in_channels + i*growth_rate, growth_rate , device)
            for i in range(n_layers)])

    def forward(self, x):
        if self.upsample:
            new_features = []
            #we pass all previous activations into each dense layer normally
            #But we only store each dense layer's output in the new_features array
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            return torch.cat(new_features,1)
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1) # 1 = channel axis
            return x

class TransitionDown(nn.Sequential):
    def __init__(self, in_channels, device):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(num_features=in_channels, device = device))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, in_channels,
                                          kernel_size=1, stride=1,
                                          padding=0, bias=True,device = device))
        self.add_module('drop', nn.Dropout2d(0.2))
        self.add_module('maxpool', nn.MaxPool2d(2))

    def forward(self, x):
        return super().forward(x)

def center_crop(layer, max_height, max_width):
    _, _, h, w = layer.size()
    xy1 = (w - max_width) // 2
    xy2 = (h - max_height) // 2
    return layer[:, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]

class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels,device):
        super().__init__()
        self.convTrans = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=3, stride=2, padding=0, bias=True, device = device)

    def forward(self, x, skip):
        out = self.convTrans(x)
        out = center_crop(out, skip.size(2), skip.size(3))
        out = torch.cat([out, skip], 1)
        return out


class Bottleneck(nn.Sequential):
    def __init__(self, in_channels, growth_rate, n_layers, device):
        super().__init__()
        self.add_module('bottleneck', DenseBlock(
            in_channels, growth_rate, n_layers,device = device, upsample=True))

    def forward(self, x):
        return super().forward(x)


class FCDenseNet(nn.Module):
    def __init__(self, in_channels=3, down_blocks=(5,5,5,5,5),
                 up_blocks=(5,5,5,5,5), bottleneck_layers=5,
                 growth_rate=16, out_chans_first_conv=48, n_classes=12,device = "cuda"):
        super().__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        cur_channels_count = 0
        skip_connection_channel_counts = []

        ## First Convolution ##

        self.add_module('firstconv', nn.Conv2d(in_channels=in_channels,
                  out_channels=out_chans_first_conv, kernel_size=3,
                  stride=1, padding=1, bias=True))
        cur_channels_count = out_chans_first_conv

        #####################
        # Downsampling path #
        #####################

        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                DenseBlock(cur_channels_count, growth_rate, down_blocks[i],device = device))
            cur_channels_count += (growth_rate*down_blocks[i])
            skip_connection_channel_counts.insert(0,cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count, device = device))

        #####################
        #     Bottleneck    #
        #####################

        self.add_module('bottleneck',Bottleneck(cur_channels_count,
                                     growth_rate, bottleneck_layers, device = device))
        prev_block_channels = growth_rate*bottleneck_layers
        cur_channels_count += prev_block_channels

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks)-1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels, device = device))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]

            self.denseBlocksUp.append(DenseBlock(
                cur_channels_count, growth_rate, up_blocks[i],device = device,
                    upsample=True))
            prev_block_channels = growth_rate*up_blocks[i]
            cur_channels_count += prev_block_channels

        ## Final DenseBlock ##

        self.transUpBlocks.append(TransitionUp(
            prev_block_channels, prev_block_channels, device = device))
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]

        self.denseBlocksUp.append(DenseBlock(
            cur_channels_count, growth_rate, up_blocks[-1],device = device,
                upsample=False))
        cur_channels_count += growth_rate*up_blocks[-1]

        ## Softmax ##

        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
               out_channels=n_classes, kernel_size=1, stride=1,
                   padding=0, bias=True, device = device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.firstconv(x)

        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)

        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)

        out = self.finalConv(out)
        out = self.sigmoid(out)
        return out


def FCDenseNet57(n_classes,device):
    return FCDenseNet(
        in_channels=3, 
        down_blocks= (4, 4, 4, 4, 4),
        up_blocks  = (4, 4, 4, 4, 4),
        bottleneck_layers=4, growth_rate=12, out_chans_first_conv=48,
        n_classes=n_classes,
        device = device)

def FCDenseNet67(n_classes, device):
    return FCDenseNet(
        in_channels=3, down_blocks=(5, 5, 5, 5, 5),
        up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
        growth_rate=16, out_chans_first_conv=48, n_classes=n_classes,
        device = device)


def FCDenseNet103(n_classes, device):
    return FCDenseNet(
        in_channels=3, down_blocks=(4,5,7,10,12),
        up_blocks=(12,10,7,5,4), bottleneck_layers=15,
        growth_rate=16, out_chans_first_conv=48, n_classes=n_classes,
        device = device)


class MyModel(nn.Module):
    def __init__(self,device):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3,10,kernel_size=5,padding=2)
        self.Relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(10,10,kernel_size=5,padding=2)
        self.Relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(10,10,kernel_size=5,padding=2)
        self.Relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(10,8,kernel_size=5,padding=2)
        self.Sigmo = nn.Sigmoid()
        self.device = device

    def dense_block(self,n,k,x):
        x_chan = x.size()[1]
        print(x.size())
        for i in range(n):
            x = torch.cat([x,nn.ReLU(nn.Conv2d(k+x_chan,k,kernel_size=5,padding=2)(x))],1)
        return x

        
    def forward(self,x):
        x = DenseLayer(3,10,self.device)(x)
        x = DenseLayer(10,20,self.device)(x)
        x = DenseLayer(20,20,self.device)(x)
        x = DenseLayer(20,20,self.device)(x)
        x = DenseLayer(20,10,self.device)(x)
        x = DenseLayer(10,8,self.device)(x)


        return x


if __name__ == "__main__":
    model  =FCDenseNet57(11,"cuda:0")
    t = torch.zeros((1,3, 1024,1024),device="cuda:0")
    summary(model, (1,3, 512,512))
    


