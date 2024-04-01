import torch
import torch.nn as nn

class resnet_layer(nn.Module):
    def __init__(self, in_channels, out_channels = 16, kernel_size = 3, stride = 1, use_relu = True):
        super(resnet_layer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(1,1), bias=False)
        if kernel_size == 1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, bias=False)
        self.relu = nn.ReLU()
        self.ifrelu = use_relu

    def forward(self, x):
        x = self.bn(x)
        if self.ifrelu:
            x = self.relu(x)
        x = self.conv(x)
        return x


def pad_depth(inputs):
    y = torch.zeros_like(inputs, device=inputs.device)
    return y

def My_freq_split1(x):  # [B, C, F, T]
    return x[:,:,0:64,:]


def My_freq_split2(x):
    return x[:,:,64:128,:]


class ResNetBlock(nn.Module):
    def __init__(self, stack_num, block_num, num_filters):
        super(ResNetBlock, self).__init__()
        self.stack_num = stack_num
        self.block_num = block_num
        strides = 1

        if stack_num > 0 and block_num == 0:
            strides = [1,2]
            self.conv1 = resnet_layer(int(num_filters/2), num_filters, stride=strides)
        else:
            self.conv1 = resnet_layer(num_filters, num_filters, stride=strides)
        self.conv2 = resnet_layer(num_filters, num_filters, stride=1)
        if stack_num > 0 and block_num == 0:
            self.avgpooling = nn.AvgPool2d(kernel_size=(3,3), stride=[1,2], padding=(1,1))

    def forward(self, residual):
        conv = self.conv1(residual)
        conv = self.conv2(conv)
        if self.stack_num > 0 and self.block_num == 0:
            residual = self.avgpooling(residual)
            padding = pad_depth(residual)
            residual = torch.concat((residual, padding), dim = 1)
        residual = torch.add(conv, residual)
        return residual

            


class ResNet(nn.Module):
    def __init__(self, num_classes, num_filters=48, num_res_blocks = 2):
        super(ResNet, self).__init__()
        self.num_filters = num_filters
        self.num_res_blocks = num_res_blocks
        self.split_path1 = resnet_layer(in_channels=3, out_channels=self.num_filters, stride=[1,2], use_relu=False)
        self.split_path2 = resnet_layer(in_channels=3, out_channels=self.num_filters, stride=[1,2], use_relu=False)
        self.resblock_path1 = nn.ModuleList()
        self.resblock_path2 = nn.ModuleList()
        for stack_num in range(4):
            for block_num in range(num_res_blocks):
                self.resblock_path1.append(ResNetBlock(stack_num=stack_num, block_num=block_num, num_filters=self.num_filters))
                self.resblock_path2.append(ResNetBlock(stack_num=stack_num, block_num=block_num, num_filters=self.num_filters))
            self.num_filters *= 2
        
        # 输入通道为384
        self.num_filters = int(self.num_filters/2)
        self.outputpath = resnet_layer(in_channels=self.num_filters, out_channels=2*self.num_filters, kernel_size=1, stride=1)
        self.classifier = resnet_layer(2*self.num_filters, num_classes, stride=1, kernel_size=1, use_relu=False)
        self.bn = nn.BatchNorm2d(num_features=num_classes)
        self.avgpooling = nn.AvgPool2d(kernel_size=(3,3),stride=[1,2], padding=(1,1))
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        split1 = My_freq_split1(x)
        split2 = My_freq_split2(x)
        ResidualPath1 = self.split_path1(split1)
        ResidualPath2 = self.split_path2(split2)
        for stack_num in range(4):
            for block_num in range(self.num_res_blocks):
                ConvPath1 = self.resblock_path1[stack_num * self.num_res_blocks + block_num](ResidualPath1)
                ConvPath2 = self.resblock_path2[stack_num * self.num_res_blocks + block_num](ResidualPath2)
                if stack_num > 0 and block_num == 0:
                    ResidualPath1 = self.avgpooling(ResidualPath1)
                    padding1 = pad_depth(ResidualPath1)
                    ResidualPath1 = torch.concat((ResidualPath1, padding1), dim=1)  # channel concat

                    ResidualPath2 = self.avgpooling(ResidualPath2)
                    padding2 = pad_depth(ResidualPath2)
                    ResidualPath2 = torch.concat((ResidualPath2, padding2), dim=1)

                ResidualPath1 = torch.add(ConvPath1, ResidualPath1)
                ResidualPath2 = torch.add(ConvPath2, ResidualPath2)

        ResidualPath = torch.concat((ResidualPath1, ResidualPath2), dim=2) # Frequency
        OutputPath = self.outputpath(ResidualPath)
        OutputPath = self.classifier(OutputPath)
        OutputPath = self.bn(OutputPath)
        OutputPath = torch.mean(OutputPath,dim=(2,3))
        x = self.softmax(OutputPath)
        ret = {
            'logits': x
        }
        return ret