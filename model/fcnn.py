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
    

class conv_layer1(nn.Module):
    def __init__(self, in_channels, num_channels=6, num_filters=14, use_relu = True):
        super(conv_layer1, self).__init__()
        kernel_size_1 = [5, 5]
        kernel_size_2 = [3, 3]
        strides1 = [2, 2]
        strides2 = [1, 1]
        self.use_relu = use_relu
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.zeropadding1 = nn.ZeroPad2d(padding=2)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=num_filters*num_channels, 
                               kernel_size=kernel_size_1, stride=strides1, padding="valid", bias= False)
        self.bn2 = nn.BatchNorm2d(num_filters*num_channels)
        self.relu = nn.ReLU()
        self.zeropadding2 = nn.ZeroPad2d(padding=1)
        self.conv2 = nn.Conv2d(num_filters * num_channels, num_filters * num_channels, 
                               kernel_size=kernel_size_2, stride=strides2, padding="valid", bias = False)
        self.bn3 = nn.BatchNorm2d(num_filters * num_channels)
        self.maxpooling = nn.MaxPool2d(kernel_size=(2,2), padding=0)

    def forward(self, x):
        x = self.bn1(x)
        x = self.zeropadding1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        if self.use_relu:
            x = self.relu(x)
        x = self.zeropadding2(x)
        x = self.conv2(x)
        x = self.bn3(x)
        if self.use_relu:
            x = self.relu(x)
        x = self.maxpooling(x)
        return x
    

class conv_layer2(nn.Module):
    def __init__(self, in_channels, num_channels=6, num_filters=28, use_relu = True):
        super(conv_layer2, self).__init__()
        kernel_size = [3, 3]
        strides = [1, 1]
        self.use_relu = use_relu
        self.zeropadding = nn.ZeroPad2d(padding=1)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels= num_filters * num_channels,
                               kernel_size= kernel_size, stride=strides, padding='valid', bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters * num_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_filters*num_channels, num_filters*num_channels, kernel_size=kernel_size,
                               stride=strides, padding='valid', bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters*num_channels)
        self.maxpooling = nn.MaxPool2d(kernel_size=(2, 2), padding=0)

    def forward(self, x):
        x = self.zeropadding(x)
        x = self.conv1(x)
        x = self.bn1(x)
        if self.use_relu:
            x = self.relu(x)
        x = self.zeropadding(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.use_relu:
            x = self.relu(x)
        x = self.maxpooling(x)
        return x
    

class conv_layer3(nn.Module):
    def __init__(self, in_channels, num_channels=6, num_filters=56, learn_bn=True, wd=1e-4, use_relu=True):
        super(conv_layer3, self).__init__()
        kernel_size = [3, 3]
        strides = [1, 1]
        self.use_relu = use_relu
        self.zeropadding = nn.ZeroPad2d(padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.conv1 = nn.Conv2d(in_channels, num_channels*num_filters, kernel_size=kernel_size,
                               stride=strides, padding='valid', bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels*num_filters)

        self.conv2 = nn.Conv2d(num_channels*num_filters, num_channels*num_filters, kernel_size=kernel_size,
                               stride=strides, padding='valid', bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels*num_filters)

        self.conv3 = nn.Conv2d(num_channels*num_filters, num_channels*num_filters, kernel_size=kernel_size,
                               stride=strides, padding='valid', bias=False)
        self.bn3 = nn.BatchNorm2d(num_channels*num_filters)

        self.conv4 = nn.Conv2d(num_channels*num_filters, num_channels*num_filters, kernel_size=kernel_size,
                               stride=strides, padding='valid', bias=False)
        self.bn4 = nn.BatchNorm2d(num_channels*num_filters)

        self.maxpooling = nn.MaxPool2d(kernel_size=(2, 2), padding=0)
    
    def forward(self, x):
        x = self.zeropadding(x)
        x = self.conv1(x)
        x = self.bn1(x)
        if self.use_relu:
            x = self.relu(x)
        x = self.dropout(x)

        x = self.zeropadding(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.use_relu:
            x = self.relu(x)
        x = self.dropout(x)

        x = self.zeropadding(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.use_relu:
            x = self.relu(x)
        x = self.dropout(x)

        x = self.zeropadding(x)
        x = self.conv4(x)
        x = self.bn4(x)
        if self.use_relu:
            x = self.relu(x)
        x = self.maxpooling(x)
        return x
    

class attention_layer(nn.Module):
    def __init__(self, in_channel, ratio):
        super(attention_layer, self).__init__()
        self.linear1 = nn.Linear(in_channel, in_channel//ratio, bias=True)
        self.linear2 = nn.Linear(in_channel//ratio, in_channel, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.Globalmaxpooling = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        channel = x.shape[1]
        batch = x.shape[0]

        avgpool = torch.mean(x, dim = (2,3))  # GlobalAveragePooling2D
        # avgpool = avgpool.reshape((batch, 1, 1, channel))
        avgpool = self.linear1(avgpool)
        avgpool = self.linear2(avgpool)

        maxpool = self.Globalmaxpooling(x).view(batch, -1)
        # maxpool = maxpool.reshape((batch, channel))
        maxpool = self.linear1(maxpool)
        maxpool = self.linear2(maxpool)

        cbam_feature = torch.add(avgpool, maxpool)
        cbam_feature = self.sigmoid(cbam_feature)
        cbam_feature = cbam_feature.reshape((batch, channel, 1, 1))
        x = torch.multiply(x, cbam_feature)

        return x


class FCNN(nn.Module):
    def __init__(self, num_classes, input_shape=[16, 3, 128, None], num_filters=[48, 96, 192]):
        super(FCNN, self).__init__()
        self.convPath1 = conv_layer1(input_shape[1], num_channels=input_shape[1], num_filters=num_filters[0], use_relu=True)
        self.convPath2 = conv_layer2(input_shape[1]*num_filters[0], num_channels=input_shape[1], num_filters=num_filters[1], use_relu=True)
        self.convPath3 = conv_layer3(input_shape[1]*num_filters[1], num_channels=input_shape[1], num_filters=num_filters[2], use_relu=True)

        self.resnet = resnet_layer(in_channels=input_shape[1]*num_filters[2], out_channels=num_classes, stride=1, kernel_size=1, use_relu=True)    # learn_bn = False
        self.attention = attention_layer(in_channel=num_classes, ratio=2)
        self.bn = nn.BatchNorm2d(num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.convPath1(x)
        x = self.convPath2(x)
        x = self.convPath3(x)
        x = self.resnet(x)
        x = self.bn(x)
        x = self.attention(x)
        x = torch.mean(x, dim=(2,3))
        x = self.softmax(x)
        ret = {
            'logits': x,
        }
        return ret




