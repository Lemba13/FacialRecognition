from loss import TripletLoss
import torch
import torch.nn as nn
import torch.nn.functional as F
import config

    
class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, 10)
        self.conv2 = nn.Conv2d(64, 128, 7)
        self.conv3 = nn.Conv2d(128, 128, 4)
        self.conv4 = nn.Conv2d(128, 256, 4)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256*15*15, 512)
        self.fc2 = nn.Linear(512, 32)
        #self.fc3 = nn.Linear(64,12)
        self.sigmoid = nn.Sigmoid()

    def convs(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, (2, 2))

        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size()[0], -1)

        x = self.fc1(x)
        x = self.fc2(x)
        #x = self.fc3(x)

        return x

    def forward_triple(self, x1, x2, x3):
        anchor_output = self.forward(x1)
        positive_output = self.forward(x2)
        negative_output = self.forward(x3)

        return anchor_output, positive_output, negative_output
    
    def forward_prediction(self, x1, x2):
        anchor_output = self.forward(x1)
        sample_output = self.forward(x2)

        return anchor_output, sample_output


if __name__ == "__main__":
    criteriion = TripletLoss()

    model1 = SiameseNet()
    print(model1)
    #print(model1)
    x1 = torch.randn((2, 1,config.IMAGE_HEIGHT, config.IMAGE_WIDTH))
    x2 = torch.randn((2, 1, config.IMAGE_HEIGHT, config.IMAGE_WIDTH))
    x3 = torch.randn((2, 1, config.IMAGE_HEIGHT, config.IMAGE_WIDTH))
    out1,out2,out3= model1.forward_triple(x1,x2,x3)
    
    out=criteriion(out1,out2,out3)
    
    print(out1.shape)
    print(out)
