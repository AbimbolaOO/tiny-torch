import torch
import torch.nn as nn
import torch.nn.functional as F

__add__ = ['ANN']

class ANN(nn.Module):
      
  def __init__(self):
    
    super(ANN, self).__init__()
    
    hidden_1 = 258
    hidden_2 = 128
    hidden_3 = 64
    
    #The first convolution layer with 16 feature map
    self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
    #The secound convolutional layer with 32 fature map
    self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
    #The is maxpooling layer it help to reduce the x and y dimension of the output
    #image size of the secound convolution layer
    self.maxpool = nn.MaxPool2d(2,2)
    
    self.fc_1 = nn.Linear(32*7*7, hidden_1)
    self.fc_2 = nn.Linear(hidden_1, hidden_2)
    self.fc_3 = nn.Linear(hidden_2, hidden_3)
    self.fc_4 = nn.Linear(hidden_3, 10)
    
    #Here implemented dropout for the fully connected layers because it help the
    #to reduce the dead relu problems
    self.dropout1 = nn.Dropout(p = 0.2)
    self.dropout2 = nn.Dropout(p = 0.4)
    self.dropout3 = nn.Dropout(p = 0.2)
    
      
  def forward(self, x):
    
    x = F.relu(self.conv1(x))
    x = self.maxpool(x)
    x = self.maxpool(F.relu(self.conv2(x)))
    
    x = x.view(-1, 32*7*7)
    x = self.dropout1(F.relu(self.fc_1(x)))
    x = self.dropout2(F.relu(self.fc_2(x)))
    x = self.dropout3(F.relu(self.fc_3(x)))
    x = self.fc_4(x)
    
    return x
    