import torch
import torchvision
import torch.nn as nn
import csv

class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression example. You may need to double check the dimensions :)
    """
    #column_count = sum(1 for column in csv.reader('train.csv')) --> work to show the dimensions and stuff? 
    #print(column_count)
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 50) # What could that number mean!?!?!? Ask an officer to find out :)
        self.fc2 = nn.Linear(50, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''
        x (tensor): the input to the model
        '''
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


