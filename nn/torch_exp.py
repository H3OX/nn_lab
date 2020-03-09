import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchsummary import summary

from sklearn.model_selection import train_test_split
import numpy as np
import joblib
dataset = joblib.load('../data/tess_rvds_savee.pkl')

X = t.from_numpy(np.expand_dims(dataset.iloc[:, 1:].values, axis=2))
y = t.from_numpy(dataset.iloc[:, 0].values)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = DataLoader(X_train, batch_size=64, shuffle=True)
X_test = DataLoader(X_test, batch_size=64, shuffle=False)


class AudioNet(nn.Module):
    def __init__(self):
        super(AudioNet, self).__init__()
        self.conv1 = nn.Conv1d(30, 128, 3)
        self.bnorm1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 64, 3)
        self.mp1 = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout(0.5)
        self.bnorm2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 32, 3)
        self.conv4 = nn.Conv1d(32, 16, 3)
        self.mp2 = nn.MaxPool1d(2)
        self.out = nn.Linear(16*26, 29)
        self.out1 = nn.Linear(100, 7)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.bnorm1(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.mp1(x)
        x = self.drop1(x)
        x = self.bnorm2(x)
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        y = self.mp2(x)
        return y


net = AudioNet()

for x in X_train:
    print(net.forward(x.T.float()))