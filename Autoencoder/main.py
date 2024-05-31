import os
import time
import torch
import pandas as pd
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

device = torch.device('cpu')

print('Device:', device)

# Load and preprocess data
df = pd.read_csv('data.csv', names=['x1', 'x2', 'x3'])
valData = pd.read_csv('val_data.csv', names=['x1', 'x2', 'x3'])
threshold1 = df['x1'].mean() + 2 * df['x1'].std()
threshold2 = df['x2'].mean() + 2 * df['x2'].std()
threshold3 = df['x3'].mean() + 2 * df['x3'].std()
df = df[(df['x1'] < threshold1) & (df['x2'] < threshold2) & (df['x3'] < threshold3)]

X_train = df.values
y_train = df.values
X_val = valData.values
y_val = valData.values


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

X_train = torch.FloatTensor(X_train).to(device)
y_train = torch.FloatTensor(y_train).to(device)
X_val = torch.FloatTensor(X_val).to(device)
y_val = torch.FloatTensor(y_val).to(device)

# Define neural network
class Net(nn.Module):
    def __init__(self, input_size=3, hidden_size1=64, hidden_size2=256, hidden_size3=512, encoding_size=1024, output_size=3):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, encoding_size)
        self.fc5 = nn.Linear(encoding_size, hidden_size3)
        self.fc6 = nn.Linear(hidden_size3, hidden_size2)
        self.fc7 = nn.Linear(hidden_size2, hidden_size1)
        self.fc8 = nn.Linear(hidden_size1, output_size)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')  
                init.constant_(m.bias, 0.01)  

    def encoder(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x
    
    def decoder(self, x):
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)
        return x
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == '__main__':
    net = Net().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.5)

    if os.path.exists('model/10bit.pth'):
        torch.load('model/10bit.pth')
    else:
        epoch = 0
        while True:
            #train
            outputs = net(X_train)
            loss = criterion(outputs, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
                
            print('Epoch [{}], Loss: {:.4f}'.format(epoch+1, loss.item()))
            if epoch >= 5000:
                break
            epoch += 1


        torch.save(net.state_dict(), 'model/10bit.pth')

    #test the model on validation data
    net.eval()
    with torch.no_grad():
        start = time.time()
        y_pred = net(X_val)
        end = time.time()
        loss = criterion(y_pred, y_val)
        print('Validation loss:', loss.item())
        print('Time:', end - start)
