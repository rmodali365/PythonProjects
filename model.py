from numpy import dtype
from pygame import K_LCTRL
# import agent
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):

    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size,output_size)
    
    def forward(self,x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self,fileName = 'model.pth'):
        modelFolderPath = './model'
        if not os.path.exists(modelFolderPath):
            os.makedirs(modelFolderPath)
        fileName = os.path.join(modelFolderPath,fileName)
        torch.save(self.state_dict(),fileName)

class QTrainer:
    def __init__(self,model,lr,gamma) :
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(),lr = self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self,state,action,reward,next_state,gameOver):
        state = torch.tensor(state,dtype = torch.float)
        action = torch.tensor(action,dtype = torch.long)
        next_state = torch.tensor(next_state,dtype = torch.float)
        reward = torch.tensor(reward,dtype = torch.float)

        if len(state.shape)==1:
            state = torch.unsqueeze(state,0)
            action = torch.unsqueeze(action,0)
            next_state = torch.unsqueeze(next_state,0)
            reward = torch.unsqueeze(reward,0)
            gameOver = (gameOver,)
            
        # Q = predicted Q values with current state
        pred = self.model(state)

        #Q_new = r + gamma * max(next_predicted Q Value)
        target = pred.clone()
        for idx in range(len(gameOver)):
            Q_new = reward[idx]
            if not gameOver[idx]:
                Q_new = reward[idx]+self.gamma*torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target,pred)
        loss.backward()

        self.optimizer.step()







