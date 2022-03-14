from shutil import move
from typing import final
import torch
import random
import numpy as np
from collections import deque
from snakegame import SnakeGameAI, Direction, Point
from model import Linear_QNet,QTrainer
from helper import plot


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self):
        self.numberOfGames = 0
        self.epsilon = 0 #parameter to control randomness
        self.gamma = 0.9 # discount rate for loss function
        self.memory = deque(maxlen=MAX_MEMORY)  #popleft()
        self.model = Linear_QNet(11,256,3)
        self.trainer = QTrainer(self.model,lr=LR,gamma=self.gamma)


    def getState(self,snakeGame):
        head = snakeGame.snake[0]

        pointL = Point(head.x-20,head.y)
        pointR = Point(head.x+20,head.y)
        pointU = Point(head.x,head.y-20)
        pointD = Point(head.x,head.y+20)

        dirL = snakeGame.direction==Direction.LEFT
        dirR = snakeGame.direction==Direction.RIGHT
        dirU = snakeGame.direction==Direction.UP
        dirD = snakeGame.direction==Direction.DOWN

        state = [
            #Danger Straight
            (dirR and snakeGame.is_collision(pointR)) or
            (dirL and snakeGame.is_collision(pointL)) or
            (dirU and snakeGame.is_collision(pointU)) or
            (dirD and snakeGame.is_collision(pointD)),

            #Danger Right
            (dirR and snakeGame.is_collision(pointD)) or
            (dirL and snakeGame.is_collision(pointU)) or
            (dirU and snakeGame.is_collision(pointR)) or
            (dirD and snakeGame.is_collision(pointL)),

            #Danger Left
            (dirR and snakeGame.is_collision(pointU)) or
            (dirL and snakeGame.is_collision(pointD)) or
            (dirU and snakeGame.is_collision(pointL)) or
            (dirD and snakeGame.is_collision(pointR)),

            #snake direction
            dirL,
            dirR,
            dirU,
            dirD,
            
            #Food Locattion
            snakeGame.food.x < snakeGame.head.x, #food to the left
            snakeGame.food.x > snakeGame.head.x, # food to the right
            snakeGame.food.y < snakeGame.head.y, # food on top of snake
            snakeGame.food.y > snakeGame.head.y #food below snake 
        ] 

        return np.array(state,dtype=int)



    def remember(self,state,action,reward,next_state,gameOver):
        self.memory.append((state,action,reward,next_state,gameOver))

    def trainLongMemory(self):
        if len(self.memory)>BATCH_SIZE:
            mini_sample = random.sample(self.memory,BATCH_SIZE) #list of tuples
        else:
            mini_sample = self.memory


        states,actions,rewards,next_states,gameOvers = zip(*mini_sample)
        self.trainer.train_step(states,actions,rewards,next_states,gameOvers)


    def trainShortMemory(self,state,action,reward,next_state,gameOver):
        self.trainer.train_step(state,action,reward,next_state,gameOver)
        

    def getAction(self,state):
        self.epsilon = 80 - self.numberOfGames
        finalMove = [0,0,0]
        if random.randint(0,200)< self.epsilon:
            move = random.randint(0,2)
            finalMove[move] = 1
        else:
            state0 = torch.tensor(state,dtype = torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            finalMove[move] = 1

        return finalMove

def train():
    plot_scores = []
    plot_avgScores = []
    totalScore = 0
    record = 0
    agent = Agent()
    snakeGame = SnakeGameAI()
    while True:

        #get old state
        stateOld = agent.getState(snakeGame)

        #get the move
        finalMove = agent.getAction(stateOld)

        #perform move and get new state
        reward,gameOver,score = snakeGame.play_step(finalMove)
        stateNew = agent.getState(snakeGame)

        #train short memeory (only for one step)
        agent.trainShortMemory(stateOld, finalMove, reward, stateNew, gameOver)

        #remember and store in memory
        agent.remember(stateOld, finalMove, reward, stateNew, gameOver)

        if gameOver:
            #train the long memory (experience replay) and plot result
            snakeGame.reset()
            agent.numberOfGames+=1
            agent.trainLongMemory()
            if score>record:
                record = score
                agent.model.save()
                
            print('Game',agent.numberOfGames,'Score',score,'Record: ',record)

            plot_scores.append(score)
            totalScore+=score
            meanScore = totalScore/agent.numberOfGames
            plot_avgScores.append(meanScore)
            plot(plot_scores,plot_avgScores)



if __name__ == '__main__':
    train()




    