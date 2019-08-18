import os
import Game
import pygame as pg
import random
from collections import deque

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np




"""
USER NOTES: 
You can change the stage_num param at line 43 to change stages. 0 or 1 

Stage 0 best params: epsilon dec: .995 1000 episodes
Stage 1 best params: epsilon dec .997 2500 episodes
(these params are at lines 45 and 59 )

If want to increase runtime, you can choose to turn off the visual aspect of the game and read
command line output to gaugue performance. comment out game.show() on line 132

Speaking of performance, pygame updates as fast as your processor allows. That means you may want to think about
hopping into Game.py and increasing the pg.delay() parameter if the game updates too fast on your system. pygame
doesn't have many options to help with this (because it's kinda trash). The line to do this is Game.py line 132

Feel free to goof anound and mess with setting to get the result you are looking for
"""

if __name__ == '__main__':
	
	
	
	SCREEN_SIZE = (500, 500) # screen size has to be 500,500 or stages break bc I'm a dummy :)
	game = Game.Environment(vel = 10, screen_size = SCREEN_SIZE, stage_num = 1)
	action_size = 4 # this should be a val of 4 (check if thsi is correct)
	state_size = 10
	batch_size = 128
	episodes = 2500
	
	out_dir = 'model_out/maze_game'
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	# -------- CREATE AGENT --------
	class DQN:
		def __init__(self, state_size, action_size):
			self.state_size = state_size
			self.action_size = action_size
			self.memory = deque(maxlen = 20000) # change to liking
			self.gamma = .95 # how much do we discount future reward
			self.epsilon = 1.0
			self.epsilon_dec = .997 # subject to change depending on how good
			self.epsilon_floor = 0.00
			self.learning_rate = 0.001
			self.model = self.build_model()
		

		def build_model(self):
			model = Sequential()
			model.add(Dense(64, input_dim = self.state_size, activation = 'relu'))
			model.add(Dense(64, activation = 'relu'))
			model.add(Dense(self.action_size, activation = 'linear'))
			model.compile(loss = 'mse', optimizer = Adam(lr = self.learning_rate)) # loss subj to chng
			return model


		def remember(self, state, action, reward, next_state, done):
			self.memory.append((state, action, reward, next_state, done))


		def act(self, state):
			# explore
			if np.random.rand() <= self.epsilon:
				#print('eplore')
				return random.randrange(self.action_size)
			# exploit
			act_values = self.model.predict(state)
			#print('greed')
			return np.argmax(act_values[0])


		def replay(self, batch_size):
			# we want a random sample from our memory so our comp doesn't explode
			minibatch = random.sample(self.memory, batch_size)
			for state, action, reward, next_state, done in minibatch:
				# if game is over
				target = reward
				
				# if game is continuing, our reward is our reward and (discounted) preidcted future reward # just 2 values
				if not done:
					target = (reward + self.gamma*np.amax(self.model.predict(next_state)[0]))

				target_f = self.model.predict(state)
				
				target_f[0][action] = target

				self.model.fit(state, target_f, epochs = 1, verbose = 0) # we dont want to see keras fitting output
			
			if self.epsilon > self.epsilon_floor:
				self.epsilon = self.epsilon * self.epsilon_dec


		def load(self, name):
			self.model.load_weights(name)


		def save(self, name):
			self.model.save_weights(name)


# ---------- GAME LOOP ----------

agent = DQN(state_size, action_size)

done = False


for ep in range(episodes):
	state = np.array(game.reset())
	state = np.reshape(state, [1,state_size]) # vector to tensor
	for time in range(1000): # change up
		game.show()
		action = agent.act(state)
		next_state, reward, done, has_won = game.step(action)
		

		#reward = reward if not has_won else 100
		
		next_state = np.reshape(next_state, [1,state_size])
		agent.remember(state, action, reward, next_state, done)
		state = next_state
		
		if done:
			print(f'episode: {ep}/{episodes}, Dist From Goal: {game.logic.get_dist_to_goal()}, epsilon{agent.epsilon:.2f}\n')
			break
	file_dir = 'game_stats.csv'
	line = f'{game.logic.stage_num},{ep},{has_won},{game.logic.get_dist_to_goal()},{time},{agent.epsilon:.3f}\n'
	#file.write(line)
	#write_stats(data_file)
	if len(agent.memory) > batch_size:
		agent.replay(batch_size)