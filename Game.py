import pygame as pg
import numpy as np
import random
from math import radians, sin, cos

def points_intersect(line_a, line_b):
	
	p0x = line_a[0][0] 
	p0y = line_a[0][1]
	p1x = line_a[1][0]
	p1y = line_a[1][1]
	p2x = line_b[0][0]
	p2y = line_b[0][1]
	p3x = line_b[1][0]
	p3y = line_b[1][1]

	A1 = p1y - p0y
	B1 = p0x - p1x
	C1 = A1 * p0x + B1 * p0y
	A2 = p3y -p2y
	B2 = p2x - p3x
	C2 = A2 * p2x + B2 * p2y
	
	denominator = A1*B2 - A2*B1
	
	# if denom is 0, lines are parralel or colinear... Technically if they are
	# colinear then they are touching, but... hey nothing here is perfect, now is it
	if denominator == 0:
		return (-1,-1)

	x_int = (B2*C1 -B1*C2)/denominator
	y_int = (A1*C2 - A2*C1)/denominator
	

	# since we are playing with line segments, have to make sure the segments intersect only within segemnts
	# to do this, we take the distance of line on x and y to where it intersects
	
	"""
	This will break down for horizontal and vertical lines... and uh we ONLY have horizontal and vert lines
	To fix we need some really goofy checks
	"""

	
	ratio_x = -1.0
	ratio_y = -1.0
	ratio_x2 = -1.0
	ratio_y2 = -1.0

	if (p1x - p0x) != 0:
		ratio_x = (x_int - p0x)/(p1x - p0x)
		
	if (p1y - p0y) != 0:
		ratio_y = (y_int - p0y)/(p1y - p0y)
		
	if (p3x - p2x) != 0:
		ratio_x2 = (x_int - p2x)/(p3x - p2x)
		
	if (p3y- p2y) != 0:
		ratio_y2 = (y_int - p2y)/(p3y- p2y)
		
	
	if (((ratio_x >=  0.0 and ratio_x <= 1.0) or (ratio_y >= 0 and ratio_y <=1.0)) and ((ratio_x2 >=  0 and ratio_x2 <= 1.0) or (ratio_y2 >= 0 and ratio_y2 <=1.0))):
		return (x_int, y_int)

	
	return (-1,-1)



# any non game logic tasks are handled here
class Environment:
	
	def __init__(self, vel, screen_size, stage_num):
		self.stage_num = stage_num
		self.player_vel = vel

		self.screen_size = screen_size
		#self.player_start = (random.randint(0, screen_size[0]), screen_size[1] -50)
		self.player_start = (260, 450)
		self.goal_start = (screen_size[0]/2, 15)
		self.is_over = False
	
	def make(self):
		"""
		Instantiate a GameLogic object for this Env
		"""
		self.logic = GameLogic(self.player_vel, self.player_start, self.screen_size, self.stage_num)
		print("------NEW GAME-------")
		print(f'Player Starting Position: {self.player_start}')


	def show(self):
		"""
		show current frame of to screen
		"""
		pg.init()
		window = pg.display.set_mode((self.logic.SCREEN_X, self.logic.SCREEN_Y)) # change
		pg.display.set_caption("Q LEARNING TEST")

		window.fill( (self.logic.BACKGROUND_COLOR))
		pg.draw.rect(window, self.logic.GOAL_COLOR, (self.logic.goal.x, self.logic.goal.y, self.logic.GOAL_SIZE, self.logic.GOAL_SIZE))
		pg.draw.rect(window, self.logic.PLAYER_COLOR, (self.logic.player.x, self.logic.player.y, self.logic.PLAYER_SIZE, self.logic.PLAYER_SIZE))
		
		# testing purposes
		player_top_left = (self.logic.player.x, self.logic.player.y)
		player_top_right = (self.logic.player.x_right, self.logic.player.y)
		player_bot_left = (self.logic.player.x, self.logic.player.y_bot)
		player_bot_right = (self.logic.player.x_right, self.logic.player.y_bot)
		
		# used for debugging, tbh it looks kinda cool if you want to uncomment lol
		"""
		pg.draw.line(window, self.logic.PLAYER_OUTLINE_COLOR, player_top_left, player_top_right) # correct
		pg.draw.line(window, self.logic.PLAYER_OUTLINE_COLOR, player_top_left, player_bot_left)
		pg.draw.line(window, self.logic.PLAYER_OUTLINE_COLOR, player_bot_left, player_bot_right)
		pg.draw.line(window, self.logic.PLAYER_OUTLINE_COLOR, player_top_right, player_bot_right)
		"""
		for obstacle in self.logic.obstacles:
			pg.draw.line(window, self.logic.OBSTACLE_COLOR, (obstacle.x_start, obstacle.y_start), (obstacle.x_end, obstacle.y_end),3)
		
		for vis in self.logic.player.vision:
			vis_ctr = (vis.x_start, vis.y_start)
			vis_end = (vis.x_end, vis.y_end)
			pg.draw.line(window, self.logic.VISION_COLOR, vis_ctr, vis_end)
		
		
		# shows logic gates for respective stage.
		"""
		for gate in self.logic.gates:
			gate_start = (gate.x_start, gate.y_start)
			gate_end = (gate.x_end, gate.y_end)
			if gate.is_active:
				pg.draw.line(window, self.logic.REWARD_GATE_COLOR,gate_start, gate_end)
		"""
		pg.display.update()
		# change this param based on how fast your computer's processing speed is. Game updates as fast as comp can handle.
		# training takes a long time, you'll thank me for just deciding to set to zero :)
		pg.time.delay(0) 
		


	def isOver(self):
		"""
		Self explainitory. Is the game over? Yes? Good. No? Also good.
		"""
		if self.logic.has_lost() or self.logic.has_won(): 
			return True
		else:
			return False

	def step(self, action):
		"""
		Handles a "step" -> given an action, change all internal logic to reflect change.
		"""
		pre_dist = self.logic.get_dist_to_goal()
		self.logic.update_player(action)
		post_dist = self.logic.get_dist_to_goal()
		self.logic.player_intersects()
		self.logic.vision_intersects()
		self.logic.hit_reward()

		#reward = self.calculate_reward()  # <- reward gate solution just straight up doesnt work.
		reward = self.dist_reward(pre_dist, post_dist)


		return self.logic.get_game_state(), reward, self.isOver(), self.logic.has_won()

			

	def dist_reward(self, pre_dist, post_dist):
		"""
		calculates reward based on how much closer agent is, or if they have won/lost or passed a gate.
		"""

		if self.logic.has_lost():
			print("---Player Lost---")
			return - 100.0
		elif self.logic.has_won():
			print('---Player Won---')
			return 1000
		if self.logic.hitting_gate:
			self.logic.hitting_gate = False
			print('---Gate Passed---')
			return 100

		return (pre_dist - post_dist)/self.player_vel # attempt 2 at reward function lol


	def reset(self):
		"""
		create a new game instance
		"""
		self.__init__(self.player_vel,self.screen_size, self.stage_num)
		self.make()
		# must return game state as well
		return self.logic.get_game_state()
	
	def calculate_reward(self):
		"""
		reward system based on how long player is taking to reach goal
		"""
		if self.logic.has_lost():
			print('---Loss---')
			return -100.0
			
		elif self.logic.has_won():
			print('---Game Won---')
			return 1000

		elif self.logic.hitting_gate:
			self.logic.hitting_gate = False
			print('---Gate Reached---')
			return 100

		return -1




# --------GAME LOGIC ------

class GameLogic: 
	

	def __init__(self, player_vel, player_loc, screen, stage):
		self.hitting_gate = False
		self.stage_num = stage
		self.is_over = False
		self.begin = False
		self.SCREEN_X = screen[0]
		self.SCREEN_Y = screen[1]
		self.PLAYER_VEL = player_vel
		
		self.BACKGROUND_COLOR = (74,78,77)  
		self.PLAYER_COLOR = (136,216,176) #          
		self.PLAYER_OUTLINE_COLOR = (150,206,180)
		self.GOAL_COLOR = (255,204,92)
		self.OBSTACLE_COLOR = (255,111,105)
		self.VISION_COLOR = (150,206,180)
		self.REWARD_GATE_COLOR = (0, 255, 0)

		self.PLAYER_SIZE = 20
		self.GOAL_SIZE = 70

		self.player = Player(player_loc[0], player_loc[1], self.PLAYER_SIZE, self.PLAYER_COLOR)
		self.set_stage(self.stage_num)
		self.vision_intersects()
		self.action_size = 4
		

	def set_stage(self, stage_num):
		
		"""
		Builds stages for game
		"""
		if stage_num == 0:
			self.obstacles = [Obstacle(100,0 ,100, 500, self.OBSTACLE_COLOR),
						  Obstacle(320,0,320, 500, self.OBSTACLE_COLOR)]
			self.gates = [RewardGate(0, 425,500, 425, self.REWARD_GATE_COLOR),
						  RewardGate(0, 350,500, 350, self.REWARD_GATE_COLOR),
						  RewardGate(0, 275,500, 275, self.REWARD_GATE_COLOR),
						  RewardGate(0, 200,500, 200, self.REWARD_GATE_COLOR),
						  RewardGate(0, 125,500, 125, self.REWARD_GATE_COLOR)]
			
			self.goal = Goal(250, 15, self.GOAL_SIZE, self.GOAL_COLOR)

			
			
		
		if stage_num == 1:
			self.obstacles = [Obstacle(200, 500 ,200, 300, self.OBSTACLE_COLOR),
							  Obstacle(200, 300 ,375, 300 , self.OBSTACLE_COLOR),
							  Obstacle(375,300 ,375, 0, self.OBSTACLE_COLOR),
							  Obstacle(300, 500 ,300, 400, self.OBSTACLE_COLOR),
							  Obstacle(300,400 ,500, 400, self.OBSTACLE_COLOR)]
			
			self.gates = [RewardGate(200, 425,300, 425, self.REWARD_GATE_COLOR),
						  RewardGate(200,300, 300, 400, self.REWARD_GATE_COLOR),
						  RewardGate(350,300, 350, 400, self.REWARD_GATE_COLOR),
						  RewardGate(375,300, 500, 400, self.REWARD_GATE_COLOR),
						  RewardGate(375,250, 500, 250, self.REWARD_GATE_COLOR)]

			self.goal = Goal(400, 100, self.GOAL_SIZE, self.GOAL_COLOR)
		

	def update_player(self, action):
		"""
		updates player position based on input
		"""
		if action == 0:
			self.player.change_pos(0, -self.PLAYER_VEL)
		if action == 1:
			self.player.change_pos(self.PLAYER_VEL,0)
		if action == 2:
			self.player.change_pos(0, self.PLAYER_VEL)
		if action == 3:
			self.player.change_pos(-self.PLAYER_VEL, 0)
	


		

	def get_game_state(self):
		"""
		returns the current position of player and distance from wall(s)
		"""
		state = []
		state.append(self.player.x)
		state.append(self.player.y)
		for vis in self.player.vision:
			distance = np.sqrt((vis.x_start - vis.x_end)**2 + (vis.y_start - vis.y_end)**2)
			state.append(distance)


		self.state_size = len(state)
		return state


	def player_intersects(self):
		"""
		Checks if a player is intersecting a wall
		"""
		
		player_lines = [] 
		player_top_left = (self.player.x, self.player.y)
		player_top_right = (self.player.x_right, self.player.y)
		player_bot_left = (self.player.x, self.player.y_bot)
		player_bot_right = (self.player.x_right, self.player.y_bot)

		
		player_lines.append((player_top_left, player_bot_left))
		player_lines.append((player_bot_right, player_top_right))
		player_lines.append((player_bot_left, player_bot_right))
		player_lines.append((player_top_left, player_top_right))
		

		for obstacle in self.obstacles:
			
			o_line = ((obstacle.x_start, obstacle.y_start), (obstacle.x_end, obstacle.y_end))
			for p_line in player_lines:
				intersects = points_intersect(p_line, o_line)
				if intersects[0] != -1:
					# print('---Player Lost---')
					return True
		
		return False

	

	def hit_reward(self):
		"""
		checks if player is hitting a reward gate
		"""

		player_lines = [] 
		player_top_left = (self.player.x, self.player.y)
		player_top_right = (self.player.x_right, self.player.y)
		player_bot_left = (self.player.x, self.player.y_bot)
		player_bot_right = (self.player.x_right, self.player.y_bot)
		
		player_lines.append((player_top_left, player_bot_left))
		player_lines.append((player_bot_right, player_top_right))
		player_lines.append((player_bot_left, player_bot_right))
		player_lines.append((player_top_left, player_top_right))

		for gate in self.gates:
			if gate.is_active:
				g_line = ((gate.x_start, gate.y_start), (gate.x_end, gate.y_end))
				for p_line in player_lines:
					intersects = points_intersect(p_line, g_line)
					if intersects[0] != -1:
						self.hitting_gate = True
						gate.is_active = False
						
	def vision_intersects(self):
		"""
		set the start and end locations of vision objects
		"""

		for vis in self.player.vision:
			vis_start = (vis.x_start, vis.y_start)
			vis_end = (vis.x_end, vis.y_end)
			
			intersects = []
			distances = []
			
						
			for obstacle in self.obstacles:
				
				o_line = ((obstacle.x_start, obstacle.y_start), (obstacle.x_end, obstacle.y_end))
				intersects_at = points_intersect((vis_start, vis_end),o_line)
				intersects.append(intersects_at)
				
				distance = np.sqrt((vis.x_start - intersects_at[0])**2 + (vis.y_start - intersects_at[1])**2)
				
				if intersects_at[0] > 0:
					distances.append(distance)

				else:
					distances.append(200) # larger than any possible line

			# make sure vision intersect is at the closest object.
			lowest_intersect_idx = distances.index(min(distances))
			lowest_distance = distances[lowest_intersect_idx]
			

			if intersects[lowest_intersect_idx][0] > 0: # if it does intersect somewhere
				if lowest_distance <= 100:
					
					
					vis.x_end = intersects[lowest_intersect_idx][0]
					vis.y_end = intersects[lowest_intersect_idx][1]
				
				
				





	def has_lost(self):
		"""
		Checks a win
		"""

		if self.player.x < 0 or (self.player.x + self.PLAYER_SIZE) > self.SCREEN_X or self.player.y < 0 or self.player.y + self.PLAYER_SIZE > self.SCREEN_Y:
			
			return True

		if self.player_intersects():
			
			return True 
		

		return False


	def has_won(self):
		"""
		checks a loss...
		"""
		#  ...
		if self.player.x > self.goal.x and (self.player.x + self.PLAYER_SIZE) < (self.goal.x + self.goal.size) and self.player.y > self.goal.y and (self.player.y + self.player.size) < (self.goal.y + self.goal.size):
			# print('WIN')
			return True
		else:
			return False


	def play_sound_effect(win):
		# decided to not have a huge file with sound effects and whatnot
		pass


	def get_dist_to_goal(self):
		# euclidean
		p_ctr_x = self.player.x + (self.PLAYER_SIZE/2)
		p_ctr_y = self.player.y + (self.PLAYER_SIZE/2)
		g_ctr_x = self.goal.x + (self.GOAL_SIZE/2)
		g_ctr_y = self.goal.y + (self.GOAL_SIZE/2)

		return np.sqrt((p_ctr_x - g_ctr_x)**2 + (p_ctr_y - g_ctr_y)**2)

# --------- MODEL ----------

class Player:

	def __init__(self, x_pos, y_pos, size, color):
		# keep in mind -> pygame draws rectangles from the top left corner, and circles from the center

		# our player is a square
		self.VISION_SIZE = 150
		self.x = x_pos  # box's furthest left x coord
		self.y = y_pos  # box's highest point
		self.size = size # our player is a square so just need 1 param.
		self.color = color
		self.reward = 0
		self.x_right = x_pos + size
		self.y_bot = y_pos + size 

		self.center_x = self.x + self.size/2
		self.center_y = self.y + self.size/2

		self.vision = [PlayerVis(self.center_x, self.center_y, 100,0),
					   PlayerVis(self.center_x, self.center_y, 100,45),
					   PlayerVis(self.center_x, self.center_y, 100,90),
					   PlayerVis(self.center_x, self.center_y, 100,135),
					   PlayerVis(self.center_x, self.center_y, 100,180),
					   PlayerVis(self.center_x, self.center_y, 100,225),
					   PlayerVis(self.center_x, self.center_y, 100,270),
					   PlayerVis(self.center_x, self.center_y, 100,315),
					   ]




	def change_pos(self, x, y):
		self.x = x + self.x
		self.x_right = x + self.x_right
		self.y = y + self.y
		self.y_bot = y + self.y_bot
		self.center_x = self.center_x + x
		self.center_y = self.center_y + y
		self.update_vision(self.center_x, self.center_y)

	def update_vision(self, x_start, y_start):
		for vis in self.vision:
			vis.update(x_start, y_start)



class Goal:
	# goal is a square
	def __init__(self, x_pos, y_pos, size, color):
		self.x = x_pos
		self.y = y_pos
		self.size = size
		self.color = color



class Obstacle:
	def __init__(self, x_start, y_start, x_end, y_end, color):
		# obstacles changed to lines for simplicity
		self.x_start = x_start
		self.y_start = y_start
		self.x_end = x_end
		self.y_end = y_end

		#self.width = width
		#self.height = height
		self.color = color

class RewardGate:
	def __init__(self, x_start, y_start, x_end, y_end, color):
		self.x_start = x_start
		self.y_start = y_start
		self.x_end = x_end
		self.y_end = y_end
		self.is_active = True

		self.color = color



class PlayerVis:
	
	def __init__(self, x_start,y_start, length,angle):
		self.angle = angle
		self.length = length
		self.x_start = x_start
		self.y_start = y_start
		# NOTE! math.cos and sin work in radians
		self.x_end = length * cos(radians(angle)) + x_start
		self.y_end = length * sin(radians(angle)) + y_start
		self.is_intersecting = False

	def update(self, x_start, y_start):
		self.x_start = x_start
		self.y_start = y_start
		self.x_end = self.length * cos(radians(self.angle)) + x_start
		self.y_end = self.length * sin(radians(self.angle)) + y_start

