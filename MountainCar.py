"""
Taken from: https://gist.github.com/EderSantana/c7222daa328f0e885093

MIT License

Copyright (c) 2016 Eder Santana (https://github.com/EderSantana)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import numpy as  np
import math


class MountainCar(object):
	def __init__(self, start, goal, Xrange, Vrange):
		self.start = start
		self.goal = goal
		self.Xrange = Xrange
		self.Vrange = Vrange
		self.num_actions = 3

	def _DoAction(self, action):
		# MountainCarDoAction: executes the action (a) into the mountain car
		# a: is the force to be applied to the car
		# x: is the vector containning the position and speed of the car
		# xp: is the vector containing the new position and velocity of the car

		position = self.state[0]
		speed = self.state[1]

		# bounds for position
		bpleft = self.Xrange[0] 

		# bounds for speed
		bsleft = self.Vrange[0]
		bsright = self.Vrange[1]
		speedt1= speed + (0.001 * (action - 1)) + (-0.0025 * math.cos(3.0 * position))	 
		speedt1= speedt1 * 0.999 # thermodynamic law, for a more real system with friction.

		if speedt1 < bsleft: 
			speedt1 = bsleft 
		elif speedt1 > bsright:
			speedt1 = bsright    

		post1 = position + speedt1 

		if post1 <= bpleft:
			post1 = bpleft
			speedt1 = 0.0

		xp = np.array([post1, speedt1])
		self.state = xp

	def _GetReward(self):
		# MountainCarGetReward returns the reward at the current state
		# x: a vector of position and velocity of the car
		# r: the returned reward.
		# f: true if the car reached the goal, otherwise f is false

		position = self.state[0]
		# bound for position; the goal is to reach position = 0.45
		bpright = self.goal

		r = -1
		f = False

		if  position >= bpright:
			r = 100
			f = True

		return r, f

	def act(self, action):
		self._DoAction(action)
		reward, game_over = self._GetReward()

		return self.observe(), reward, game_over

	def observe(self):
		return self.state.reshape((1, -1))

	def reset(self):
		self.state = np.array([self.start[0], self.start[1]])
