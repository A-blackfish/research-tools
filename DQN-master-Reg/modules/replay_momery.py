"""
Giacomo Spigler
"""
import sys
import os
import numpy as np
import pickle
import random
from collections import deque

path = os.getcwd()
sys.path.append(path)
from regression.gym_dqn_test import  collection,ENV_NAME,Pkl
if collection:
	replay_package = file(Pkl+'.pkl', 'wb')

class ReplayMemoryFast:
	""" Simple queue for storing and sampling of minibatches.
		This implementation has been optimized for speed by pre-allocating the buffer memory and 
		by allowing the same element to be non-unique in the sampled minibatch. In practice, however, this 
		will never happen (minibatch size ~32-128 out of 10-100-1000k memory size).
	"""
	def __init__(self, memory_size, minibatch_size):
		self.memory_size = memory_size # max number of samples to store
		self.minibatch_size = minibatch_size

		self.experience = [None]*self.memory_size  #deque(maxlen = self.memory_size)
		# self.experience2 = [None] * self.memory_size  # deque(maxlen = self.memory_size)
		# self.experience3 = [None] * self.memory_size  # deque(maxlen = self.memory_size)
		# self.experience4 = [None] * self.memory_size  # deque(maxlen = self.memory_size)
		# self.experience5 = [None] * self.memory_size  # deque(maxlen = self.memory_size)
		# self.experience6 = [None] * self.memory_size  # deque(maxlen = self.memory_size)
		# self.experience7 = [None] * self.memory_size  # deque(maxlen = self.memory_size)
		# self.experience8 = [None] * self.memory_size  # deque(maxlen = self.memory_size)

		self.current_index = 1
		self.size = 0

	def store(self, observation, action, reward, newobservation, is_terminal, q_values):
		self.experience[self.current_index] = (observation, action, reward, newobservation, is_terminal, q_values)
		self.current_index += 1
		self.size = min(self.size+1, self.memory_size)
		if self.current_index >= self.memory_size:
			self.current_index -= self.memory_size

	# store the teacher's samples into a file, Duan
	def store_file(self,observation, action, reward, newobservation, is_terminal, q_values, TD):
		if self.current_index <= self.memory_size:
			if self.current_index % 5000 == 0:
				print 'current_index:', self.current_index
			pickle.dump([observation, action, reward, newobservation, is_terminal, q_values, TD], replay_package, True)
		self.current_index += 1

	# push the teacher's samples into replay memory, Duan
	def restore(self):
		# self.experience = [None] * 10000  # deque(maxlen = self.memory_size)
		f1 = open(Pkl +'.pkl')
		replay_buffer1 = deque()
		i = 0
		while i <= self.memory_size:
			try:
				replay_buffer1.append(pickle.load(f1))
				minibatch = replay_buffer1.popleft()

				# state_value = minibatch[5]
				# state_value = state_value.reshape(np.size(state_value))
				# state_value = np.hstack([state_value, np.zeros([18 - np.size(state_value)])])
				# state_value = state_value.reshape(1, 18)
				# minibatch[5] = state_value

				self.experience[i] = minibatch
				i = i + 1
			except:
				i = self.memory_size+2
				f1.close()
		print 'length of self.experience:',len(self.experience)  #self.experience is ok

		#---------------------------------------------------------------------------------
		# f2 = open('BeamRider-v0' + '.pkl')
		# replay_buffer2 = deque()
		# i = 0
		# while i <= self.memory_size:
		# 	try:
		# 		replay_buffer2.append(pickle.load(f2))
		# 		minibatch2 = replay_buffer2.popleft()
        #
		# 		# state_value = minibatch2[5]
		# 		# state_value = state_value.reshape(np.size(state_value))
		# 		# state_value = np.hstack([state_value, np.zeros([18 - np.size(state_value)])])
		# 		# state_value = state_value.reshape(1, 18)
		# 		# minibatch2[5] = state_value
        #
		# 		self.experience2[i] = minibatch2
		# 		i = i + 1
		# 	except:
		# 		i = self.memory_size + 2
		# 		f2.close()
		# print 'length of self.experience2:', len(self.experience2)  # self.experience is ok
        #
        #
		# f3 = open('CrazyClimber-v0' + '.pkl')
		# replay_buffer3 = deque()
		# i = 0
		# while i <= self.memory_size:
		# 	try:
		# 		replay_buffer3.append(pickle.load(f3))
		# 		minibatch3 = replay_buffer3.popleft()
        #
		# 		# state_value = minibatch3[5]
		# 		# state_value = state_value.reshape(np.size(state_value))
		# 		# state_value = np.hstack([state_value, np.zeros([18 - np.size(state_value)])])
		# 		# state_value = state_value.reshape(1, 18)
		# 		# minibatch3[5] = state_value
        #
		# 		self.experience3[i] = minibatch3
		# 		i = i + 1
		# 	except:
		# 		i = self.memory_size + 2
		# 		f3.close()
		# print 'length of self.experience3:', len(self.experience3)  # self.experience is ok
		# #------------------------------------------------------------------------------------
        #
		# f4 = open('Gopher-v0'+'.pkl')
		# replay_buffer4 = deque()
		# i = 0
		# while i <= self.memory_size:
		# 	try:
		# 		replay_buffer4.append(pickle.load(f4))
		# 		minibatch4 = replay_buffer4.popleft()
        #
		# 		# state_value = minibatch4[5]
		# 		# state_value = state_value.reshape(np.size(state_value))
		# 		# state_value = np.hstack([state_value, np.zeros([18 - np.size(state_value)])])
		# 		# state_value = state_value.reshape(1, 18)
		# 		# minibatch4[5] = state_value
        #
		# 		self.experience4[i] = minibatch4
		# 		i = i + 1
		# 	except:
		# 		i = self.memory_size+2
		# 		f4.close()
		# print 'length of self.experience:',len(self.experience4)  #self.experience is ok
        #
		# #---------------------------------------------------------------------------------
		# f5 = open('Krull-v0' + '.pkl')
		# replay_buffer5 = deque()
		# i = 0
		# while i <= self.memory_size:
		# 	try:
		# 		replay_buffer5.append(pickle.load(f5))
		# 		minibatch5 = replay_buffer5.popleft()
        #
		# 		# state_value = minibatch5[5]
		# 		# state_value = state_value.reshape(np.size(state_value))
		# 		# state_value = np.hstack([state_value, np.zeros([18 - np.size(state_value)])])
		# 		# state_value = state_value.reshape(1, 18)
		# 		# minibatch5[5] = state_value
        #
		# 		self.experience5[i] = minibatch5
		# 		i = i + 1
		# 	except:
		# 		i = self.memory_size + 2
		# 		f5.close()
		# print 'length of self.experience2:', len(self.experience5)  # self.experience is ok
        #
        #
		# f6 = open('RoadRunner-v0' + '.pkl')
		# replay_buffer6 = deque()
		# i = 0
		# while i <= self.memory_size:
		# 	try:
		# 		replay_buffer6.append(pickle.load(f6))
		# 		minibatch6 = replay_buffer6.popleft()
        #
		# 		# state_value = minibatch6[5]
		# 		# state_value = state_value.reshape(np.size(state_value))
		# 		# state_value = np.hstack([state_value, np.zeros([18 - np.size(state_value)])])
		# 		# state_value = state_value.reshape(1, 18)
		# 		# minibatch6[5] = state_value
        #
		# 		self.experience6[i] = minibatch6
		# 		i = i + 1
		# 	except:
		# 		i = self.memory_size + 2
		# 		f6.close()
		# print 'length of self.experience3:', len(self.experience6)  # self.experience is ok
        #
        #
		# #---------------------------------------------------------------------------------
		# f7 = open('Robotank-v0' + '.pkl')
		# replay_buffer7 = deque()
		# i = 0
		# while i <= self.memory_size:
		# 	try:
		# 		replay_buffer7.append(pickle.load(f7))
		# 		minibatch7 = replay_buffer7.popleft()
        #
		# 		# state_value = minibatch7[5]
		# 		# state_value = state_value.reshape(np.size(state_value))
		# 		# state_value = np.hstack([state_value, np.zeros([18 - np.size(state_value)])])
		# 		# state_value = state_value.reshape(1, 18)
		# 		# minibatch7[5] = state_value
        #
		# 		self.experience7[i] = minibatch7
		# 		i = i + 1
		# 	except:
		# 		i = self.memory_size + 2
		# 		f7.close()
		# print 'length of self.experience2:', len(self.experience7)  # self.experience is ok
        #
        #
		# f8 = open('VideoPinball-v0' + '.pkl')
		# replay_buffer8 = deque()
		# i = 0
		# while i <= self.memory_size:
		# 	try:
		# 		replay_buffer8.append(pickle.load(f8))
		# 		minibatch8 = replay_buffer8.popleft()
        #
		# 		# state_value = minibatch8[5]
		# 		# state_value = state_value.reshape(np.size(state_value))
		# 		# state_value = np.hstack([state_value, np.zeros([18 - np.size(state_value)])])
		# 		# state_value = state_value.reshape(1, 18)
		# 		# minibatch8[5] = state_value
        #
		# 		self.experience8[i] = minibatch8
		# 		i = i + 1
		# 	except:
		# 		i = self.memory_size + 2
		# 		f8.close()
		# print 'length of self.experience3:', len(self.experience8)  # self.experience is ok
		# #-------------------------------------------------------------------------------------

	# sample from teacher's replay memory, regression methods, Duan
	def sample_file(self,num):
		samples_index  = np.floor(np.random.random((self.minibatch_size,))*self.memory_size)
		samples = [self.experience[int(i)] for i in samples_index]
		return  samples
		# if num % 8 == 0:
		# 	samples	= [self.experience[int(i)] for i in samples_index]
		# 	return samples
		# elif num % 8 == 1:
		# 	samples = [self.experience2[int(i)] for i in samples_index]
		# 	return samples
		# elif num % 8 == 2:
		# 	samples = [self.experience3[int(i)] for i in samples_index]
		# 	return samples
		# elif num % 8 == 3:
		# 	samples = [self.experience4[int(i)] for i in samples_index]
		# 	return samples
		# elif num % 8 == 4:
		# 	samples = [self.experience5[int(i)] for i in samples_index]
		# 	return samples
		# elif num % 8 == 5:
		# 	samples = [self.experience6[int(i)] for i in samples_index]
		# 	return samples
		# elif num % 8 == 6:
		# 	samples = [self.experience7[int(i)] for i in samples_index]
		# 	return samples
		# elif num % 8 == 7:
		# 	samples = [self.experience8[int(i)] for i in samples_index]
		# 	return samples

	def sample(self):
		""" Samples a minibatch of minibatch_size size. """
		if self.size <  self.minibatch_size:
			return []
		samples_index  = np.floor(np.random.random((self.minibatch_size,))*self.size)
		samples		= [self.experience[int(i)] for i in samples_index]
		return samples



