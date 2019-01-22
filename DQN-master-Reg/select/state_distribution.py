import numpy as np
import pickle
import random
from collections import deque
import gym
import sys
import os

size = 25000
action_space = 18
length = 100

def main():
    replay_buffer1 = [None] * size  # deque(maxlen = self.memory_size)
    experience = [None] * size  # deque(maxlen = self.memory_size)
    f1 = open('Assault_small1.pkl')
    replay_buffer1 = deque()
    i = 0
    while i <= size:
        try:
            replay_buffer1.append(pickle.load(f1))
            minibatch = replay_buffer1.popleft()

            state_value = minibatch[5]
            state_value = state_value.reshape(np.size(state_value))

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            state_t=np.hstack([state_value,np.zeros([18-np.size(state_value)])])
            # print "state:",state_t
            s = state_t.reshape(1,18)
            # print 'state reshape:',s, 'shape of s:',s.shape
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            experience[i] = max(state_value[0:action_space])
            # print 'state values:',state_value,'experience:', experience[i]
            i= i + 1
        except:
            i = size+100
            f1.close()

    # print "length of exp:",len(experience)
    # print "type of exp:",type(experience)
    state = np.array(experience, dtype = float)
    # print "shape of exp:",state.shape
    print "Max of state:",max(state),"Min of state:",min(state)
    max_n = max(state)
    min_n = min(state)

    space = (max_n-min_n)/length
    bar = [0.0]*(length+1)
    for i in xrange(len(state)):
        x = (state[i]-min_n)/space
        x = int(x)
        # print x
        bar[x]+=1
    print sum(bar),bar


if __name__ == '__main__':
    main()