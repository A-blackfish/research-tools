from collections import deque
import numpy as np

# que = deque()
array = []
size = 100

marker = []
score = [0]*size

# que = [i for i in np.random.randint(1000, size=size)]
array = [i for i in np.random.randint(1000, size=size)]

# print('que:', que, 'type of que:', type(que))
print('array:', array, 'type of array:', type(array))
print('score:', score,'length of score:', len(score))

for j in range(10):
    for i in range(100):
        minibatch = [i for i in np.random.randint(size, size=4)]
        # print('minibatch:', minibatch)
        marker.extend(minibatch)

    for m in marker:
        score[m] += 1
    marker = []

print('score After:', score)

dex = []
for i in range(size):
    if score[i]>40:
        dex.append(i)
print('the index of larger than 30:', dex)
# print('marker:', marker)
# marker = list(set(marker))
# print('marker:', marker)

# for i in minibatch:
#     print('que[%d]:%d array[%d]:%d'%(i, que[i], i, array[i]))
