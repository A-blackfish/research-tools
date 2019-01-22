import numpy as np
import pickle
from collections import deque

size = 50000
Length = 25000
area = 100
out = '/home/server4/PycharmProjects/dqn-master-Reg/'
stu = 'Seaquest-v0.pkl'
tea = 'Assault-v0.pkl'

Seaquest = [1.0, 0.0, 0.0, 0.0, 2.0, 2.0, 1.0, 4.0, 3.0, 4.0, 3.0, 16.0, 10.0, 23.0, 33.0, 38.0, 78.0, 114.0, 126.0,
                107.0, 85.0, 79.0, 89.0, 78.0, 95.0, 95.0, 88.0, 81.0, 72.0, 80.0, 80.0, 61.0, 55.0, 99.0, 102.0, 117.0,
                122.0, 122.0, 158.0, 169.0, 147.0, 146.0, 211.0, 129.0, 131.0, 108.0, 140.0, 146.0, 161.0, 164.0, 147.0,
                119.0, 96.0, 94.0, 110.0, 116.0, 168.0, 153.0, 180.0, 228.0, 242.0, 234.0, 195.0, 171.0, 205.0, 217.0,
                215.0, 304.0, 369.0, 477.0, 558.0, 710.0, 775.0, 787.0, 695.0, 708.0, 677.0, 741.0, 840.0, 924.0, 901.0,
                994.0, 1074.0, 1035.0, 918.0, 785.0, 693.0, 616.0, 598.0, 459.0, 416.0, 335.0, 292.0, 205.0, 125.0,
                76.0, 33.0, 10.0, 3.0, 1.0, 1.0]

# Krull_v0 = [3648.0, 3762.0, 586.0, 884.0, 750.0, 395.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 90.0, 156.0, 81.0, 56.0, 74.0, 79.0, 113.0, 400.0, 213.0, 81.0, 32.0, 28.0, 40.0, 47.0, 87.0, 155.0, 104.0, 82.0, 64.0, 69.0, 70.0, 59.0, 66.0, 65.0, 71.0, 87.0, 217.0, 233.0, 106.0, 112.0, 224.0, 239.0, 124.0, 138.0, 144.0, 154.0, 144.0, 129.0, 144.0, 155.0, 88.0, 98.0, 94.0, 90.0, 83.0, 94.0, 112.0, 138.0, 159.0, 151.0, 141.0, 110.0, 124.0, 116.0, 134.0, 128.0, 131.0, 139.0, 165.0, 141.0, 222.0, 348.0, 592.0, 693.0, 877.0, 1131.0, 1204.0, 1505.0, 1904.0, 2529.0, 3266.0, 3805.0, 4189.0, 4057.0, 3350.0, 2001.0, 745.0, 280.0, 72.0, 26.0, 9.0, 0.0, 1.0]


def Sort_Index(Seaquest, Tea):
    # print sum(Seaquest)
    # print max(Seaquest),min(Seaquest)
    S = sorted(Seaquest)
    # print type(S)

    D = []

    for i in xrange(len(Seaquest)):
        # print S[i],Seaquest.index(S[i])
        Temp =  Seaquest.index(S[i])
        D.append(Temp)
        Seaquest[Temp] = -1
    # print D

    Sum = []
    T = []

    for i in xrange(len(Tea)):
        x = Tea[D[i]]
        Sum.append(x)
        T.append(D[i])
        # print Tea[D[i]], D[i]
        if sum(Sum) >= Length:
            break
    print len(T),T,sum(Sum)
    return T

def read_datas(file_name):     # read the pkl into buffer deque()
    f1 = open(file_name)
    replay_buffer = deque()
    i = 1
    while i <= size:
        try:
            replay_buffer.append(pickle.load(f1))
            i = i+1
        except:
            i = size + 100
            f1.close()
    return  replay_buffer

def Cal_bar(filename):
    experience = [None] * size  # deque(maxlen = self.memory_size)
    f1 = open(filename)
    replay_buffer1 = deque()
    i = 0
    while i <= size:
        try:
            replay_buffer1.append(pickle.load(f1))
            minibatch = replay_buffer1.popleft()

            state_value = minibatch[5]
            state_value = state_value.reshape(np.size(state_value))

            experience[i] = max(state_value)
            i = i + 1
        except:
            i = size + 100
            f1.close()

    state = np.array(experience, dtype=float)
    # print "Max of state:", max(state), "Min of state:", min(state)
    max_n = max(state)
    min_n = min(state)

    space = (max_n - min_n) / area
    bar = [0.0] * (area + 1)
    for i in xrange(len(state)):
        x = (state[i] - min_n) / space
        x = int(x)
        # print x
        bar[x] += 1
    print sum(bar),bar
    return bar

def Sta_dis(filename, Choose):
    experience = [None] * size  # deque(maxlen = self.memory_size)
    f1 = open(filename)
    replay_buffer1 = deque()
    i = 0
    while i <= size:
        try:
            replay_buffer1.append(pickle.load(f1))
            minibatch = replay_buffer1.popleft()

            state_value = minibatch[5]
            state_value = state_value.reshape(np.size(state_value))

            experience[i] = max(state_value)
            i = i + 1
        except:
            i = size + 100
            f1.close()

    state = np.array(experience, dtype=float)
    print "Max of state:", max(state), "Min of state:", min(state)
    max_n = max(state)
    min_n = min(state)

    space = (max_n - min_n) / area
#-------------------------------------------------------------
    replay = deque()
    File_small = file(out+'Assault_small1.pkl','wb')

    Exp_state = read_datas(filename)

    while len(Exp_state)>0:
        temp = Exp_state.popleft()

        state_value = temp[5]
        state_value = state_value.reshape(np.size(state_value))


        x = (max(state_value) - min_n) / space
        x = int(x)
        if x in Choose:
            replay.append(temp)
            pickle.dump(temp, File_small, True)
    print len(replay)

def main():
    teacher = Cal_bar(out+tea)
    #student = Cal_bar(out+stu)

    Stu = Sort_Index(Seaquest,teacher)
    Sta_dis(out+ tea, Stu)

if __name__ == '__main__':
    main()