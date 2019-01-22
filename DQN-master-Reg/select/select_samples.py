from collections import deque
import pickle
import random

TD = []
outdir = '/home/server4/PycharmProjects/dqn-master-Reg/'

def find_mid(mid,length):  #find the num of number that bigger than mid
    i = 0
    j = 0
    while i < length:
        if TD[i] >= mid:
            j += 1
        i += 1
    return j

def quick_sort(aList):
    first = 0
    # define the first position of list
    last = len(aList) - 1
    # quick sort function
    aList = quick_sort_helper(aList, first, last)
    return aList

def quick_sort_helper(aList, first, last):
    if first >= last:
        return aList
    pivot = aList[first]
    count = first
    for index in range(first + 1, last + 1):
        if aList[index] < pivot:
            count += 1
            aList[index], aList[count] = aList[count], aList[index]
    aList[count], aList[first] = aList[first], aList[count]
    quick_sort_helper(aList, first, count - 1)
    quick_sort_helper(aList, count + 1, last)
    return aList

def read_datas(file_name):     # read the pkl into buffer deque()
    replay_buffer = deque()
    i = 1
    while i == 1:
        try:
            replay_buffer.append(pickle.load(file_name))
        except:
            i = 0
            file_name.close()
    return  replay_buffer

def main():
    Eva = outdir + 'Assault-v0.pkl'
    f1 = open(Eva)
    replay_buffer1=read_datas(f1)

    while len(replay_buffer1)>0:  # append the td_erron into TD list
        temp = replay_buffer1.popleft()
        TD.append(temp[6])

    td_sorted=quick_sort(TD)    # do quick sort of TD list
    mid = len(TD)/2         # in a sorted list , the mid is the index of middle number
    # test_mid=find_mid(td_sorted[mid],len(TD))    # varify the middle number is correct or not
    # print mid, test_mid

    f2 = open(Eva)
    replay_buffer2=read_datas(f2)
    replay = deque()
    print 'the length of replay memery is:',len(replay_buffer2)
    #mid = td_sorted[mid]

    File_small = file(outdir + 'Assault_small.pkl','wb')
    # File_rand = file('0.2_noexplore4_rand.pkl','wb')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # selecting the half smaller of the replay
    while len(replay_buffer2)>0:
        temp = replay_buffer2.popleft()
        if temp[6]<td_sorted[mid]:
            replay.append(temp)
            pickle.dump(temp, File_small, True)

    print 'length of half replay is:',len(replay)
if __name__ == '__main__':
    main()
