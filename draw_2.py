import matplotlib.pyplot as plt
import pickle
import random
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
def main():
    fr = open('Cal_KrullSea_T_StarGunner.txt','rb')
    fr1 = open('Cal_random_starGunner.txt','rb')
    fr2 = open('Cal_RoadGoph_To_starG.txt','rb')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    data1 = pickle.load(fr)
    x = data1[0]
    y = data1[1]
    # print data1[0], data1[1]
    fr.close()

    data2 = pickle.load(fr1)
    x2 = data2[0]
    y2 = data2[1]
    print data2[0], data2[1]
    fr1.close()

    plt.grid(True)
    plt.xlabel('StarGunner Transfer')
    plt.ylabel('score')
    plt.ylim()
    plt.xlim(0,6000000)

    data3 = pickle.load(fr2)
    x3 = data3[0]
    y3 = data3[1]
    # print data1[0], data1[1]
    fr.close()

    # new_ticks = np.linspace(0, 32000,20)
    # plt.yticks(new_ticks)

    # plt.xlim()
    # plt.xticks([1000000, 2000000, 3000000, 4000000, 5000000, 6000000],
    #            [ '1.0', '2.0', '3.0', '4.0', '5.0', '6.0'])
    # new_ticks = np.linspace(0, 6000000, 6)
    # plt.xticks(new_ticks)

    #--------------------------------------------------------------------------
    sum = 0
    for i in xrange(len(data2[1])):
        sum = sum+data2[1][i]
    ave = sum/len(data2[1])
    print 'ave:',ave

    dsum = 0
    for i in xrange(len(data1[1])):
        dsum = dsum+data1[1][i]
    dave = dsum/len(data1[1])
    print 'dave:',dave
    #--------------------------------------------------------------------------

    plt.plot(x, y, 'red', linewidth=0.8, label='Trans1 dqn')
    plt.plot(x2, y2, 'blue', linewidth=0.8, label='Random dqn')
    plt.plot(x3, y3, 'green', linewidth=0.8, label='Trans2 dqn')
    # plt.plot([0,6000000],[ave,ave],'blue',linewidth=0.8, label = 'regression ave')
    # plt.plot([0, 6000000], [dave, dave], 'red', linewidth=0.8, label='dqn ave')

    plt.legend(loc='lower right')
    plt.show()
if __name__ == '__main__':
  main()
