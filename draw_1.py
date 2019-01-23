import matplotlib.pyplot as plt
import pickle
import random
import numpy as np
def main():
    fr = open('SeaBeamRider.pkl','rb')
    data1 = pickle.load(fr)
    x = data1[0]
    y = data1[1]
    print data1[1]
    fr.close()

    #
    fr1 = open('SeaBeamRider_v1_smallTD.pkl','rb')
    data1 = pickle.load(fr1)
    x1 = data1[0]
    y1 = data1[1]
    # print data1[2]
    fr1.close()
    #
    fr2 = open('SeaBeamRider_v2_MaxKL.pkl','rb')
    data2 = pickle.load(fr2)
    x2 = data2[0]
    y2 = data2[1]
    fr2.close()
    #
    fr3 = open('seaGopher.pkl','rb')
    data3 = pickle.load(fr3)
    x3 = data3[0]
    y3 = data3[1]
    #print data1[0], data1[1], data1[2], data1[3]
    fr3.close()
    #
    fr4 = open('SeaKrull.pkl','rb')
    data4 = pickle.load(fr4)
    x4 = data4[0]
    y4 = data4[1]
    #print data1[0], data1[1], data1[2], data1[3]
    fr4.close()

    fr5 = open('SeaRoad.pkl','rb')
    data5 = pickle.load(fr5)
    x5 = data5[0]
    y5 = data5[1]
    #print data1[0], data1[1], data1[2], data1[3]
    fr5.close()

    fr6 = open('SeaVideoPinball.pkl','rb')
    data6 = pickle.load(fr6)
    x6 = data6[0]
    y6 = data6[1]
    #print data1[0], data1[1], data1[2], data1[3]
    fr6.close()

    # fr7 = open('SeaRobotank.pkl','rb')
    # data7 = pickle.load(fr7)
    # x7 = data7[0]
    # y7 = data7[1]
    # #print data1[0], data1[1], data1[2], data1[3]
    # fr7.close()

    fr8 = open('SeaAssault.pkl','rb')
    data8 = pickle.load(fr8)
    x8 = data8[0]
    y8 = data8[1]
    #print data1[0], data1[1], data1[2], data1[3]
    fr8.close()
    # x_ave = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    # y_ave = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    # for i in xrange(0,26):
    #     x_ave[i] = (x[i]+x1[i]+x2[i]+x3[i]+x4[i])/5
    #     y_ave[i] = (y[i]+y1[i]+y2[i]+y3[i]+y4[i])/5
    # print x_ave
    # print y_ave

    # create a new figure and it will be plot at end
    plt.figure(1)
    #ini1

    # plt.subplot(211)
    plt.title('Tea: BeamRider Stu:Seaquest ')
    plt.xlabel('training times')
    plt.ylabel('score')
    # plt.ylim(0,205)
    # plt.xlim(0,100000)
    plt.plot(x,y, 'black', linewidth=1.0, label='No select')
    plt.plot(x1,y1,'blue', linewidth=1.0, label='Selected1')
    plt.plot(x2,y2,'red', linewidth=1.0, label='Selected2')
    # plt.plot(x3,y3,'green', linewidth=1.0, label='SeaGoppher')
    # plt.plot(x4,y4,'purple', linewidth=1.0, label='SeaKrull')
    # plt.plot(x5, y5, 'grey', linewidth=1.0, label='SeaRoadRunner')
    # plt.plot(x6, y6, 'cyan', linewidth=1.0, label='SeaVideoPinball')
    # plt.plot(x7, y7, 'orange', linewidth=1.0, label='SeaRobotank')
    # plt.plot(x8, y8, 'brown', linewidth=1.0, label='SeaAssault')
    plt.legend(loc='upper left')
    plt.show()
if __name__ == '__main__':
  main()
