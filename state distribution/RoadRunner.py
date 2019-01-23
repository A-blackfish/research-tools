import matplotlib.pyplot as plt
from math import log
import tensorflow as tf

def Cal_KL(sta_a,sta_b):
    a= 0
    b=0
    s_a = [0.0]*10
    s_b = [0.0] * 10

    for i in xrange(len(sta_a)-1):
        a+=sta_a[i]
    for i in xrange(len(sta_b)-1):
        b+=sta_b[i]

    for i in xrange(len(sta_a)-1):
        s_a[i] = sta_a[i]/a
    for i in xrange(len(sta_b)-1):
        s_b[i] = sta_b[i]/b
    # print "s_a:",sum(s_a),s_a
    # print "s_b:",sum(s_b),s_b

    with tf.Session() as sess:
        y_input = tf.placeholder("float", [10])
        x_input = tf.placeholder("float", [10])
        y = tf.nn.softmax(logits=y_input)
        x = tf.nn.softmax(logits=x_input)
        cost = tf.reduce_sum(y_input * tf.log(tf.div(y_input, x_input)))
        KL = sess.run(cost,{y_input:s_b,
                            x_input:s_a})

    return KL

def main():
    Krull = [10025.0, 247.0, 1157.0, 777.0, 1420.0, 1318.0, 1172.0, 1648.0, 17506.0, 14729.0, 1.0]
    Krull_h = [3648.0, 3762.0, 586.0, 884.0, 750.0, 395.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
               90.0, 156.0, 81.0, 56.0, 74.0, 79.0, 113.0, 400.0, 213.0, 81.0, 32.0, 28.0, 40.0, 47.0, 87.0, 155.0,
               104.0, 82.0, 64.0, 69.0, 70.0, 59.0, 66.0, 65.0, 71.0, 87.0, 217.0, 233.0, 106.0, 112.0, 224.0, 239.0,
               124.0, 138.0, 144.0, 154.0, 144.0, 129.0, 144.0, 155.0, 88.0, 98.0, 94.0, 90.0, 83.0, 94.0, 112.0, 138.0,
               159.0, 151.0, 141.0, 110.0, 124.0, 116.0, 134.0, 128.0, 131.0, 139.0, 165.0, 141.0, 222.0, 348.0, 592.0,
               693.0, 877.0, 1131.0, 1204.0, 1505.0, 1904.0, 2529.0, 3266.0, 3805.0, 4189.0, 4057.0, 3350.0, 2001.0,
               745.0, 280.0, 72.0, 26.0, 9.0, 0.0, 1.0]
    Seaquest = [48.0, 1222.0, 1528.0, 2224.0, 3122.0, 2816.0, 5351.0, 14411.0, 16072.0, 3205.0, 1.0]
    RoadRunner = [4327.0, 220.0, 5.0, 894.0, 1470.0, 11276.0, 18014.0, 10448.0, 2972.0, 373.0, 1.0]
    StarGunner = [16.0, 135.0, 4653.0, 2032.0, 10628.0, 14792.0, 9255.0, 6305.0, 2025.0, 158.0, 1.0]
    VideoPinball = [2426.0, 4116.0, 1016.0, 379.0, 891.0, 2653.0, 9048.0, 21968.0, 6917.0, 585.0, 1.0]
    Gopher = [17.0, 74.0, 1973.0, 2428.0, 7769.0, 15759.0, 13227.0, 7027.0, 1531.0, 194.0, 1.0]

    name_list = [101] * 101
    for i in xrange(100):
        name_list[i] = i
    print name_list
    plt.bar(range(len(Krull_h)), Krull_h)
    plt.xlabel('Anyway-v0')
    plt.show()
    # print "krull v5-v4",Cal_KL(StarGunner,Gopher)
    # Cal_KL(StarGunner,Krull) 3.12427
    # Cal_KL(StarGunner,Seaquest)  0.893408
    # Cal_KL(StarGunner,RoadRunner)  0.726751
    # Cal_KL(StarGunner,StarGunner)  0.0
    # Cal_KL(StarGunner, VideoPinball)  1.07557
    # Cal_KL(StarGunner, Gopher)  0.0471299

if __name__ == '__main__':
    main()