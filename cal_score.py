import sys
import pickle

f = open('SeaBeamRider_v2_MaxKL.txt','r')
line = f.readline()
text = []
m=0
n=0
frame = []
reward = []
while line:
    text.append(line)
    line = f.readline()

for i in xrange(len(text)):
    # print 'line', i, ':', text[i]
    if i == 60 + m and i != 18500:
        print 'frames:',i,':',text[i][15:25]
        a = float(text[i][15:25])
        m+=20
        frame.append(a)
    if i == 61 + n and i != 18501:
        print 'reward:',i,':',text[i][33:41]
        b = float(text[i][33:41])
        n+=20
        reward.append(b)
#
# frame.append(6000000)
# reward.append(3019.0 )
print len(frame)
print len(reward)

fw = open('SeaBeamRider_v2_MaxKL.pkl', 'wb')
pickle.dump([frame, reward], fw, -1)
fw.close()

f.close()