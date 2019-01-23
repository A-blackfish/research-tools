from bs4 import BeautifulSoup
from urllib import urlopen
import re
#file:///home/anny/PycharmProjects/dqn-2-master/gym_results/RoadRunner-v0/reg_score2.html
#file:///home/anny/PycharmProjects/drawline/python%20warm/test.html
html = urlopen("file:///home/anny/PycharmProjects/dqn-2-master/gym_results/RoadRunner-v0/reg_score2.html")\
    .read().decode('utf-8')
soup = BeautifulSoup(html, features='lxml')
# print soup.body  # print the body
# print 'over'
#------------------------------------------------------------------------
# img_link = soup.find_all('img',{'src': re.compile('.*?\.jpg')})
# for link in img_link:
#     print link['src']
#
# course_links = soup.find_all('a', {'href': re.compile('https://morvan.*')})
# for link in course_links:
#     print link['href']
#------------------------------------------------------------------------
test = soup.find_all(text = re.compile("Average Reward"))
for t in test:
    print t
