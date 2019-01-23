import matplotlib.pyplot as plt

num_list = [1.5,0.6,7.8,6]
name_list = ['Monday','Tuesday','Friday','Sunday']
plt.bar(range(len(num_list)), num_list, color='rgb',tick_label=name_list)
plt.show()