import csv
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import os, random, shutil
# matplotlib.use('TkAgg')
base_path = []
base_path.append(r'D:\UIUC\Tag\DATA\nsdi_DATA_class_20_10_50\1/')
base_path.append(r'D:\UIUC\Tag\DATA\nsdi_DATA_class_20_10_50\2/')
base_path.append(r'D:\UIUC\Tag\DATA\nsdi_DATA_class_20_10_50\3/')
base_path.append(r'D:\UIUC\Tag\DATA\nsdi_DATA_class_20_10_50\4/')
base_path.append(r'D:\UIUC\Tag\DATA\nsdi_DATA_class_20_10_50\5/')
base_path0 = r'D:\UIUC\Tag\DATA\nsdi_DATA_class_20_10_50\1/'
base_path1 = r'D:\UIUC\Tag\DATA\nsdi_DATA_class_20_10_50\2/'
base_path2 = r'D:\UIUC\Tag\DATA\nsdi_DATA_class_20_10_50\3/'
base_path3 = r'D:\UIUC\Tag\DATA\nsdi_DATA_class_20_10_50\4/'
base_path4 = r'D:\UIUC\Tag\DATA\nsdi_DATA_class_20_10_50\5/'
save_path = '../../DATA_pics/nsdi_DATA_20_10_50/'

num_trace = np.zeros([5, 1])
num_trace[0] = len([name for name in os.listdir(base_path0) if os.path.isfile(os.path.join(base_path0, name))])
num_trace[1] = len([name for name in os.listdir(base_path1) if os.path.isfile(os.path.join(base_path1, name))])
num_trace[2] = len([name for name in os.listdir(base_path2) if os.path.isfile(os.path.join(base_path2, name))])
num_trace[3] = len([name for name in os.listdir(base_path3) if os.path.isfile(os.path.join(base_path3, name))])
num_trace[4] = len([name for name in os.listdir(base_path4) if os.path.isfile(os.path.join(base_path4, name))])
plt.style.use('dark_background')

fig = plt.figure(figsize=(17, 17))
ax = plt.gca()
for c in range(5):
    sample = random.sample(os.listdir(base_path[c]), 5)
    for i in range(5):
        trace = scipy.io.loadmat(base_path[c]+sample[i])
        trace = np.array(trace['M'])
        ax = plt.subplot(5, 5, 5*i+c+1)
        plt.plot(trace[:, 0], trace[:, 1], linewidth=5, color='white')
        value = max(trace[:, 0].max()-trace[:, 0].min(), trace[:, 1].max()-trace[:, 1].min())
        if i ==4:
            ax.set_title('Class {}'.format(c), y=-0.35, fontsize=25)
plt.suptitle('Generated Trajectory Datasets', y=0.95, fontsize=32)
plt.savefig('Data_class.jpg')
plt.close()