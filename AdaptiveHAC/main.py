import matlab.engine
import numpy as np
import os 
from AdaptiveHAC.pointTransformer import train_cls

def save_PC(dir, PC, labels):
    activities = ['na','wlk','stat','sitdn','stupsit','bfrsit','bfrstand','ffw','stup','ffs']
    for act in activities:
        os.mkdir(f'{dir}/{act}')
    for sample, label, i in zip(PC, labels, range(len(PC))):
        sample_act = activities[int(np.mean(label))]
        with open(f'{dir}/{sample_act}/{sample_act}_{i}.txt', "a") as file:
            for node in sample[0]:
                for point in node:
                    file.write(f'{point[0]} {point[1]} {point[2]} {point[3]} {point[4]}')
    return

os.environ['HYDRA_FULL_ERROR'] = '1'
eng = matlab.engine.start_matlab()
eng.addpath('processing')
eng.addpath('segmentation')

# paramenters
window = 256
chunks = 6
npoints = 1024
thr = 0.8

#path = 'W:/staff-groups/ewi/me/MS3/MS3-Shared/Ronny_MonostaticData/Nicolas/MAT_data_aligned/'
path = './test/data/'
file_name = '002_mon_wal_Nic'

# load data file with the matlab engine and unpack data
mat = eng.load(f'{path}{file_name}.mat')
data = mat['hil_resha_aligned']
lbl = mat['lbl_out']
del(mat) # cleanup

# resize the labels to the data length
if lbl.size[1] > data.size[1]:
    lbl = np.array(lbl)[:,:data.size[1]]

# split sequence into samples
samples = []
labels = []
data = np.array(data)
for i in range(int(data.shape[1]/window)):
    sample = data[:,int(i*window):int((i+1)*window),:]
    label = lbl[:,int(i*window):int((i+1)*window)]
    samples.append(sample)
    labels.append(label)
del(data) # cleanup

print('Starting processing')
# process individual samples
sample_PC = []
for sample in samples:
    node_PC = []
    # process individual nodes
    for node in range(sample.shape[2]):
        PC = eng.PointCloud(eng.raw2PC(sample[:,:,node], matlab.double(chunks), matlab.double(npoints), matlab.double(thr))) # point cloud generation
        node_PC.append(PC)
    sample_PC.append(node_PC)

save_PC('./py_test/', sample_PC, labels)

#train_cls.main()

