import matlab.engine
import numpy as np

from AdaptiveHAC.pointTransformer import train_cls

import os 

os.environ['HYDRA_FULL_ERROR'] = '1'
""" eng = matlab.engine.start_matlab()
eng.addpath('processing')
eng.addpath('segmentation')

chunks = 6
npoints = 1024
thr = 0.8

#path = 'W:/staff-groups/ewi/me/MS3/MS3-Shared/Ronny_MonostaticData/Nicolas/MAT_data_aligned/'
path = './data/'
file_name = '002_mon_wal_Nic'ß

mat = eng.load(f'{path}{file_name}.mat')
mat = mat['hil_resha_aligned']
mat = np.array(mat)[:,:,1]
#mat = matlab.double(mat, is_complex=True)
print('Starting processing')
raw_PC = eng.raw2PC(mat, matlab.double(chunks), matlab.double(npoints), matlab.double(thr))
print(raw_PC.size) """

train_cls.main()

