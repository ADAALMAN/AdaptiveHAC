import matlab.engine
import numpy as np
import os, sys
from AdaptiveHAC.processing import PointCloud
from AdaptiveHAC.lib import timing_decorator

# initialize matlab
os.environ['HYDRA_FULL_ERROR'] = '1'
eng = matlab.engine.start_matlab()
eng.addpath('processing')

@timing_decorator.timing_decorator
def save_PC(dir, PC, labels):
    activities = ['na','wlk','stat','sitdn','stupsit','bfrsit','bfrstand','ffw','stup','ffs']
    for act in activities:
        if os.path.isdir(f'{dir}/{act}') == False:
            os.mkdir(f'{dir}/{act}')
        elif os.path.isdir(f'{dir}/{act}') == True:
            files = os.listdir(f'{dir}/{act}')
            for file in files:
                os.remove(os.path.join(f'{dir}/{act}', file)) # remove all existing files

    for sample, label, i in zip(PC, labels, range(len(PC))):
        sample_act = activities[int(np.mean(label))]
        for node in sample:
            np.savetxt(f'{dir}/{sample_act}/{sample_act}_{i+1}.txt', node.normalise().data, fmt='%f')
    return

@timing_decorator.timing_decorator
def sample(data, lbl, sample_size = 'default'):
    # split sequence into samples
    samples = []
    labels = []
    data = np.array(data)

    if sample_size == 'default':
        window = 256
        sample_size = int(data.shape[1]/window)
    else:
        sample_size = int(sample_size)

    for i in range(sample_size):
        sample = data[:,int(i*window):int((i+1)*window),:]
        label = lbl[:,int(i*window):int((i+1)*window)]
        samples.append(sample)
        labels.append(label)
    del(data) # cleanup
    return samples, labels

@timing_decorator.timing_decorator
def PC_generation(samples, subsegmentation, chunks, npoints, thr, features, labels):
    print('Starting processing')
    # process individual samples
    samples_PC = []
    for sample, label in zip(samples, labels):
        node_PC = []
        # process individual nodes
        for node in range(sample.shape[2]):
            PC = PointCloud.PointCloud(np.asarray(eng.raw2PC(sample[:,:,node], 
                                                            subsegmentation,
                                                            matlab.double(chunks), 
                                                            matlab.double(npoints), 
                                                            matlab.double(thr),
                                                            features)),
                                       label) # point cloud generation
            #PC.visualise()
            node_PC.append(PC)
        samples_PC.append(node_PC)
    return samples_PC

#eng.eval("dbstop in raw2PC at 10", nargout=0)