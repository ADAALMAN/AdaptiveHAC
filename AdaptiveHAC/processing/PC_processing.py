from types import NoneType
import matlab.engine
import numpy as np
import os, sys
from AdaptiveHAC.processing import PointCloud
from AdaptiveHAC.lib import timing_decorator

def init_matlab(root):
    # initialize matlab
    os.environ['HYDRA_FULL_ERROR'] = '1'
    eng = matlab.engine.start_matlab()
    eng.addpath(f'{root}/processing')
    return eng

def save_PC_txt(dir, PC, labels):
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

def SNsample(data, lbl, sample_size = 'default'):
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
        sample = data[:,int(i*window):int((i+1)*window)]
        label = lbl[:,int(i*window):int((i+1)*window)]
        samples.append(sample)
        labels.append(label)
    del(data) # cleanup
    return samples, labels

def PC_generation(samples, subsegmentation, param, npoints, thr, features, labels, eng):
    # process individual samples
    samples_PC = []
    for i, (sample, label) in enumerate(zip(samples, labels)):
        node_PC = []
        # process individual nodes
        for node in range(sample.shape[2]):
            PC = PointCloud.PointCloud(np.asarray(eng.raw2PC(sample[:,:,node], 
                                                            subsegmentation,
                                                            matlab.double(param), 
                                                            matlab.double(npoints), 
                                                            matlab.double(thr),
                                                            ("standard" if not (features["time"]) else features["time"][:]))), # cant use i
                                       label) # point cloud generation
            
            if not features == False:
                ft = []
                for key, item in features.items():
                    if key != "time":
                        ft.append(features[key][i])
                PC.add_features(ft)
                
            PC.normalise()
            #PC.visualise()
            node_PC.append(PC)
        samples_PC.append(node_PC)
    return samples_PC

def SNPC_generation(samples, subsegmentation, param, npoints, thr, features, labels, eng):
    # process individual samples
    samples_PC = []
    for sample, label in zip(samples, labels):
        PC = PointCloud.PointCloud(np.asarray(eng.raw2PC(sample[:,:], 
                                                        subsegmentation,
                                                        matlab.double(param), 
                                                        matlab.double(npoints), 
                                                        matlab.double(thr),
                                                        features)),
                                    label) # point cloud generation
        PC.visualise()
        samples_PC.append(PC)
    return samples_PC

#eng.eval("dbstop in raw2PC at 10", nargout=0)