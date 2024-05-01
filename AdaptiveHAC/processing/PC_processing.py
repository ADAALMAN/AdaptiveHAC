import matlab.engine
import numpy as np
import os
from AdaptiveHAC.processing import PointCloud

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
                                                            (features["time"][0] if ("time" in features.keys()) else "standard"))), # cant use i
                                       label) # point cloud generation
            
            if not features == False:
                ft = []
                for key in features.keys():
                    if key != "time":
                        ft.append(features[key][i])
                PC.add_features(ft, features["time"] if ("time" in features.keys()) else None)
                
            PC.normalise()
            #PC.visualise()
            if PC.mean_label == 0: # filter out all N/A pointclouds
                continue
            else:
                node_PC.append(PC)
        if len(node_PC) == 0: # to prevent adding emty lists
            continue
        else:
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
                                                        (features["time"][:] if ("time" in features.keys()) else "standard"))),
                                    label) # point cloud generation
        
        if not features == False:
            ft = []
            for key in features.keys():
                if key != "time":
                    ft.append(features[key])
            PC.add_features(ft, features["time"] if ("time" in features.keys()) else None)
                
        #PC.visualise()
        if PC.mean_label == 0: # filter out all N/A pointclouds
            continue
        else:
            samples_PC.append(PC)
    return samples_PC

#eng.eval("dbstop in raw2PC at 10", nargout=0)