import matlab.engine
import numpy as np
import os, sys, time
from AdaptiveHAC.pointTransformer import train_cls
from AdaptiveHAC.processing import PointCloud
from pympler import asizeof

# initialize matlab
os.environ['HYDRA_FULL_ERROR'] = '1'
eng = matlab.engine.start_matlab()
eng.addpath('processing')
eng.addpath('segmentation')

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time} seconds to run.")
        return result
    return wrapper

@timing_decorator
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

@timing_decorator
def load_data(path, file_name = None):
    # load data file with the matlab engine and unpack data
    if file_name != None:
        mat = eng.load(f'{path}{file_name}.mat')
        data = mat['hil_resha_aligned']
        lbl = mat['lbl_out']
        del(mat) # cleanup

    # resize the labels to the data length
    if lbl.size[1] > data.size[1]:
        lbl = np.array(lbl)[:,:data.size[1]]

    return data, lbl

@timing_decorator
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

@timing_decorator
def PC_generation(samples, chunks, npoints, thr):
    print('Starting processing')
    # process individual samples
    samples_PC = []
    for sample in samples:
        node_PC = []
        # process individual nodes
        for node in range(sample.shape[2]):
            PC = PointCloud.PointCloud(np.asarray(eng.raw2PC(sample[:,:,node], 
                                                            matlab.double(chunks), 
                                                            matlab.double(npoints), 
                                                            matlab.double(thr)))) # point cloud generation
            node_PC.append(PC)
        samples_PC.append(node_PC)
    return samples_PC

@timing_decorator
def segmentation(data, lbl):
    
    # create spectogram
    spectogram, t, f = eng.process(data, './segmentation/config_monostatic_TUD.mat', nargout=3)
    data_len = data.shape[1]
    #del(data)

    entropy = np.asarray(eng.renyi(spectogram, nargout=1))
    del(spectogram)
    H_avg = np.zeros((entropy.shape[1], entropy.shape[1], entropy.shape[0]))
    H_score = np.zeros((entropy.shape[1], entropy.shape[1]))

    # GT time stamps
    tr2 = eng.sig2timestamp(lbl.T, t,'nonzero', nargout=1)

    # compare node entropy
    for i in range(entropy.shape[1]):
            for j in range(entropy.shape[1]):
                if i<j:
                    H_avg[i,j,:] = np.mean(entropy[:,[i,j]], axis=1)
                elif i == j:
                    H_avg[i,j,:] = entropy[:,i]
                # H-time stamps
                d1 = H_avg[i,j,:].reshape(-1,1)
                _, lag, _ = eng.lagSearch(d1, nargout=3)
                tr1 = eng.sig2timestamp(lag, t, nargout=1)
                # compute score
                #if i <= j: 
                    #H_score[i,j], _ = eng.perfFuncLin(tr1,tr2, nargout=2)
                
    # implement measure to choose timestamps from entropy
    # temporarily take the mean
    # 5-node averaged H
    d_avg = np.mean(entropy, axis=1)
    del(entropy)
    _, s_avg, _ = eng.lagSearch(d_avg, nargout=3)
    tr_avg = eng.sig2timestamp(s_avg, t, nargout=1)   

    config = eng.load(f'./segmentation/config_monostatic_TUD.mat')
    Ts = config['sample_time']

    index = np.asarray(tr_avg)/Ts
    index = np.insert(index, 0, 0, axis=1)
    index = np.append(index, data_len)#int(np.asarray(t).T[-1]/Ts)) # would be better with data.shape[1], but takes too long

    segments = []
    for i in range(len(index)-1):
        segments.append(data[:,int(index[i]):int(index[i+1]),:])
    
    return segments

def main():
    # data path
    #path = 'W:/staff-groups/ewi/me/MS3/MS3-Shared/Ronny_MonostaticData/Nicolas/MAT_data_aligned/'
    path = './test/data/'
    file_name = '029_mon_Mix_Nic'

    # paramenters
    chunks = 6
    npoints = 1024
    thr = 0.8

    data, lbl = load_data(path, file_name)
    data = np.asarray(data)
    segments = segmentation(data, lbl)
    del(data)
    
    """ samples, labels = sample(data, lbl, sample_size = 'default')
    del(data, lbl)
    samples_PC = PC_generation(samples, chunks, npoints, thr)
    del(samples)
    save_PC('./py_test/', samples_PC, labels) """

    #train_cls.main()

if __name__ == '__main__':
    main()