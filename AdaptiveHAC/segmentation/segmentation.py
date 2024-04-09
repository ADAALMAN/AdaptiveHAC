import matlab.engine
import numpy as np
import matplotlib.pyplot as plt
from AdaptiveHAC.lib import timing_decorator
import os

def init_matlab(root):
    # initialize matlab
    os.environ['HYDRA_FULL_ERROR'] = '1'
    eng = matlab.engine.start_matlab()
    eng.addpath(f'{root}/segmentation')
    return eng

def H_score(tr_avg, lbl, data_len, eng):
    tr_GT = eng.sig2timestamp(lbl, np.linspace(0, data_len-1, num=data_len), 'nonzero')
    if tr_GT.size[1] == 0:
        tr_GT = np.asarray([0.0])[:,np.newaxis]

    # compute score
    # give 1 score to the entire segment
    H_score, _ = eng.perfFuncLin(tr_avg, tr_GT,  data_len/60,nargout=2)
    # evaluate each transition
    H_score_trans, _ = eng.perfFuncLinSeg(tr_avg, tr_GT,  data_len/180,nargout=2)
    return H_score

    
#@timing_decorator.timing_decorator
def segmentation(data, lbl, eng, root): # multinode segmentation
    
    # create spectogram
    spectogram, t, f = eng.process(data, f'{root}/segmentation/config_monostatic_TUD.mat', nargout=3)
    plt.figure(0)
    plt.imshow(np.asarray(spectogram)[:,:,0],cmap='viridis')
    plt.colorbar(label='Magnitude (dB)')
    plt.title('Spectrogram')
    
    data_len = data.shape[1]
    

    entropy = np.asarray(eng.renyi(spectogram, nargout=1))
    plt.figure(1)
    plt.plot(np.linspace(0, t.size[1], num=t.size[1]), entropy[:,0][:,np.newaxis])
    plt.title('Entropy')
    del(spectogram)
                
    # implement measure to choose timestamps from entropy
    # temporarily take the mean
    # 5-node averaged H
    d_avg = np.mean(entropy, axis=1)
    _, s_avg, _ = eng.lagSearch(d_avg, nargout=3)
    
    plt.figure(2)
    plt.plot(np.linspace(0, t.size[1], num=t.size[1]), s_avg*np.max(entropy[:,0]))
    plt.plot(np.linspace(0, t.size[1], num=t.size[1]), entropy[:,0][:,np.newaxis])
    plt.title('Lag')
    #plt.show()

    tr_avg = eng.sig2timestamp(s_avg, np.linspace(0, data_len-1, num=data_len), nargout=1)   

    H_avg_score = H_score(tr_avg, lbl, data_len, eng)

    index = np.asarray(tr_avg)
    index = np.insert(index, 0, 0, axis=1)
    index = np.append(index, data_len)

    avg_entropies = []
    segments = []
    labels = []
    for i in range(len(index)-1):
        avg_entropies.append(d_avg[int(index[i]/data.shape[1]*len(d_avg)):int(index[i+1]/data.shape[1]*len(d_avg))])
        segments.append(data[:,int(index[i]):int(index[i+1]),:])
        labels.append(lbl[:,int(index[i]):int(index[i+1])])
    return segments, labels, H_avg_score, avg_entropies

def SNsegmentation(data, lbl, eng, root): # single node segmentation
    
    # create spectogram
    spectogram, t, f = eng.process(data, f'{root}/segmentation/config_monostatic_TUD.mat', nargout=3)
    plt.figure(0)
    plt.imshow(np.asarray(spectogram)[:,:],cmap='viridis')
    plt.colorbar(label='Magnitude (dB)')
    plt.title('Spectrogram')
    
    data_len = data.shape[1]
    

    entropy = np.asarray(eng.renyi(spectogram, nargout=1))
    plt.figure(1)
    plt.plot(np.linspace(0, t.size[1], num=t.size[1]), entropy[:,0][:,np.newaxis])
    plt.title('Entropy')
    del(spectogram)
                
    # implement measure to choose timestamps from entropy
    # temporarily take the mean
    # 5-node averaged H
    d_avg = np.mean(entropy, axis=1)
    _, s_avg, _ = eng.lagSearch(d_avg, nargout=3)
    
    plt.figure(2)
    plt.plot(np.linspace(0, t.size[1], num=t.size[1]), s_avg*np.max(entropy[:,0]))
    plt.plot(np.linspace(0, t.size[1], num=t.size[1]), entropy[:,0][:,np.newaxis])
    plt.title('Lag')
    #plt.show()

    tr_avg = eng.sig2timestamp(s_avg, np.linspace(0, data_len-1, num=data_len), nargout=1)   

    H_avg_score = H_score(tr_avg, lbl, data_len, eng)

    index = np.asarray(tr_avg)
    index = np.insert(index, 0, 0, axis=1)
    index = np.append(index, data_len)

    avg_entropies = []
    segments = []
    labels = []
    for i in range(len(index)-1):
        avg_entropies.append(d_avg[int(index[i]/data.shape[1]*len(d_avg)):int(index[i+1]/data.shape[1]*len(d_avg))])
        segments.append(data[:,int(index[i]):int(index[i+1])])
        labels.append(lbl[:,int(index[i]):int(index[i+1])])
    return segments, labels, H_avg_score, avg_entropies

@timing_decorator.timing_decorator
def segmentation_thresholding(segments, labels, threshold, method="shortest"):
    match method:
        case "split":
            j = 0
            for i in range(len(segments)):
                i = i-j
                if segments[i].shape[1] < threshold:
                    if i == (len(segments)-1):
                        segments[i-1] = np.append(segments[i-1], segments[i], axis=1)
                        labels[i-1] = np.append(labels[i-1], labels[i], axis=1)
                        segments.pop(i)
                        labels.pop(i)
                        j = j + 1
                    elif i == 0:
                        segments[i+1] = np.concatenate((segments[i], segments[i+1]), axis=1)
                        labels[i+1] = np.concatenate((labels[i], labels[i+1]), axis=1)
                        segments.pop(i)
                        labels.pop(i)
                        j = j + 1
                    else:
                        segments[i-1] = np.concatenate((segments[i-1], np.array_split(segments[i], 2, axis=1)[0]), axis=1)
                        segments[i+1] = np.concatenate((np.array_split(segments[i], 2, axis=1)[1], segments[i+1]), axis=1)
                        labels[i-1] = np.concatenate((labels[i-1], np.array_split(labels[i], 2, axis=1)[0]), axis=1)
                        labels[i+1] = np.concatenate((np.array_split(labels[i], 2, axis=1)[1], labels[i+1]), axis=1)
                        segments.pop(i)
                        labels.pop(i)
                        j = j + 1
        case "shortest":
            j = 0
            for i in range(len(segments)):
                i = i-j
                if segments[i].shape[1] < threshold:
                    if i == (len(segments)-1):
                        segments[i-1] = np.append(segments[i-1], segments[i], axis=1)
                        labels[i-1] = np.append(labels[i-1], labels[i], axis=1)
                        segments.pop(i)
                        labels.pop(i)
                        j = j + 1
                    elif i == 0:
                        segments[i+1] = np.concatenate((segments[i], segments[i+1]), axis=1)
                        labels[i+1] = np.concatenate((labels[i], labels[i+1]), axis=1)
                        segments.pop(i)
                        labels.pop(i)
                        j = j + 1
                    elif segments[i-1].shape[1] < segments[i+1].shape[1]:
                        segments[i-1] = np.append(segments[i-1], segments[i], axis=1)
                        labels[i-1] = np.append(labels[i-1], labels[i], axis=1)
                        segments.pop(i)
                        labels.pop(i)
                        j = j + 1
                    elif segments[i-1].shape[1] > segments[i+1].shape[1]:
                        segments[i+1] = np.append(segments[i+1], segments[i], axis=1)
                        labels[i+1] = np.append(labels[i+1], labels[i], axis=1)
                        segments.pop(i)
                        labels.pop(i)
                        j = j + 1
        case _:
            print("select method")
    return segments, labels