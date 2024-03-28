import matlab.engine
import numpy as np
import matplotlib.pyplot as plt
from AdaptiveHAC.lib import timing_decorator
eng = matlab.engine.start_matlab()
eng.addpath('segmentation')

def H_score(entropy, tr2, data_len):
    H_avg = np.zeros((entropy.shape[1],entropy.shape[1],entropy.shape[0])) #5x5x1813
    H_score = np.zeros((entropy.shape[1],entropy.shape[1])) #5x5

    # computing H-score
    for i in range(entropy.shape[1]): # compare each node
        for j in range(entropy.shape[1]):
            if i<j:
                H_avg[i,j,:] = np.mean(entropy[:, [i, j]], axis=1) # average the entrpy of 2 nodes
            elif i == j:
                H_avg[i,j,:] = entropy[:,i]
            # H-time stamps
                
            d1 = H_avg[i,j,:].reshape(-1,1)
            _, s1, _ = eng.lagSearch(d1, nargout=3)
            tr1 = eng.sig2timestamp(s1, np.linspace(0, data_len-1, num=s1.size[0]), nargout=1)
            # compute score
            if i <= j:
                H_score[i,j], _ = eng.perfFuncLin(tr1, tr2, data_len/60,nargout=2)
    return H_score

    
#@timing_decorator.timing_decorator
def segmentation(data, lbl):
    
    # create spectogram
    spectogram, t, f = eng.process(data, './segmentation/config_monostatic_TUD.mat', nargout=3)
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

    tr_avg = eng.sig2timestamp(s_avg, np.linspace(0, data_len-1, num=s_avg.size[0]), nargout=1)   
    tr_GT = eng.sig2timestamp(lbl, np.linspace(0, data_len-1, num=s_avg.size[0]), 'nonzero')
    if tr_GT.size[1] == 0:
        tr_GT = np.asarray([0.0])[:,np.newaxis]

    d1 = np.mean(entropy,axis=1)
    _, s1, _ = eng.lagSearch(d1, nargout=3)
    tr1 = eng.sig2timestamp(s1, t, nargout=1)
    # compute score
    H_avg_score, _ = eng.perfFuncLin(tr_avg, tr_GT,  data_len/60,nargout=2)

    
    H_score(entropy, tr_GT, data_len)

    config = eng.load(f'./segmentation/config_monostatic_TUD.mat')

    index = np.asarray(tr_avg)
    index = np.insert(index, 0, 0, axis=1)
    index = np.append(index, data_len)

    segments = []
    labels = []
    for i in range(len(index)-1):
        segments.append(data[:,int(index[i]):int(index[i+1]),:])
        labels.append(lbl[:,int(index[i]):int(index[i+1])])
    return segments, labels

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