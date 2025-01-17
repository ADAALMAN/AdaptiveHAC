import numpy as np
import matplotlib.pyplot as plt
import matlab

def H_score(tr_avg, lbl, data_len, eng):
    tr_GT = eng.sig2timestamp(lbl, np.linspace(0, data_len-1, num=data_len), 'nonzero')
    if isinstance(tr_GT, float):
        tr_GT = np.asarray([0.0])[:,np.newaxis]
    elif (tr_GT.size[1] == 0):
        tr_GT = np.asarray([0.0])[:,np.newaxis]


    # compute score
    # give 1 score to the entire segment
    H_score, _ = eng.perfFuncLin(tr_avg, tr_GT,  data_len/60,nargout=2)
    # evaluate each transition
    H_score_trans, _ = eng.perfFuncLinSeg(tr_avg, tr_GT,  data_len/180,nargout=2)
    return H_score

def segmentation(data, lbl, eng, args): # multinode segmentation
    
    # create spectrogram
    spectrogram, t, f = eng.process(data, f'{args.root}/segmentation/config_monostatic_TUD.mat', nargout=3)
    plt.figure(0)
    plt.imshow(np.asarray(spectrogram)[:,:,0],cmap='viridis')
    plt.colorbar(label='Magnitude (dB)')
    plt.title('Spectrogram')
    
    data_len = data.shape[1]
    

    entropy = np.asarray(eng.renyi(spectrogram, nargout=1))
    PBC = np.asarray(eng.pbc(spectrogram, f'{args.root}/segmentation/config_monostatic_TUD.mat', nargout=1))
    plt.figure(1)
    plt.plot(np.linspace(0, t.size[1], num=t.size[1]), entropy[:,0][:,np.newaxis])
    plt.title('Entropy')
    del(spectrogram)
                
    # implement measure to choose timestamps from entropy
    # temporarily take the mean
    # 5-node averaged H
    entropy_avg = np.mean(entropy, axis=1)
    PBC_avg = np.mean(PBC, axis=1)
    _, s_avg, _ = eng.lagSearch(entropy_avg, matlab.double(47), matlab.double(args.lag_search_th), nargout=3)
    
    plt.figure(2)
    plt.plot(np.linspace(0, t.size[1], num=t.size[1]), s_avg*np.max(entropy[:,0]))
    plt.plot(np.linspace(0, t.size[1], num=t.size[1]), entropy[:,0][:,np.newaxis])
    plt.title('Lag')
    plt.close('all')
    
    tr_avg = eng.sig2timestamp(s_avg, np.linspace(0, data_len-1, num=data_len), nargout=1)   

    H_avg_score = H_score(tr_avg, lbl, data_len, eng)

    index = np.asarray(tr_avg)
    index = np.insert(index, 0, 0, axis=1)
    index = np.append(index, data_len)

    segments = []
    labels = []
    for i in range(len(index)-1):
        segments.append(data[:,int(index[i]):int(index[i+1]),:])
        labels.append(lbl[:,int(index[i]):int(index[i+1])])
    return segments, labels, H_avg_score, entropy_avg, PBC_avg

def SNsegmentation(data, lbl, eng, args): # single node segmentation
    
    # create spectrogram
    spectrogram, t, f = eng.process(data, f'{args.root}/segmentation/config_monostatic_TUD.mat', nargout=3)
    plt.figure(0)
    plt.imshow(np.asarray(spectrogram)[:,:],cmap='viridis')
    plt.colorbar(label='Magnitude (dB)')
    plt.title('Spectrogram')
    
    data_len = data.shape[1]

    entropy = np.asarray(eng.renyi(spectrogram, nargout=1))
    PBC = np.asarray(eng.pbc(spectrogram, f'{args.root}/segmentation/config_monostatic_TUD.mat', nargout=1))
    plt.figure(1)
    plt.plot(np.linspace(0, t.size[1], num=t.size[1]), entropy[:,0][:,np.newaxis])
    plt.title('Entropy')
    del(spectrogram)
                
    # implement measure to choose timestamps from entropy
    # temporarily take the mean
    # 5-node averaged H
    entropy_avg = np.mean(entropy, axis=1)
    PBC_avg = np.mean(PBC, axis=1)
    _, s_avg, _ = eng.lagSearch(entropy_avg, matlab.double(47), matlab.double(args.lag_search_th), nargout=3)
    
    plt.figure(2)
    plt.plot(np.linspace(0, t.size[1], num=t.size[1]), s_avg*np.max(entropy[:,0]))
    plt.plot(np.linspace(0, t.size[1], num=t.size[1]), entropy[:,0][:,np.newaxis])
    plt.title('Lag')
    plt.close('all')
    
    tr_avg = eng.sig2timestamp(s_avg, np.linspace(0, data_len-1, num=data_len), nargout=1)   

    H_avg_score = H_score(tr_avg, lbl, data_len, eng)

    index = np.asarray(tr_avg)
    index = np.insert(index, 0, 0, axis=1)
    index = np.append(index, data_len)

    segments = []
    labels = []
    for i in range(len(index)-1):
        segments.append(data[:,int(index[i]):int(index[i+1])])
        labels.append(lbl[:,int(index[i]):int(index[i+1])])
    return segments, labels, H_avg_score, entropy_avg, PBC_avg

def GTsegmentation(data, lbl):
    index = np.where(lbl[:, :-1] != lbl[:, 1:])[1]
    index = np.append(index, len(lbl[0])-1)
    
    segments = []
    labels = []
    for i in range(len(index)-1):
        segments.append(data[:,int(index[i]):int(index[i+1]),:])
        labels.append(lbl[:,int(index[i]):int(index[i+1])])
    return segments, labels

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