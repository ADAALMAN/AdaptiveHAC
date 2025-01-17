import numpy as np
import os, hydra, omegaconf, logging, gc
from AdaptiveHAC.lib import load_data
from AdaptiveHAC.segmentation import segmentation
from AdaptiveHAC.processing import PC_processing
import matlab.engine
import pandas as pd
logger = logging.getLogger(__name__) 

def init_matlab(root):
    # initialize matlab
    os.environ['HYDRA_FULL_ERROR'] = '1'
    eng = matlab.engine.start_matlab()
    eng.addpath(f'{root}/segmentation')
    eng.addpath(f'{root}/processing')
    return eng
    
def process(args, file_name):
    omegaconf.OmegaConf.set_struct(args, False)
    data_path = hydra.utils.to_absolute_path(args.data_path)
    eng = init_matlab(args.root)
    #print(eng.version())
    #print('\n',eng.license('checkout','Signal_Toolbox', nargout=2), eng.license('checkout','Image_Toolbox', nargout=2), eng.license('checkout','Statistics_Toolbox', nargout=2))
    match args.node_method:
        case "all":
            data, lbl = load_data.load_data(data_path, file_name)
        case "individual":
            return # depreciated
            i = 0
            data, lbl = load_data.load_data(data_path, file_name)
            data = data[:,:, i]
        case _ if isinstance(args.node_method, int):
            data, lbl = load_data.load_data(data_path, file_name)
            data = data[:,:, args.node_method]

    match args.sample_method:
        case "windowing":
            seg_th = "NA"
            H_avg_score = "NA"
            if isinstance(args.node_method, int):
                samples, labels = PC_processing.SNsample(data, lbl, sample_size = 'default')
            else:
                samples, labels = PC_processing.sample(data, lbl, sample_size = 'default')
        case "segmentation":
            seg_th = 100
            if isinstance(args.node_method, int):
                samples, labels, H_avg_score, entropy, PBC = segmentation.SNsegmentation(data, lbl, eng, args)
            else:
                samples, labels, H_avg_score, entropy, PBC = segmentation.segmentation(data, lbl, eng, args)
            samples, labels = segmentation.segmentation_thresholding(samples, labels, seg_th, "split")
            pd.DataFrame(entropy).to_csv(os.path.join(f'./entropy/{file_name}_entropy.csv'), index=False) 
        case "GTsegmentation":
            seg_th = 100
            H_avg_score = "NA"
            samples, labels, = segmentation.GTsegmentation(data, lbl)
            samples, labels = segmentation.segmentation_thresholding(samples, labels, seg_th, "split")
            
    data_len = data.shape[1]  
    del data, lbl
    gc.collect()

    features = {}
    for feature in args.features:
        match feature:
            case "none":
                pass
            case "entropy":
                entropies = []
                j = 0
                for i in range(len(samples)):
                    entropies.append(np.mean(entropy[int(j/data_len*len(entropy)):
                                                     int((j+samples[i].shape[1])/data_len*len(entropy))]))
                    j = j + samples[i].shape[1]
                features["entropy"] = entropies
            case "PBC":
                PBCs = []
                j = 0
                for i in range(len(samples)):
                    PBCs.append(np.mean(PBC[int(j/data_len*len(PBC)):
                                                     int((j+samples[i].shape[1])/data_len*len(PBC))]))
                    j = j + samples[i].shape[1]
                features["PBC"] = PBCs
            case "time":
                features["time"] = ["sequence-based", data_len]
        
    npoints = 1024
    thr = 0.8
    match args.subsegmentation:
        case "fixed-amount":
            if isinstance(args.subsegparam, int):
                param = args.subsegparam # amount of subsegments
            else:
                param = 6
                
            if isinstance(args.node_method, int):
                samples_PC = PC_processing.SNPC_generation(samples, args.subsegmentation, param, npoints, thr, features, labels, eng)
            else:
                samples_PC = PC_processing.PC_generation(samples, args.subsegmentation, param, npoints, thr, features, labels, eng)
        case "fixed-length":
            if isinstance(args.subsegparam, int):
                param = args.subsegparam # subsegments length
            else:
                param = 40
                
            if isinstance(args.node_method, int):
                samples_PC = PC_processing.SNPC_generation(samples, args.subsegmentation, param, npoints, thr, features, labels, eng)
            else:
                samples_PC = PC_processing.PC_generation(samples, args.subsegmentation, param, npoints, thr, features, labels, eng)  
                
    if isinstance(args.node_method, int):
        for index, PC in enumerate(samples_PC):
            PC.add_attributes(file_name, index, H_avg_score)
    else:
        for index, PC in enumerate(samples_PC):
            for node in PC:
                node.add_attributes(file_name, index, H_avg_score)
    
    eng.quit()
    del samples, labels
    gc.collect()
    return samples_PC