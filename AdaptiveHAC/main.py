import numpy as np
import os, sys, hydra, omegaconf, yaml, argparse
from AdaptiveHAC.pointTransformer import train_cls
from AdaptiveHAC.lib import timing_decorator
from AdaptiveHAC.segmentation import segmentation
from AdaptiveHAC.processing import PC_processing
from memory_profiler import profile
import scipy.io as sci
np.set_printoptions(threshold=sys.maxsize)

# initialize matlab
os.environ['HYDRA_FULL_ERROR'] = '1'

#@timing_decorator.timing_decorator
def load_data(path, file_name = None):
    # load data file with the matlab engine and unpack data
    if file_name != None:
        mat = sci.loadmat(f'{path}/{file_name}.mat')
        data = np.asarray(mat['hil_resha_aligned'], dtype=np.complex64) # reduce data in memory
        lbl = np.asarray(mat['lbl_out'])

        # resize the labels to the data length
        if lbl.shape[1] > data.shape[1]:
            lbl = np.array(lbl)[:,:data.shape[1]]

    return data, lbl   
            
def load_PT_config(PT_config_path):
    with open(f'{PT_config_path}cls.yaml', 'r') as file:
        dict_args =  yaml.safe_load(file)
        
    if 'defaults' in dict_args:
        for default in dict_args['defaults']:
            if isinstance(default, dict):
                for key, value in default.items():
                    with open(f'{PT_config_path}{key}/{value}.yaml', 'r') as file:
                        ref_config = yaml.safe_load(file)
                    dict_args[key] = argparse.Namespace(**ref_config)
                dict_args.pop('defaults', None)
                
    dict_args = argparse.Namespace(**dict_args)
    dict_args.experiment_folder = './'
    return dict_args

#@profile
@hydra.main(config_path="conf", config_name="paramsweep", version_base='1.3')
def main(args):
    omegaconf.OmegaConf.set_struct(args, False)
    data_path = hydra.utils.to_absolute_path(args.data_path)
    
    # data path
    #path = 'W:/staff-groups/ewi/me/MS3/MS3-Shared/Ronny_MonostaticData/Nicolas/MAT_data_aligned/'
    #file_name = '002_mon_Wal_Nic'
    file_name = '029_mon_Mix_Nic'
    
    match args.node_method:
        case "all":
            data, lbl = load_data(data_path, file_name)
        case "individual":
            return # depreciated
            i = 0
            data, lbl = load_data(data_path, file_name)
            data = data[:,:, i]
        case _ if isinstance(args.node_method, int):
            data, lbl = load_data(data_path, file_name)
            data = data[:,:, args.node_method]
    
    match args.sample_method:
        case "windowing":
            seg_th = "NA"
            if isinstance(args.node_method, int):
                samples, labels = PC_processing.SNsample(data, lbl, sample_size = 'default')
            else:
                samples, labels = PC_processing.sample(data, lbl, sample_size = 'default')
        case "segmentation":
            seg_th = 100
            segmentation_eng = segmentation.init_matlab(args.root)
            if isinstance(args.node_method, int):
                samples, labels, H_avg_score, entropy, PBC = segmentation.SNsegmentation(data, lbl, segmentation_eng, args.root)
            else:
                samples, labels, H_avg_score, entropy, PBC = segmentation.segmentation(data, lbl, segmentation_eng, args.root)
            samples, labels = segmentation.segmentation_thresholding(samples, labels, seg_th, "split")
            
    data_len = data.shape[1]        
    del(data, lbl)
    
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
                return
                #dict["time"]
        
    npoints = 1024
    thr = 0.8
    match args.subsegmentation:
        case "fixed-amount":
            param = 6 # amount of subsegments
            processing_eng = PC_processing.init_matlab(args.root)
            if isinstance(args.node_method, int):
                samples_PC = PC_processing.SNPC_generation(samples, args.subsegmentation, param, npoints, thr, features, labels, processing_eng)
            else:
                samples_PC = PC_processing.PC_generation(samples, args.subsegmentation, param, npoints, thr, features, labels, processing_eng)
        case "fixed-length":
            param = 20 # subsegment length
            processing_eng = PC_processing.init_matlab(args.root)
            if isinstance(args.node_method, int):
                samples_PC = PC_processing.SNPC_generation(samples, args.subsegmentation, param, npoints, thr, features, labels, processing_eng)
            else:
                samples_PC = PC_processing.PC_generation(samples, args.subsegmentation, param, npoints, thr, features, labels, processing_eng)        
    
    del(samples)
    os.mkdir(f'./seg_th_{seg_th}/')
    #PC_processing.save_PC_txt(f'./seg_th_{seg_th}/', samples_PC, labels) #needs to be updated for single node

    PT_args = load_PT_config(args.PT_config_path)
    train_cls.main([PT_args, samples_PC])

if __name__ == '__main__':
    main()
    #train_cls.main()