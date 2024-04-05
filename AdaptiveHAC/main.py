import matlab.engine
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
@hydra.main(config_path="conf", config_name="paramsweep", version_base='1.1')
def main(args):
    omegaconf.OmegaConf.set_struct(args, False)
    data_path = hydra.utils.to_absolute_path(args.data_path)
    
    # data path
    #path = 'W:/staff-groups/ewi/me/MS3/MS3-Shared/Ronny_MonostaticData/Nicolas/MAT_data_aligned/'
    #file_name = '001_mon_Wal_Nic'
    file_name = '029_mon_Mix_Nic'
    
    match args.node_method:
        case "all":
            data, lbl = load_data(data_path, file_name)
        case "individual":
            return #WIP
            i = 0
            data, lbl = load_data(data_path, file_name)
            data = data[:,:, i]
    
    match args.sample_method:
        case "windowing":
            seg_th = "NA"
            samples, labels = PC_processing.sample(data, lbl, sample_size = 'default')
        case "segmentation":
            seg_th = 100
            segmentation_eng = segmentation.init_matlab(args.root)
            samples, labels, H_avg_score = segmentation.segmentation(data, lbl, segmentation_eng, args.root)
            samples, labels = segmentation.segmentation_thresholding(samples, labels, seg_th, "split")
    del(data, lbl)
    
    match args.features:
        case "none":
            pass
        
    npoints = 1024
    thr = 0.8
    match args.subsegmentation:
        case "fixed-amount":
            chunks = 6
            features = []
            processing_eng = PC_processing.init_matlab(args.root)
            samples_PC = PC_processing.PC_generation(samples, args.subsegmentation, chunks, npoints, thr, features, labels, processing_eng)
        case "fixed-length":
            return
    
    del(samples)
    os.mkdir(f'./seg_th_{seg_th}/')
    PC_processing.save_PC(f'./seg_th_{seg_th}/', samples_PC, labels)

    PT_args = load_PT_config(args.PT_config_path)
    train_cls.main([PT_args, samples_PC])

if __name__ == '__main__':
    main()
    #train_cls.main()