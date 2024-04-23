import numpy as np
import os, sys, hydra, omegaconf, yaml, argparse, logging, pickle
from AdaptiveHAC.pointTransformer import train_cls, point_transformer
from AdaptiveHAC.lib import timing_decorator
from AdaptiveHAC.segmentation import segmentation
from AdaptiveHAC.processing import PC_processing, PointCloud
import scipy.io as sci
from tqdm import tqdm
from memory_profiler import memory_usage, profile
import gc
np.set_printoptions(threshold=sys.maxsize)

# initialize matlab
os.environ['HYDRA_FULL_ERROR'] = '1'

def load_data(path, file_name = None):
    # load data file with the matlab engine and unpack data
    if file_name != None:
        mat = sci.loadmat(f'{path}/{file_name}')
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
    
def process(args, file_name):
    omegaconf.OmegaConf.set_struct(args, False)
    data_path = hydra.utils.to_absolute_path(args.data_path)
    
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
                features["time"] = "sequence-based"
        
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
    return np.asarray(samples_PC)
    
@hydra.main(config_path="conf", config_name="paramsweep", version_base='1.3')
def main(args):
    omegaconf.OmegaConf.set_struct(args, False)
    data_path = hydra.utils.to_absolute_path(args.data_path)
    
    logger = logging.getLogger(__name__)
    logger.info(args)
    PC_names = []
    os.mkdir("PC/")
    for i, file in tqdm(enumerate(os.listdir(data_path)), total=len(os.listdir(data_path)), smoothing=0.9):
        if file.endswith(".mat"):
            samples_PC = process(args, file)
            for j, PC in enumerate(samples_PC):
                if isinstance(PC, PointCloud.PointCloud):
                    np.save(f"PC/PC_cls_{PC.mean_label}_{i}_{j}", PC.data)
                    PC_names.append(f"PC_cls_{PC.mean_label}_{i}_{j}")
                else:
                    PC_node_names = []
                    for k, node in enumerate(PC):
                        np.save(f"PC/PC_cls_{node.mean_label}_{i}_{j}_{k}", node.data)
                        PC_node_names.append(f"PC_cls_{node.mean_label}_{i}_{j}_{k}")
                    PC_names.append(PC_node_names)

    PT_args = load_PT_config(args.PT_config_path)
    PC_path = f"{os.getcwd()}/PC/"
    TEST_PC, model = train_cls.main([PT_args, PC_names, PC_path])
    logger.info("Testing on dataset...")
    F1_scores, acc, balanced_acc = point_transformer.test(PT_args, model, args.fusion, TEST_PC, PC_path)
    
    if args.fusion != "none":
        logger.info(f"Fused: F1 score: {F1_scores}, accuracy: {acc}, balanced accuracy: {balanced_acc}")
    else: 
        logger.info("\n".join([f"Node {i}: F1 score: {F1_scores[i]}, accuracy: {acc[i]}, balanced accuracy: {balanced_acc[i]}" for i in range(len(F1_scores))]))
                

if __name__ == '__main__':
    main()