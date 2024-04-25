import numpy as np
import os, sys, hydra, omegaconf, yaml, argparse, logging, gc
from AdaptiveHAC.pointTransformer import train_cls, point_transformer
from AdaptiveHAC.lib import timing_decorator, load_data, process
from AdaptiveHAC.segmentation import segmentation
from AdaptiveHAC.processing import PC_processing
import scipy.io as sci
from tqdm import tqdm
from memory_profiler import memory_usage
import multiprocessing as mp
np.set_printoptions(threshold=sys.maxsize)

# initialize matlab
os.environ['HYDRA_FULL_ERROR'] = '1'
            
def load_PT_config(PT_config_path):
    with open(f'{PT_config_path}/cls.yaml', 'r') as file:
        dict_args =  yaml.safe_load(file)
        
    if 'defaults' in dict_args:
        for default in dict_args['defaults']:
            if isinstance(default, dict):
                for key, value in default.items():
                    with open(f'{PT_config_path}/{key}/{value}.yaml', 'r') as file:
                        ref_config = yaml.safe_load(file)
                    dict_args[key] = argparse.Namespace(**ref_config)
                dict_args.pop('defaults', None)
                
    dict_args = argparse.Namespace(**dict_args)
    dict_args.experiment_folder = './'
    return dict_args
    
def log_memory_usage(logger):
    """Logs the current memory usage."""
    mem_usage = memory_usage(-1, interval=0.1, timeout=1)  # Get current memory usage
    logger.info(f"Current memory usage: {mem_usage[0]} MiB")
logger = logging.getLogger(__name__)    

def process_wrapper(args_file):
    args, file = args_file
    return process.process(args, file)
        
@hydra.main(config_path="conf", config_name="paramsweep", version_base='1.3')
def main(cfg):
    mp.set_start_method('spawn')
    omegaconf.OmegaConf.set_struct(cfg, False)
    data_path = hydra.utils.to_absolute_path(cfg.data_path)
    cfg.PT_config_path = hydra.utils.to_absolute_path(cfg.PT_config_path)
    cfg.root = hydra.utils.to_absolute_path(cfg.root)
    cfg.data_path = data_path
    logger.info(cfg)
    PC_dataset = []
    try:
        files = [file for file in os.listdir(data_path) if file.endswith(".mat")]
        total_files = len(files)
        files_with_args = [(cfg, file) for file in files]
        for i in tqdm(range(0, total_files-1, 2), total=total_files):
            for result in mp.Pool(processes=mp.cpu_count()).imap(process_wrapper, [files_with_args[i]]):
                    PC_dataset.extend(result)
              
        PT_args = load_PT_config(cfg.PT_config_path)
        TEST_PC, model = train_cls.main([PT_args, PC_dataset])
        logger.info("Testing on dataset...")
        F1_scores, acc, balanced_acc = point_transformer.test(PT_args, model, cfg.fusion, TEST_PC)
        
        if cfg.fusion != "none":
            logger.info(f"Fused: F1 score: {F1_scores}, accuracy: {acc}, balanced accuracy: {balanced_acc}")
        else: 
            logger.info("\n".join([f"Node {i}: F1 score: {F1_scores[i]}, accuracy: {acc[i]}, balanced accuracy: {balanced_acc[i]}" for i in range(len(F1_scores))]))
        
                
    except Exception as error:
        logger.exception(error)

if __name__ == '__main__':
    main()
