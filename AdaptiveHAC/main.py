from AdaptiveHAC.processing import PointCloud
import numpy as np
import os, hydra, omegaconf, yaml, argparse, logging
from AdaptiveHAC.pointTransformer import train_cls, point_transformer
from AdaptiveHAC.lib import process
from tqdm import tqdm
import multiprocessing as mp
import pickle
import matplotlib.pyplot as plt

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
    
logger = logging.getLogger(__name__)    

def process_wrapper(args_file):
    args, file = args_file
    return process.process(args, file)
        
@hydra.main(config_path="conf", config_name="paramsweep", version_base='1.3')
def main(cfg):
    os.mkdir(os.path.join("./entropy"))
    omegaconf.OmegaConf.set_struct(cfg, False)
    data_path = hydra.utils.to_absolute_path(cfg.data_path)
    cfg.PT_config_path = hydra.utils.to_absolute_path(cfg.PT_config_path)
    cfg.root = hydra.utils.to_absolute_path(cfg.root)
    cfg.data_path = data_path
    logger.info(cfg)
    PC_dataset = []
    try:
        # process all sequences to generate pointclouds
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        files = [file for file in os.listdir(data_path) if file.endswith(".mat")]
        total_files = len(files)
        files_with_args = [(cfg, file) for file in files]
        batch_size = mp.cpu_count()-4
        logger.info(f"Total files: {total_files}, processing batch size: {batch_size}")
        for i in tqdm(range(0, total_files, batch_size), total=int(np.ceil(total_files/batch_size))):
            with mp.Pool(processes=mp.cpu_count()-2) as pool:
                for result in pool.imap(process_wrapper, files_with_args[i:i+batch_size]):
                    PC_dataset.extend(result)
        
        # save processed data
        with open('Processed_data.pkl', 'wb') as file:
            pickle.dump(PC_dataset, file)
            
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
    except KeyboardInterrupt:
    	return

if __name__ == '__main__':
    plt.close("all")
    main()
    """ i="SegCustomLossWith10Test"
    cwd = f"./test/paramsweep/2024-05-02/19-23-41/{i}"
    args = load_PT_config('./pointTransformer/config')
    model = f"{cwd}/best_model.pth"
    fusion = 'softmax'
    with open(f'{cwd}/TEST_PC.pkl', 'rb') as file:
            TEST_PC = pickle.load(file)
    args.input_dim = TEST_PC[0][0].data.shape[1]
    args.experiment_folder = cwd
    F1_scores, acc, balanced_acc = point_transformer.test(args, model, fusion, TEST_PC)
    print(F1_scores, acc, balanced_acc) """
