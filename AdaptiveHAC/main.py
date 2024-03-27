import matlab.engine
import numpy as np
import os, sys
#from AdaptiveHAC.pointTransformer import train_cls
from AdaptiveHAC.lib import timing_decorator
from AdaptiveHAC.segmentation import segmentation
from AdaptiveHAC.processing import PC_processing
from memory_profiler import profile
import scipy.io as sci

np.set_printoptions(threshold=sys.maxsize)

# initialize matlab
os.environ['HYDRA_FULL_ERROR'] = '1'
eng = matlab.engine.start_matlab()

@timing_decorator.timing_decorator
def load_data(path, file_name = None):
    # load data file with the matlab engine and unpack data
    if file_name != None:
        mat = eng.load(f'{path}{file_name}.mat')
        data = np.asarray(mat['hil_resha_aligned'], dtype=np.complex64) # reduce data in memory
        lbl = np.asarray(mat['lbl_out'])

        # resize the labels to the data length
        if lbl.shape[1] > data.shape[1]:
            lbl = np.array(lbl)[:,:data.shape[1]]

    return data, lbl

@profile
def main(sample_method="segmentation", node_method="all"):
    # data path
    #path = 'W:/staff-groups/ewi/me/MS3/MS3-Shared/Ronny_MonostaticData/Nicolas/MAT_data_aligned/'
    path = './test/data/'
    file_name = '029_mon_Mix_Nic'

    match node_method:
        case "all":
            data, lbl = load_data(path, file_name)
        case "individual":
            i = 0
            data, lbl = load_data(path, file_name)
            data = data[:,:, i]
    
    match sample_method:
        case "window":
            samples, labels = PC_processing.sample(data, lbl, sample_size = 'default')
        case "segmentation":
            seg_th = 100
            samples, labels = segmentation.segmentation(data, lbl)
            samples, labels = segmentation.segmentation_thresholding(samples, labels, seg_th, "split")
    #del(data, lbl)
    
    chunks= 6
    npoints = 1024
    thr = 0.8
    samples_PC = PC_processing.PC_generation(samples, chunks, npoints, thr)

    """ 
    del(samples)
    os.mkdir(f'./py_test/seg_th_{seg_th}/')
    PC_processing.save_PC(f'./py_test/seg_th_{seg_th}/', samples_PC, labels)

    #train_cls.main()"""

if __name__ == '__main__':
    main(sample_method="segmentation")