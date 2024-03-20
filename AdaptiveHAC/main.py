import matlab.engine
import numpy as np
import os, sys
from AdaptiveHAC.pointTransformer import train_cls
from AdaptiveHAC.lib import timing_decorator
from AdaptiveHAC.segmentation import segmentation
from AdaptiveHAC.processing import PC_processing

# initialize matlab
os.environ['HYDRA_FULL_ERROR'] = '1'
eng = matlab.engine.start_matlab()

@timing_decorator.timing_decorator
def load_data(path, file_name = None):
    # load data file with the matlab engine and unpack data
    if file_name != None:
        mat = eng.load(f'{path}{file_name}.mat')
        data = mat['hil_resha_aligned']
        lbl = mat['lbl_out']
        del(mat) # cleanup

    # resize the labels to the data length
    if lbl.size[1] > data.size[1]:
        lbl = np.array(lbl)[:,:data.size[1]]

    return data, lbl

def main(sample_method="segmentation"):
    # data path
    #path = 'W:/staff-groups/ewi/me/MS3/MS3-Shared/Ronny_MonostaticData/Nicolas/MAT_data_aligned/'
    path = './test/data/'
    file_name = '029_mon_Mix_Nic'

    # paramenters
    chunks = 6
    npoints = 1024
    thr = 0.8

    data, lbl = load_data(path, file_name)
    data = np.asarray(data)
    match sample_method:
        case "window":
            samples, labels = PC_processing.sample(data, lbl, sample_size = 'default')
        case "segmentation":
            samples, labels = segmentation.segmentation(data, lbl)
            samples, labels = segmentation.segmentation_thresholding(samples, labels, 350, "split")
    del(data, lbl)
    
    """ 
    samples_PC = PC_processing.PC_generation(samples, chunks, npoints, thr)
    del(samples)
    save_PC('./py_test/', samples_PC, labels) """

    #train_cls.main()

if __name__ == '__main__':
    main(sample_method="segmentation")