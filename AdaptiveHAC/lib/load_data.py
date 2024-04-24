import scipy.io as sci
import numpy as np

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