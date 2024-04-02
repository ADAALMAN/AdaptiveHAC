import numpy as np

def perfFuncLin(trans_test, trans_GT, window_size):
    #perfFuncLin Use linear scoring to evaluate the quality of found transitions
    #   [score, used_trans] = perfFuncLin(x,gt) returns a scoring for the similarity between a
    #   sequence x and a ground truth gt. Both x and gt should be sorted 1D
    #   vectors containing time stamps.

    #   Author: Nicolas Kruse

    ## init
    dist_matrix = np.zeros((trans_test.shape[1], trans_GT.shape[1]))
    used_trans = [] 
    unused_trans = trans_test
    score = 0

    ## computation
    for i in range(trans_GT.shape[1]):
        dist_matrix[:,i] = np.abs(trans_GT[i]-trans_test)

    while ~isempty(dist_matrix):
        delay = min(min(dist_matrix))
        [row, col] = np.find(dist_matrix==delay , 1 )
        used_trans = [used_trans, unused_trans(row)]
        unused_trans[row] = []
        dist_matrix[row,:] = [] 
        dist_matrix[:,col] = []
        if delay < window_size:
            score = score + 1 #-delay./window_size #remove comment to go to non-trapezoid
        elif delay >= window_size && delay < 1.5*window_size: #trapezoid with width (window_size) and falling slope of length (window_size/2)
            score = score + 1-(delay-window_size)./(window_size/2)
    
    score = score/max([len(trans_test),len(trans_GT)])