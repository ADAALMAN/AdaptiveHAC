import pandas as pd 
import numpy as np 

for i in range(15):
    file_true = 'test_pred'
    file_pred = 'test_true'     
    # create a dummy array 
    arr1 = np.load(f'{i}/{file_true}.npy')
    arr2 = np.load(f'{i}/{file_pred}.npy')
    #arr = normalise(arr)  
    # convert array into dataframe 
    DF1 = pd.DataFrame(arr1) 
    DF2 = pd.DataFrame(arr2) 
    # save the dataframe as a csv file 
    DF1.to_csv(f'{i}/{file_true}.csv', index=False)
    DF2.to_csv(f'{i}/{file_pred}.csv', index=False)