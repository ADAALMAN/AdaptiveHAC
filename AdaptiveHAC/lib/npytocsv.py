import pandas as pd 
import numpy as np 

for i in range(1):
    file = 'spectogram'
     
    # create a dummy array 
    arr1 = np.load(f'./{file}.npy')

    #arr = normalise(arr)  
    # convert array into dataframe 
    DF1 = pd.DataFrame(arr1) 

    # save the dataframe as a csv file 
    DF1.to_csv(f'./{file}.csv', index=False)
