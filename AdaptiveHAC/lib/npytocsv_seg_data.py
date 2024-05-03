import pandas as pd 
import numpy as np 

i=8
files = ['sequence_names', 'H_scores', 'per_labels', 'per_mean_label']
# create a dummy array 
for file in files:
    arr1 = np.load(f'{i}/{file}.npy')
    DF1 = pd.DataFrame(arr1[:,0]) 
    DF1.to_csv(f'{i}/{file}.csv', index=False)