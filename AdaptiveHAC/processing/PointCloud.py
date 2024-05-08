import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class PointCloud:
    def __init__(self, data:np.ndarray, label:np.ndarray):
        self.sequence_name = None
        self.H_score = None
        self.total_time = 1
        self.activities = ["N/A", "Walking", "Stationary", "Sitting down","Standing up (sitting)",
                            "Bending (sitting)","Bending (standing)",
                            "Falling (walking)","Standing up (ground)","Falling (standing)"]
        self.PRF = 122
        self.time = "standard"
        self.features = "none"
        
        self.data = data
        self.label = label
        
        self.segment_length = self.label.size
        self.mean_label = stats.mode(label, axis=1)[0][0]
        self.per_labels = np.asarray([np.sum(self.label == self.activities.index(act))/self.label.size for act in self.activities]) # percentage of label occurance in segment
        self.per_mean_label = self.per_labels[self.mean_label] # percentage of mean label occurance in segment
        
        

    def add_features(self, features, time_feature):
        for feature in features:
            ft = np.full((self.data.shape[0],1), fill_value=feature)
            self.data = np.append(self.data, ft, axis=1)
        if time_feature != None:
            self.time = time_feature[0]
            self.total_time = time_feature[1]
            
    def normalise(self):
        temp_data = np.copy(self.data)
        temp_data[:, 0] = (temp_data[:, 0] - np.mean(temp_data[:, 0])) / 480                         # range
        temp_data[:, 1] = temp_data[:, 1] / self.PRF                                                 # doppler
        if self.time == "standard":
            temp_data[:, 2] = (temp_data[:, 2] - np.mean(temp_data[:, 2])) / np.std(temp_data[:, 2]) # time
        elif self.time == "sequence-based":
            temp_data[:, 2] = temp_data[:, 2]/self.total_time
        temp_data[:, 3] = (temp_data[:, 3] - np.mean(temp_data[:, 3])) / np.std(temp_data[:, 3])     # power
        temp_data[:, 4] = temp_data[:, 4] / 5                                                        # node
        if "entropy" in self.features and "PBC" in self.features:
            temp_data[:, 5] = temp_data[:, 5]/10
            temp_data[:, 6] = temp_data[:, 6]/1e6
        elif "entropy" in self.features and "PBC" not in self.features:
            temp_data[:, 5] = temp_data[:, 5]/10
        elif "entropy" not in self.features and "PBC" in self.features:
            temp_data[:, 5] = temp_data[:, 5]/1e6
        self.data = temp_data 
      
    def visualise(self):   
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(self.data[:,0], self.data[:,1], self.data[:,2])
        ax.set_xlabel('Range')
        ax.set_ylabel('Frequency')
        ax.set_zlabel('Time')
        ax.set_title(self.activities[self.mean_label])
        plt.show()
    
    def add_attributes(self, name, H_score):
        self.sequence_name = name
        self.H_score = H_score