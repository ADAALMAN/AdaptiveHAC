import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
class PointCloud:
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.mean_label = stats.mode(label, axis=1)[0][0]
        self.PRF = 122
        self.time = "normal"
        self.activities = ["N/A", "Walking", "Stationary", "Sitting down","Standing up (sitting)",
                            "Bending (sitting)","Bending (standing)",
                            "Falling (walking)","Standing up (ground)","Falling (standing)"]

    def add_features(self, features):
        for feature in features:
            ft = np.full((1,self.data.shape[1]), fill_value=feature)
            self.data.append(ft, axis=1)
            
    def normalise(self):
        (self.data[:, 0] - np.mean(self.data[:, 0])) / 480                         # range
        self.data[:, 1] / self.PRF                                                 # doppler
        if self.time == "normal":
            (self.data[:, 2] - np.mean(self.data[:, 2])) / np.std(self.data[:, 2])     # time
        elif self.time == "sequence-based":
            pass
        (self.data[:, 3] - np.mean(self.data[:, 3])) / np.std(self.data[:, 3])     # power
        self.data[:, 4] / 5                                                        # node
        return self    
      
    def visualise(self):   
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(self.data[:,0], self.data[:,1], self.data[:,2])
        ax.set_xlabel('Range')
        ax.set_ylabel('Frequency')
        ax.set_zlabel('Time')
        ax.set_title(self.activities[self.mean_label])
        plt.show()