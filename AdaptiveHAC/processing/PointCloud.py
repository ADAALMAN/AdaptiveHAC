import numpy as np
import matplotlib.pyplot as plt
class PointCloud:
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.mean_label = int(np.mean(label, axis=1))
        self.PRF = 122
        self.activities = ["N/A", "Walking", "Stationary","Standing up (sitting)",
                            "Bending (sitting)","Bending (standing)","Standing up (ground)",
                            "Falling (walking)","Standing up (ground)","Falling (standing)"]

    def normalise(self):
        (self.data[:, 0] - np.mean(self.data[:, 0])) / 480                         # range
        self.data[:, 1] / self.PRF                                                 # doppler
        (self.data[:, 2] - np.mean(self.data[:, 2])) / np.std(self.data[:, 2])     # time
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