import numpy as np
import matplotlib.pyplot as plt
class PointCloud:
    def __init__(self, data):
        self.data = data
        self.PRF = 122

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
        plt.show()