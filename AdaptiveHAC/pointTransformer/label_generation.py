
import os
import random




def get_num(filename:str):
    file = filename.split('_')[-1]
    num = int(file.split('.')[0])
    return num

def label_generation(folder,trainset,testset,CV=['','']):
    with open(os.path.join(folder,'train_set.txt'),'w') as f:
        for motions in trainset:
            for motion in motions:
                f.write(CV[0]+motion.split('_')[0]+'/'+motion.split('.')[0])
                f.write('\n')

    with open(os.path.join(folder,'test_set.txt'),'w') as f:
        for motions in testset:
            for motion in motions:
                f.write(CV[1]+motion.split('_')[0]+'/'+motion.split('.')[0])
                f.write('\n')
    
random.seed(10)

Train_rate = 0.8 # Percentage for training
cross_validation = False  # index will be added to the filename
using_name = False # using human subjects' names as labels
norm = True
folder = 'data/'  # dataset folder
names = {'name1':1,'name2':2}
motions =['bfrsit','bfrstand','ffs','ffw','sitdn','stat','stup','stupsit','wlk']  #labels
#motions =['fall','insitu','stat','stup','wlk']  #labels
#motions = ['bfsi', 'stup']
with open(os.path.join(folder,'class_names''.txt'),'w') as f:
        for motion in motions:
            f.write(motion)
            f.write('\n')

#   212424 6224421 4676 4124565 1165 65656
if using_name:
    for name in names:
        trainset = []
        testset  = []
        for motion in motions:
            path = os.path.join(folder,motion)
            files = os.listdir(path)

            for file in files:
                if 60*(names[name]-1)<get_num(file)<=60*(names[name]):
                    testset.append(file)
                else:
                    trainset.append(file)
        with open(os.path.join(folder,'Yubin_train_'+name+'.txt'),'w') as f:
            for motion in trainset:
                f.write(motion.split('.')[0])
                f.write('\n')
        with open(os.path.join(folder,'Yubin_test_'+name+'.txt'),'w') as f:
            for motion in testset:
                f.write(motion.split('.')[0])
                f.write('\n')        
if cross_validation :
        trainset = []
        testset = []
        for motion in motions:
            path = os.path.join(folder,'train',motion)
            files = os.listdir(path)
            random.shuffle(files)
            trainset.append(files)

            path = os.path.join(folder,'test',motion)
            files = os.listdir(path)
            random.shuffle(files)
            testset.append(files)

        label_generation(folder,trainset,testset,['train/','test/'])
if norm:
    trainset = []
    testset = []
    for motion in motions:
        path = os.path.join(folder,motion)

        files = os.listdir(path)
        random.shuffle(files)
                
        train_num = int(len(files)*Train_rate)
        test_num = len(files)-train_num

        testset.append(files[0:test_num])
        del files[0:test_num]
        trainset.append(files)

    label_generation(folder,trainset,testset)
    






# 

        
