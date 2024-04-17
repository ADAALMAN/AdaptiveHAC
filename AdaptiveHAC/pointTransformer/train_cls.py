'''
sourcr: https://github.com/POSTECH-CVLab/point-transformer
modified by zhongyuan
'''
from AdaptiveHAC.pointTransformer.dataset import ModelNetDataLoader, PCModelNetDataLoader
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import logging
from tqdm import tqdm
from AdaptiveHAC.pointTransformer import provider
from AdaptiveHAC.processing.PointCloud import PointCloud
import importlib
import shutil
import hydra
import omegaconf
import matplotlib.pyplot as plt
import argparse
from sklearn.model_selection import train_test_split

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path=os.getcwd()
# dataset = 'MMA_xyzI'

def test(model, loader, num_class): #num_class should change !!!
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]
        points, target = points.to(device), target.to(device)
        classifier = model.eval()
        pred = classifier(points)
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc

#@hydra.main(config_path='config', config_name='cls', version_base='1.1')
def main(args):
    if isinstance(args, list):
        PC = args[1]
        args = args[0]
        args.input_dim = PC[0][0].data.shape[1]
    elif isinstance(args, omegaconf.dictconfig.DictConfig):  
        PC = None  
        omegaconf.OmegaConf.set_struct(args, False)
    elif isinstance(args, dict):
        PC = None

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)

    '''DATA LOADING'''
    logger.info('Load dataset ...')
    if PC != None:
        if isinstance(PC[0][0], PointCloud):
            TRAIN_PC, TEST_PC = train_test_split(PC, train_size=0.8, shuffle=True)
            TRAIN_DATASET = PCModelNetDataLoader(PC=TRAIN_PC, npoint=args.num_point)
            TEST_DATASET = PCModelNetDataLoader(PC=TEST_PC, npoint=args.num_point)
    else:
        dataset = args.dataset
        DATA_PATH = hydra.utils.to_absolute_path(dataset)
        TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='train')
        TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test')
        
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    shutil.copy(hydra.utils.to_absolute_path('pointTransformer/models/{}/model.py'.format(args.model.name)), '.')

    classifier = getattr(importlib.import_module('pointTransformer.models.{}.model'.format(args.model.name)), 'PointTransformerCls')(args)
    if torch.cuda.device_count() > 1:
        classifier = torch.nn.DataParallel(classifier)
    classifier = classifier.to(device)
    # print(classifier)
    criterion = torch.nn.CrossEntropyLoss(torch.FloatTensor([0,1,1,1,1,1,1,1,1,1]).to(device))

    try:
        checkpoint = torch.load('best_model.pth')
        start_epoch = checkpoint['epoch']
        
        classifier.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Use pretrain model')
    except:
        logger.info('No existing model, starting training from scratch...')
        start_epoch = 0


    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=float(args.weight_decay)
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.6)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    best_epoch = 0
    mean_correct = []

    '''TRANING'''
    logger.info('Start training...')
    loss_value =[]
    train_accuracy = []
    test_accuracy = []

    for epoch in range(start_epoch,args.epoch):
        logger.info('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        classifier.train()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
            points = torch.Tensor(points)
            target = target[:, 0]

            points, target =  points.to(device), target.to(device)
            optimizer.zero_grad()

            pred = classifier(points)
            
            loss = criterion(pred, target.long())
            loss_value.append(loss.cpu().detach().numpy())
            
            # pred_choice = pred.data.max(1)[1]
            # correct = pred_choice.eq(target.long().data).cpu().sum()
            # mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1
            
        scheduler.step()

        with torch.no_grad():

            instance_train_accuracy, _ = test(classifier.eval(), trainDataLoader, args.num_class)
            train_accuracy.append(instance_train_accuracy)
            logger.info('Train Instance Accuracy: %f' % instance_train_accuracy)
            logger.info('loss:%f'% loss)



            instance_acc, class_acc = test(classifier.eval(), testDataLoader, args.num_class)
            test_accuracy.append(instance_acc)
            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            logger.info('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))
            logger.info('Best Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = 'best_model.pth'
                logger.info('Saving at %s'% savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')
    
    plt.figure(1)
    plt.plot(loss_value)
    plt.xlabel('batches')
    plt.title('Loss')
    plt.savefig(os.path.join(args.experiment_folder +'loss.png'))
    plt.figure(2)
    plt.plot(train_accuracy,label='train accuracy')
    plt.plot(test_accuracy, label = 'test accuracy')
    plt.xlabel('epochs')
    plt.title('accuracy')
    plt.legend()
    plt.savefig(os.path.join(args.experiment_folder +'accuracy.png'))
    logger.info('End of savefig...')
    with open(os.path.join(args.experiment_folder +'loss.txt'),'w') as f:
        for loss in loss_value:
            f.write(str(loss))
    with open(os.path.join(args.experiment_folder +'accuracy.txt'),'w') as f:
        for acc in train_accuracy:
            f.write(str(acc))
            f.write(' ')
        
        f.write('\n')
        for acc in test_accuracy :
            f.write(str(acc))
            f.write(' ')
    logger.info('End of savetxt...')
if __name__ == '__main__':
    main()