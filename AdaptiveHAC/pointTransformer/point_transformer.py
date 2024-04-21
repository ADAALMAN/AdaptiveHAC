import torch
import os
import importlib
import numpy as np
from tqdm import tqdm
from AdaptiveHAC.pointTransformer import provider
from AdaptiveHAC.processing.PointCloud import PointCloud
from AdaptiveHAC.pointTransformer.dataset import ModelNetDataLoader, PCModelNetDataLoader
import logging
from scipy import stats

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path=os.getcwd()

def test(args, model, fusion, TEST_PC):
    
    if isinstance(model, str):
        # import model from directory
        model = torch.load(model)
        classifier = getattr(importlib.import_module('pointTransformer.models.{}.model'.format(args.model.name)), 'PointTransformerCls')(args)
        if torch.cuda.device_count() > 1:
            classifier = torch.nn.DataParallel(classifier)
        classifier = classifier.to(device)
        classifier.load_state_dict(model)
        classifier = classifier.eval()
    else:
        classifier = model.eval()
    
    pred_all = []
    true_all = []
    # classify segments
    for dataset in TEST_PC:
            
        TEST_DATASET = PCModelNetDataLoader(PC=dataset, npoint=args.num_point)
        testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
        nodes_pred = []
        class_acc = np.zeros((len(dataset[0].activities),3))
        
        # classify nodes
        for j, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
            points, target = data
            target = target[:, 0]
            points, target = points.to(device), target.to(device)

            pred = classifier(points)
            nodes_pred.extend(pred, axis=1)
            
        true = dataset[0].mean_label
            
        match fusion:
            case "none":
                pred_choice = pred.data.max(1)[1]
                pred_cls = stats.mode(pred_choice.cpu())[0]
            case "softmax":
                
                pred_choice = pred.data.max(1)[1]
                pred_cls = stats.mode(pred_choice.cpu())[0]
        
    return