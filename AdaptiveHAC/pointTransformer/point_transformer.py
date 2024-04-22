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
from sklearn.metrics import ConfusionMatrixDisplay

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
        activities = dataset[0].activities
        
        # classify nodes
        for j, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
            points, target = data
            target = target[:, 0]
            points, target = points.to(device), target.to(device)

            pred = classifier(points)
                
        match fusion:
            case "none":
                pred_choice = pred.data.max(1)[1]
                pred_choice = np.asarray(pred_choice.cpu())[:, np.newaxis].T
                pred_cls = stats.mode(pred_choice)
                pred_all.extend(pred_choice)
            case "softmax":
                pred_choice = pred.sum(dim=0)
                pred_choice = pred_choice.data.max(0)[1]
                pred_choice = np.asarray(pred_choice.cpu())[:, np.newaxis].T
                pred_all.extend(pred_choice)
        
        true = dataset[0].mean_label
        true_all.append(true)
        
    pred_all = np.asarray(pred_all)
    true_all = np.asarray(true_all)
    np.save(os.path.join(args.experiment_folder +'test_pred.npy'), pred_all)
    np.save(os.path.join(args.experiment_folder +'test_true.npy'), true_all)
    
    for i in range(pred_all.shape[1]):
        ConfusionMatrixDisplay.from_predictions(true_all[:,np.newaxis], pred_all[:,i][:,np.newaxis], labels=activities)
    return