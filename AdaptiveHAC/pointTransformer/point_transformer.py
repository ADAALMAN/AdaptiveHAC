import torch
import os
import importlib
import numpy as np
from tqdm import tqdm
from AdaptiveHAC.processing.PointCloud import PointCloud
from AdaptiveHAC.pointTransformer.dataset import ModelNetDataLoader, PCModelNetDataLoader
import logging
from scipy import stats
from sklearn import metrics
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path=os.getcwd()
logger = logging.getLogger(__name__)

def test(args, model, fusion, TEST_PC):
    
    if isinstance(model, str):
        # import model from directory
        model = torch.load(model)
        classifier = getattr(importlib.import_module('pointTransformer.models.{}.model'.format(args.model.name)), 'PointTransformerCls')(args)
        if torch.cuda.device_count() > 1:
            classifier = torch.nn.DataParallel(classifier)
        classifier = classifier.to(device)
        classifier.load_state_dict(model['model_state_dict'])
        classifier = classifier.eval()
    else:
        classifier = model.eval()
    
    y_pred_all = []
    y_true_all = []
    # classify segments
    
    TEST_PC_ALL = []
    # for bigger batches -> faster processing
    if isinstance(TEST_PC[0], PointCloud):
        TEST_PC_ALL = TEST_PC # single node
    elif isinstance(TEST_PC[0][0], PointCloud):
        for nodes in TEST_PC:
            TEST_PC_ALL.extend(nodes)
        
    TEST_DATASET = PCModelNetDataLoader(PC=TEST_PC_ALL, npoint=args.num_point)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # classify nodes
    for j, data in tqdm(enumerate(testDataLoader, 0), total=len(testDataLoader), smoothing=0.9):
        points, target = data
        target = target[:, 0]
        points, target = points.to(device), target.to(device)

        pred = classifier(points)
    
    for dataset in TEST_PC:
        if isinstance(dataset, PointCloud):
            activities = dataset.activities
            y_true = dataset.mean_label
        elif isinstance(dataset[0], PointCloud):
            activities = dataset[0].activities
            y_true = dataset[0].mean_label
            nr_nodes = len(dataset)
        y_true_all.append(y_true)
    
    # turn back into TEST_PC format for fusion
    pred_all = [pred[i:i+nr_nodes] for i in range(int(len(pred)/nr_nodes))]     
    for pred in pred_all:       
        match fusion:
            case "none":
                pred_choice = pred.data.max(1)[1]
                pred_choice = np.asarray(pred_choice.cpu())[:, np.newaxis].T
                pred_cls = stats.mode(pred_choice)
                y_pred_all.extend(pred_choice)
            case "softmax":
                pred_choice = pred.sum(dim=0)
                pred_choice = pred_choice.data.max(0)[1]
                pred_choice = np.asarray(pred_choice.cpu()).T
                y_pred_all.append(pred_choice)
         
    y_pred_all = np.asarray(y_pred_all)
    if y_pred_all.ndim == 1:
        y_pred_all = y_pred_all[:,np.newaxis]
    y_true_all = np.asarray(y_true_all)[:,np.newaxis]
    np.save(os.path.join(args.experiment_folder +'test_pred.npy'), y_pred_all)
    np.save(os.path.join(args.experiment_folder +'test_true.npy'), y_true_all)
    
    F1_scores = []
    acc = []
    balanced_acc = []
    for i in range(y_pred_all.shape[1]):
        F1_scores.append(metrics.f1_score(y_true=y_true_all, y_pred=y_pred_all[:,i][:,np.newaxis], average="macro"))
        acc.append(metrics.accuracy_score(y_true=y_true_all, y_pred=y_pred_all[:,i][:,np.newaxis], normalize=True))
        balanced_acc.append(metrics.accuracy_score(y_true=y_true_all, y_pred=y_pred_all[:,i][:,np.newaxis]))
        plt.close("all")
        metrics.ConfusionMatrixDisplay.from_predictions(y_true=y_true_all, y_pred=y_pred_all[:,i][:,np.newaxis], 
                                                labels=np.arange(1, len(activities)-1), xticks_rotation=90, display_labels=activities[1:-1], 
                                                cmap=plt.cm.Blues)
        plt.title("Fused" if fusion != "none" else f"Node: {i}")
        plt.savefig("conf_matrix_fused.jpg" if fusion != "none" else f"conf_matrix_node_{i}.jpg", bbox_inches='tight')
    return F1_scores, acc, balanced_acc
