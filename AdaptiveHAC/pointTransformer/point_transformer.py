import pickle
import torch
import os
import importlib
import numpy as np
from tqdm import tqdm
from AdaptiveHAC.processing.PointCloud import PointCloud
from AdaptiveHAC.pointTransformer.dataset import PCModelNetDataLoader
import logging
from scipy import stats
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path=os.getcwd()
logger = logging.getLogger(__name__)

def test(args, model, args_fusion, TEST_PC):
    with torch.no_grad():
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
        pred = []
        true = []
        sequence_name_temp = []
        H_score_temp = []
        per_labels_temp = []
        per_mean_label_temp = []
        seg_length_temp = []
        for j, data in tqdm(enumerate(testDataLoader, 0), total=len(testDataLoader), smoothing=0.9):
            points, target, sequence_name, H_score, per_labels, per_mean_label, seg_length = data
            target = target[:, 0]
            points, target = points.to(device), target.to(device)
            true.extend(target)
            pred.extend(classifier(points))
            sequence_name_temp.extend(sequence_name)
            H_score_temp.extend(H_score)
            per_labels_temp.extend(per_labels)
            per_mean_label_temp.extend(per_mean_label)
            seg_length_temp.extend(seg_length)
        
        for dataset in TEST_PC:
            if isinstance(dataset, PointCloud):
                activities = dataset.activities
                nr_nodes = 1
            elif isinstance(dataset[0], PointCloud):
                activities = dataset[0].activities
                nr_nodes = len(dataset)
        
        # turn back into TEST_PC format for fusion
        pred_all            = [pred[i*(nr_nodes):(i+1)*(nr_nodes)] for i in range(int(len(pred)/nr_nodes))]
        true_all            = [true[i*(nr_nodes):(i+1)*(nr_nodes)] for i in range(int(len(true)/nr_nodes))]
        sequence_name_all   = [sequence_name_temp[i*(nr_nodes):(i+1)*(nr_nodes)] for i in range(int(len(true)/nr_nodes))]  
        H_score_all         = [H_score_temp[i*(nr_nodes):(i+1)*(nr_nodes)] for i in range(int(len(true)/nr_nodes))]  
        per_labels_all      = [per_labels_temp[i*(nr_nodes):(i+1)*(nr_nodes)] for i in range(int(len(true)/nr_nodes))]  
        per_mean_label_all  = [per_mean_label_temp[i*(nr_nodes):(i+1)*(nr_nodes)] for i in range(int(len(true)/nr_nodes))]  
        seg_length_all      = [seg_length_temp[i*(nr_nodes):(i+1)*(nr_nodes)] for i in range(int(len(true)/nr_nodes))]  
        
        # do all fusion
        scores = []
        for fusion in ["none", "softmax", "majvote"]:
            y_pred_all = []
            y_true_all = []
            os.mkdir(os.path.join(args.experiment_folder + "/" + fusion))
            for y_pred, y_true in zip(pred_all, true_all):       
                match fusion:
                    case "none":
                        pred_choice = torch.stack(y_pred,dim=0).data.max(1)[1]
                        pred_choice = np.asarray(pred_choice.cpu())[:, np.newaxis].T  
                        y_pred_all.extend(pred_choice)  
                    case "softmax":
                        pred_choice = torch.stack(y_pred,dim=0).sum(dim=0)
                        pred_choice = pred_choice.data.max(0)[1]
                        pred_choice = np.asarray(pred_choice.cpu()).T
                        y_pred_all.append(pred_choice)
                    case "majvote":
                        pred_choice = torch.stack(y_pred,dim=0).mode(dim=0)
                        pred_choice = pred_choice[0].max(0)[1]
                        pred_choice = np.asarray(pred_choice.cpu()).T
                        y_pred_all.append(pred_choice)   
                y_true_all.append(y_true[0].cpu())
                        
            y_pred_all = np.asarray(y_pred_all)
            if y_pred_all.ndim == 1:
                y_pred_all = y_pred_all[:,np.newaxis]
            y_true_all = np.asarray(y_true_all)[:,np.newaxis]
            
            F1_scores = []
            acc = []
            balanced_acc = []
            for i in range(y_pred_all.shape[1]):
                F1_scores.append(metrics.f1_score(y_true=y_true_all, y_pred=y_pred_all[:,i][:,np.newaxis], average="macro"))
                acc.append(metrics.accuracy_score(y_true=y_true_all, y_pred=y_pred_all[:,i][:,np.newaxis], normalize=True))
                balanced_acc.append(metrics.balanced_accuracy_score(y_true=y_true_all, y_pred=y_pred_all[:,i][:,np.newaxis]))
                plt.close("all")
                metrics.ConfusionMatrixDisplay.from_predictions(y_true=y_true_all, y_pred=y_pred_all[:,i][:,np.newaxis], 
                                                        labels=np.arange(1, len(activities)-1), xticks_rotation=90, display_labels=activities[1:-1], 
                                                        cmap=plt.cm.Blues)
                plt.title("Fused" if fusion != "none" else f"Node: {i}")
                plt.savefig(os.path.join(args.experiment_folder + "/" + fusion + "/conf_matrix_fused.jpg") if fusion != "none" 
                            else os.path.join(args.experiment_folder + "/" + fusion + f"/conf_matrix_node_{i}.jpg"), bbox_inches='tight')
            
            scores.append([F1_scores, acc, balanced_acc])
                
            # save data    
            np.save(os.path.join(args.experiment_folder + "/" + fusion + '/test_pred.npy'), y_pred_all)
            np.save(os.path.join(args.experiment_folder + "/" + fusion + '/test_true.npy'), y_true_all)
            np.save(os.path.join(args.experiment_folder + "/" + fusion + '/sequence_names.npy'), np.asarray(sequence_name_all)[:,np.newaxis])
            np.save(os.path.join(args.experiment_folder + "/" + fusion + '/H_scores.npy'), np.asarray(H_score_all)[:,np.newaxis])
            np.save(os.path.join(args.experiment_folder + "/" + fusion + '/per_labels.npy'), np.asarray(per_labels_all))
            np.save(os.path.join(args.experiment_folder + "/" + fusion + '/per_mean_label.npy'), np.asarray(per_mean_label_all)[:,np.newaxis])
            np.save(os.path.join(args.experiment_folder + "/" + fusion + '/seg_length.npy'), np.asarray(seg_length_all)[:,np.newaxis])
            pd.DataFrame(y_pred_all).to_csv(os.path.join(args.experiment_folder + "/" + fusion + '/test_pred.csv'), index=False)
            pd.DataFrame(y_true_all).to_csv(os.path.join(args.experiment_folder + "/" + fusion + '/test_true.csv'), index=False)
            pd.DataFrame(np.asarray(sequence_name_all)[:,np.newaxis][:,0]).to_csv(os.path.join(args.experiment_folder + "/" + fusion + '/sequence_names.csv'), index=False)
            pd.DataFrame(np.asarray(H_score_all)[:,np.newaxis][:,0]).to_csv(os.path.join(args.experiment_folder + "/" + fusion + '/H_scores.csv'), index=False)
            pd.DataFrame(np.asarray(per_labels_all)[:,:,0]).to_csv(os.path.join(args.experiment_folder + "/" + fusion + '/per_labels.csv'), index=False)
            pd.DataFrame(np.asarray(per_mean_label_all)[:,np.newaxis][:,0]).to_csv(os.path.join(args.experiment_folder + "/" + fusion + '/per_mean_label.csv'), index=False)
            pd.DataFrame(np.asarray(seg_length_all)[:,np.newaxis][:,0]).to_csv(os.path.join(args.experiment_folder + "/" + fusion + '/seg_length.csv'), index=False)
            np.savetxt(os.path.join(args.experiment_folder + "/" + fusion + '/F1.txt'), np.asarray(F1_scores), fmt='%1.5f', delimiter=',', newline='\n')
            np.savetxt(os.path.join(args.experiment_folder + "/" + fusion + '/Accuracy.txt'), np.asarray(acc), fmt='%1.5f', delimiter=',', newline='\n')
            np.savetxt(os.path.join(args.experiment_folder + "/" + fusion + '/Balanced_accuracy.txt'), np.asarray(balanced_acc), fmt='%1.5f', delimiter=',', newline='\n')
    torch.cuda.empty_cache()
    
    if args_fusion == "none":
        F1_scores, acc, balanced_acc = scores[0] # unpack no fusion
    elif args_fusion == "softmax":
        F1_scores, acc, balanced_acc = scores[1] # unpack softmax fusion
    elif args_fusion == "majvote":
        F1_scores, acc, balanced_acc = scores[2] # unpack majority vote fusion    
    return F1_scores, acc, balanced_acc

    