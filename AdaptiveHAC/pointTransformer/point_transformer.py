import torch
import os
import importlib
from tqdm import tqdm
from AdaptiveHAC.pointTransformer import provider
from AdaptiveHAC.processing.PointCloud import PointCloud
from AdaptiveHAC.pointTransformer.dataset import ModelNetDataLoader, PCModelNetDataLoader
import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path=os.getcwd()

def test(args, model, TEST_PC):
    
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
        
    TEST_DATASET = PCModelNetDataLoader(PC=TEST_PC, npoint=args.num_point)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    y_pred = torch.empty(0,dtype=torch.long).to(device)
    y_true = torch.empty(0,dtype=torch.long).to(device)
    idx = torch.empty(0,dtype=torch.long).to(device)
    for batch_id, (input, targets, sample_idx) in enumerate(testDataLoader):
        input, targets = input.float().requires_grad_().to(device), torch.squeeze(targets.long(),dim = 2).to(device)
        out = classifier(input)
        y_pred = torch.cat((y_pred,torch.argmax(out,2).flatten()))
        y_true = torch.cat((y_true,targets.flatten()))
        idx = torch.cat((idx,sample_idx.to(device).flatten()))
        
    return y_true.cpu().numpy(), y_pred.cpu().numpy(), idx.cpu().numpy()