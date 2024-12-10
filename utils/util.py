import os
import random
import torch
import numpy as np
import transformers
from pathlib import Path
from input_data.data_loader import ME2Data
from scipy.stats import norm
import json
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn import metrics
from typing import Tuple, List

from input_data.data_loader import ME2Data

import numpy as np


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    transformers.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True





def predict(model: torch.nn.Module,
            dataset: ME2Data,
            device: str,
            batch_size: int = 64,
            domain_adapt: bool = True,
            is_adv: bool = True, threeshold=0.5, no_gate=False, weights=None) -> Tuple[List, List, List]:
    model.eval()
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=False)
    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0

    mf_preds = []
    mf_preds_conf = []
    domain_preds = []

    while i < len_dataloader:
        data_target, _ = next(data_target_iter)
        for k, v in data_target.items():
            data_target[k] = data_target[k].to(device)

        with torch.no_grad():
            if not domain_adapt:
                outputs = model(data_target['input_ids'],
                                data_target['attention_mask'])
            else:
                if 'emotion_labels' in data_target:
                    emo_labels = data_target['emotion_labels'] 
                    outputs = model(data_target['input_ids'],
                    data_target['attention_mask'],
                    data_target['domain_labels'],
                    lambda_domain=0,
                    adv=is_adv, emotion_labels=emo_labels, no_gate=no_gate) # emo_labels
                else:
                    emo_labels = torch.zeros((batch_size, 5)).to(device)
                    if weights is not None:
                        for ii in range(len(weights)):
                        
                            emo_labels[:, ii] = weights[ii]
    

                    outputs = model(data_target['input_ids'],
                    data_target['attention_mask'],
                    data_target['domain_labels'],
                    lambda_domain=0,
                    adv=is_adv, emotion_weights=emo_labels, no_gate=no_gate )
                
                

            if outputs['class_output'].shape[1]>=10:
                outputs['class_output'] = (outputs['class_output'][:, :5] + outputs['class_output'][:, 5:-1])/2
                    
        
        mf_pred_confidence = torch.sigmoid(outputs['class_output'])
        mf_preds_conf.extend(mf_pred_confidence.to('cpu').tolist())
        mf_pred = ((mf_pred_confidence) >= threeshold).long()
        mf_preds.extend(mf_pred.to('cpu').tolist())

        if domain_adapt and is_adv:
            domain_pred = outputs['domain_output'].data.max(1, keepdim=True)[1]
            domain_preds.extend(domain_pred.to('cpu').tolist())

        i += 1

    return mf_preds, mf_preds_conf, domain_preds


def evaluate(dataset_s: ME2Data,
             dataset_t: ME2Data = None,
             batch_size: int = 64,
             model: torch.nn.Module = None,
             model_path: str = None,
             is_adv: bool = True,
             test: bool = False, device: int = 0, domain_adapt: bool = True, threeshold=0.5, no_gate=False, weights=None, return_mf_preds=False, extend_non_moral=False) -> float:
    """
    Evalute test data and print F1 scores.

    Args:
        dataset: test data, an MFData instance
        batch_size: default is 64
        model: MFBasic or MFDomainAdapt instance, either model or model path should be given
        model_path: if no model instance is given, will load model from this path
        is_adv: if doing adv training, will pass it to model forward fn and predict domain labels
        test: whether in training or test mode

    Returns:
        f1 score
        also print detailed classification report to log file.
    """
    device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
    if dataset_t is None:
        dataset = dataset_s
    else:
        dataset = ME2Data(np.concatenate((dataset_s.encodings, dataset_t.encodings)),
                          np.concatenate((dataset_s.mf_labels, dataset_t.mf_labels)), 
                          np.concatenate((dataset_s.domain_labels, dataset_t.domain_labels)), np.concatenate((dataset_s.emotion_labels, dataset_t.emotion_labels))) 
    assert (model is not None
            or model_path is not None), 'Provide a model instance or a model path.'
    if model is None:
        model = torch.load(model_path, map_location=torch.device(device))


    model = model.to(device)


    mf_preds, _, domain_preds = predict(model, dataset, device, batch_size,
                                        domain_adapt, is_adv, threeshold=threeshold, no_gate=no_gate, weights=weights)
    
    
    if dataset.mf_labels.shape[1]>=10:
        dataset.mf_labels = (dataset.mf_labels[:, :5] + dataset.mf_labels[:, 5:])/2
       
    bin_labels = (dataset.mf_labels >= threeshold).astype(float)#[:800]
    # print reports
    if isinstance(mf_preds, list):
        mf_preds = torch.tensor(mf_preds, dtype=torch.float32)
    
    if return_mf_preds:
        return mf_preds
    if extend_non_moral:
        new_class_labels = np.where(np.all(bin_labels == 0, axis=1), 1, 0)
        new_class_preds = np.where(np.all(mf_preds.numpy() == 0, axis=1), 1, 0)

        # Aggiungi questa nuova etichetta a bin_labels e mf_preds
        bin_labels = np.hstack((bin_labels, new_class_labels[:, np.newaxis]))
        mf_preds = np.hstack((mf_preds, new_class_preds[:, np.newaxis]))
        
    mf_report = metrics.classification_report(bin_labels,
                                              mf_preds,
                                              zero_division=0,
                                              output_dict=True)
    macro_f1 = mf_report['weighted avg']['f1-score']
    
    return macro_f1, mf_preds, mf_report
    
    
def average_classification_reports(reports: list) -> dict:
    """
    Average classification reports from multiple folds.

    Args:
        reports: list of classification reports (dicts) from each fold

    Returns:
        A single classification report with averaged metrics.
    """
    avg_report = {}
    labels = reports[0].keys()

    for label in labels:
        avg_report[label] = {}
        for metric in reports[0][label].keys():
            avg_report[label][metric] = np.mean([report[label][metric] for report in reports])

    return avg_report


def average_and_std_classification_reports(reports: list) -> dict:
    """
    Average and standard deviation of classification reports from multiple folds.

    Args:
        reports: list of classification reports (dicts) from each fold

    Returns:
        A single classification report with averaged metrics and standard deviations.
    """
    avg_report = {}
    labels = reports[0].keys()

    for label in labels:
        avg_report[label] = {}
        for metric in reports[0][label].keys():
            values = [report[label][metric] for report in reports]
            avg_report[label][metric] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }

    return avg_report


def save_dictionary(dic, file_name):
    with open(file_name, 'w') as file:
        json.dump(dic, file)


def load_dictionary(file_name):
    with open(file_name, 'r') as file: 
        dic = json.load(file)
    return dic