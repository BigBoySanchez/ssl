import os 

from collections import defaultdict
from copy import deepcopy
from scipy.special import softmax
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import time 
from torchmetrics.classification import MulticlassCalibrationError
import torch.nn.functional as F
import torch.nn as nn
from custom_dataset import CustomDataset, CustomDataset_tracked
from aum import AUMCalculator
import pandas as pd

import logging
import math
import numpy as np
import pandas as pd 
import os
import sampler
import torch
import json 
import statistics 
import random
from multiprocessing import Process, Pool
from torch.multiprocessing import Pool, Process, set_start_method
import matplotlib.pyplot as plt

logger = logging.getLogger('UST')


if torch.cuda.device_count() >= 2:
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

device = torch.device("cuda")# if torch.cuda.is_available() else "cpu")

# device = "cpu"
print("The device is : ", device)



def multigpu(model):
    if torch.cuda.device_count() >= 2:
        model = nn.DataParallel(model).to(device)
        return model 
    model = model.to(device)
    return model


class BertModel(torch.nn.Module):
    def __init__(self, checkpoint, num_labels=10):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)
        #self.model.classifier.dropout.p = 0.5
        self.T = torch.nn.Parameter(torch.ones(1) * 1.0)


    def forward(self, input_ids, token_type_ids, attention_mask, temperature_scaling=False):
        if temperature_scaling:
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            temperature = self.T.unsqueeze(1).expand(outputs.logits.size(0), outputs.logits.size(1))
            outputs.logits /= temperature
        else:
            outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return outputs


def evaluate(model, test_dataloader, criterion, batch_size, num_labels, temp_scaling=False):
    full_predictions = []
    true_labels = []
    probabilities = []

    model.eval()
    crt_loss = 0

    with torch.no_grad():
        for elem in tqdm(test_dataloader):
            x = {key: elem[key].to(device)
                for key in elem if key not in ['idx']}
            logits = model(
                input_ids=x['input_ids'], token_type_ids=x.get('token_type_ids'), attention_mask=x['attention_mask'], temperature_scaling=temp_scaling)
            results = torch.argmax(logits.logits, dim=1)
            prob = F.softmax(logits.logits.to('cpu'), dim=1)
            probabilities += list(prob)

            crt_loss += criterion(logits.logits, x['lbl']
                                ).cpu().detach().numpy()
            full_predictions = full_predictions + \
                list(results.cpu().detach().numpy())
            true_labels = true_labels + list(elem['lbl'].cpu().detach().numpy())


    model.train()

    metric = MulticlassCalibrationError(num_classes=num_labels, n_bins=10, norm='l1')
    metric = metric.to(device)

    preds = torch.stack(probabilities)
    preds = preds.to(device)

    orig = torch.tensor(true_labels, dtype=torch.float, device=device)

    ece_metric = metric(preds, orig).to(device)

    return f1_score(true_labels, full_predictions, average='macro'), crt_loss / len(test_dataloader), ece_metric


def predict_unlabeled(model, ds_unlabeled):
    model.eval()
    data_loader_unlabeled = torch.utils.data.DataLoader(ds_unlabeled, batch_size=128, shuffle=False) 
    y_pred_unlbl = []  

    with torch.no_grad():
        for elem in data_loader_unlabeled:
                x = {key: elem[key].to(device) for key in elem if key not in ['idx', 'weights']}
                pred = model(input_ids=x['input_ids'], token_type_ids=x.get('token_type_ids'), attention_mask=x['attention_mask'])
                y_pred_unlbl.extend(pred.logits.cpu().numpy())

        y_pred_unlbl = np.array(y_pred_unlbl)
        y_pred_unlbl = np.argmax(y_pred_unlbl, axis=-1).flatten()

    pseudolabeled_data = CustomDataset_tracked(ds_unlabeled.text_list, y_pred_unlbl, ds_unlabeled.idxes, ds_unlabeled.tokenizer, labeled=True)
    return pseudolabeled_data


def train_ssl_with_aum(pt_teacher_checkpoint, ds_train, ds_pseudolabeled, ulb_epochs, aum_calculator, ls, sup_batch_size, num_labels):
    model = BertModel(pt_teacher_checkpoint, num_labels=num_labels)
    #model.to(device)
    model = multigpu(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-05)
    model.train()

    loss_fn_supervised = torch.nn.CrossEntropyLoss(reduction='mean', label_smoothing=ls)
    loss_fn_unsupervised = torch.nn.CrossEntropyLoss(reduction='none', label_smoothing=ls)

    data_sampler = torch.utils.data.RandomSampler(
        ds_train, num_samples=10**4)
    batch_sampler = torch.utils.data.BatchSampler(
        data_sampler, sup_batch_size, drop_last=False)
    train_dataloader = torch.utils.data.DataLoader(
        ds_train, batch_sampler=batch_sampler)
    
    data_loader_unlabeled = torch.utils.data.DataLoader(ds_pseudolabeled, batch_size=128, shuffle=False) 

    for epoch in range(ulb_epochs):
        for data_supervised, data_unsupervised in tqdm(zip(train_dataloader, data_loader_unlabeled)):
            cuda_tensors_supervised = {key: data_supervised[key].to(
                device) for key in data_supervised if key not in ['idx']}

            cuda_tensors_unsupervised = {key: data_unsupervised[key].to(
                device) for key in data_unsupervised if key not in ['idx']}

            merged_tensors = {}
            for k in cuda_tensors_supervised:
                merged_tensors[k] = torch.cat(
                    (cuda_tensors_supervised[k], cuda_tensors_unsupervised[k]))

            num_lb = cuda_tensors_supervised['input_ids'].shape[0]

            optimizer.zero_grad()
            # logits = model(input_ids=merged_tensors['input_ids'], token_type_ids=merged_tensors[
            #     'token_type_ids'], attention_mask=merged_tensors['attention_mask'])
            
            logits_lbls = model(input_ids=cuda_tensors_supervised['input_ids'], token_type_ids=cuda_tensors_supervised.get(
                'token_type_ids'), attention_mask=cuda_tensors_supervised['attention_mask']).logits
            logits_ulbl = model(input_ids=cuda_tensors_unsupervised['input_ids'], token_type_ids=cuda_tensors_unsupervised.get(
                'token_type_ids'), attention_mask=cuda_tensors_unsupervised['attention_mask']).logits

            aum_calculator.update(logits_ulbl.detach(), cuda_tensors_unsupervised['lbl'], data_unsupervised['idx'].numpy())

            loss_sup = loss_fn_supervised(
                logits_lbls, cuda_tensors_supervised['lbl'])
            loss_unsup = loss_fn_unsupervised(
                logits_ulbl, cuda_tensors_unsupervised['lbl'])
            loss = 0.5 * loss_sup + 0.5 * torch.mean(loss_unsup)
            loss.backward()
            optimizer.step()


def train_ssl_no_aum_with_mixup(pt_teacher_checkpoint, ds_train,val_dataloader, ds_low_aum, ds_high_aum, ulb_epochs, ls, model_dir, 
                                sup_batch_size, best_f1_overall, best_f1, num_labels):
    model = BertModel(pt_teacher_checkpoint, num_labels)
    model = multigpu(model)
    #model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-05)
    model.train()

    loss_fn_supervised = torch.nn.CrossEntropyLoss(reduction='mean', label_smoothing=ls)
    loss_fn_unsupervised = torch.nn.CrossEntropyLoss(reduction='none', label_smoothing=ls)

    data_sampler = torch.utils.data.RandomSampler(
        ds_train, num_samples=10**4)
    batch_sampler = torch.utils.data.BatchSampler(
        data_sampler, sup_batch_size, drop_last=False)
    train_dataloader = torch.utils.data.DataLoader(
        ds_train, batch_sampler=batch_sampler)
    
    unlabeled_low = torch.utils.data.DataLoader(ds_low_aum, batch_size=128, shuffle=False) 
    unlabeled_high = torch.utils.data.DataLoader(ds_high_aum, batch_size=128, shuffle=False) 
    
    crt_patience = 0
    for epoch in range(ulb_epochs):
        for data_supervised, data_unsup_low, data_unsup_high in tqdm(zip(train_dataloader, unlabeled_low, unlabeled_high)):
            cuda_tensors_supervised = {key: data_supervised[key].to(
                    device) for key in data_supervised if key not in ['idx', 'weights']}

            cuda_tensors_unsup_low = {key: data_unsup_low[key].to(
                    device) for key in data_unsup_low if key not in ['idx']}
                
            cuda_tensors_unsup_high = {key: data_unsup_high[key].to(
                    device) for key in data_unsup_high if key not in ['idx']}

            num_lb = cuda_tensors_supervised['input_ids'].shape[0]
            num_ulb_low = cuda_tensors_unsup_low['input_ids'].shape[0]
            num_ulb_high = cuda_tensors_unsup_high['input_ids'].shape[0]
            
            merged_tensors = {}
            for k in cuda_tensors_supervised:
                merged_tensors[k] = torch.cat((cuda_tensors_supervised[k], cuda_tensors_unsup_low[k], cuda_tensors_unsup_high[k]))
            
            optimizer.zero_grad()
            logits = model(input_ids=merged_tensors['input_ids'], token_type_ids=merged_tensors.get('token_type_ids'), attention_mask=merged_tensors['attention_mask'])

            logits_lbls = logits.logits[:num_lb]
            logits_ulbl_low = logits.logits[num_lb:num_lb+num_ulb_low]
            logits_ulbl_high = logits.logits[num_lb+num_ulb_low:]

            alpha = 0.4
            lam = np.random.beta(alpha,alpha)


            #---- OLD --- 
                
            # ------ mixup loss here ---------
            labels_lbls = F.one_hot(cuda_tensors_supervised['lbl'],num_classes=logits_lbls.shape[1])
            
            #one hot of unlabeled low and high
            labels_ulbl_low = F.one_hot(cuda_tensors_unsup_low['lbl'],num_classes=logits_ulbl_low.shape[1])
            labels_ulbl_high = F.one_hot(cuda_tensors_unsup_high['lbl'],num_classes=logits_ulbl_high.shape[1]) 
            
            batch_size = logits_lbls.shape[0] 
            
            # mixup of labeled  and low aum unlabeled 
            M_logits_lbl_ulbl_low = logits_lbls * lam + (1-lam) * logits_ulbl_low[:batch_size]
            M_labels_lbl_ulbl_low = labels_lbls * lam + (1-lam) * labels_ulbl_low[:batch_size]
            
            #mixup of unlabeled low aum and high aum
            M_logits_ulbl_high_ulbl_low = logits_ulbl_high * lam + (1-lam) * logits_ulbl_low
            M_labels_ulbl_high_ulbl_low = labels_ulbl_high * lam + (1-lam) * labels_ulbl_low

            loss_lbl = loss_fn_supervised(logits_lbls, cuda_tensors_supervised['lbl'])
            loss_ulbl_high = loss_fn_unsupervised(logits_ulbl_high, cuda_tensors_unsup_high['lbl'])
            
            
            loss_M1 = torch.mean(torch.sum(-M_labels_lbl_ulbl_low * torch.log_softmax(M_logits_lbl_ulbl_low, dim=-1), dim=0))
            #print(loss_M1)
            loss_M2 = torch.mean(torch.sum(-M_labels_ulbl_high_ulbl_low * torch.log_softmax(M_logits_ulbl_high_ulbl_low, dim=-1), dim=0))
            #print(loss_M2)

            loss = 0.25 * loss_lbl + 0.25 * torch.mean(loss_ulbl_high) + 0.25 * loss_M1 + 0.25 * loss_M2
            torch.autograd.set_detect_anomaly(True)
            loss.backward()
            optimizer.step()

            f1_macro_validation, loss_validation, ece = evaluate(model, val_dataloader, loss_fn_supervised, 128, num_labels)
            print('Confident learning metrics', f1_macro_validation)

            if f1_macro_validation >= best_f1:
                crt_patience = 0
                best_f1 = f1_macro_validation
                if best_f1 > best_f1_overall:
                    torch.save(model.state_dict(), "/home/b/bharanibala/datasets/humaid_data/data/" + model_dir + "/pytorch_model.bin")
                    best_f1_overall = best_f1
                print('New best macro validation', best_f1, 'Epoch', epoch)
                continue
        
            if crt_patience == 3:
                crt_patience = 0
                print('Exceeding max patience; Exiting..')
                break

            crt_patience += 1

    return best_f1_overall, best_f1


def	train_model_st_with_aummixup(ds_train, ds_dev, ds_test, ds_unlabeled, pt_teacher_checkpoint, cfg, model_dir, aum_save_dir, num_labels, sup_batch_size=16, unsup_batch_size=64, unsup_size=4096, sample_size=16384,
                                 T=30, alpha=0.1, sup_epochs=20, unsup_epochs=25, N_base=10, dense_dropout=0.5, attention_probs_dropout_prob=0.3, hidden_dropout_prob=0.3,
                                 results_file="", temp_scaling=False, ls=0.0):

    load_best = False
    logger_dict = {}
    logger_dict["Temperature Scaling"] = temp_scaling
    logger_dict["Label Smoothing"]= ls

    train_dataloader = torch.utils.data.DataLoader(
        ds_train, batch_size=sup_batch_size, shuffle=True)   
    validation_dataloader = torch.utils.data.DataLoader(
        ds_dev, batch_size=sup_batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(
        ds_test, batch_size=128, shuffle=False)
    
    cfg.num_labels = num_labels
    copy_cfg = deepcopy(cfg)
    copy_cfg.attention_probs_dropout_prob = 0.1
    copy_cfg.hidden_dropout_prob = 0.1

    best_f1_overall = 0
    crt_patience = 0
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=ls)

    if load_best == False:
        for counter in range(N_base):
            best_f1 = 0
            copy_cfg.return_dict  = True

            model = BertModel(pt_teacher_checkpoint, num_labels)
            model = multigpu(model)
            #model.to(device)
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-05)
            if counter == 0:
                logger.info(model)
            for epoch in range(sup_epochs):
                for data in tqdm(train_dataloader):
                    cuda_tensors = {key: data[key].to(
                        device) for key in data if key not in ['idx', 'weights']}
                    optimizer.zero_grad()
                    logits = model(input_ids=cuda_tensors['input_ids'], token_type_ids=cuda_tensors.get('token_type_ids'), attention_mask=cuda_tensors['attention_mask'])
                    loss = loss_fn(logits.logits, cuda_tensors['lbl'])
                    loss.backward()
                    optimizer.step()

                f1_macro_validation, loss_validation, ece = evaluate(model, validation_dataloader, loss_fn, unsup_batch_size, num_labels)

                if f1_macro_validation >= best_f1:
                    crt_patience = 0
                    best_f1 = f1_macro_validation
                    if best_f1 > best_f1_overall:
                        torch.save(model.state_dict(), "/home/b/bharanibala/datasets/humaid_data/data/" + model_dir + "/pytorch_model.bin")
                        best_f1_overall = best_f1
                    print('New best macro validation', best_f1, 'Epoch', epoch)
                    continue
            
                if crt_patience == 3:
                    crt_patience = 0
                    print('Exceeding max patience; Exiting..')
                    break

                crt_patience += 1

        del model

    cfg.return_dict = True

    best_model = BertModel(pt_teacher_checkpoint, num_labels)
    best_model = multigpu(best_model)
    #best_model.to(device)
    state_dict = torch.load("/home/b/bharanibala/datasets/humaid_data/data/" + model_dir + "/pytorch_model.bin")
    best_model.load_state_dict(state_dict)

    for epoch in range(unsup_epochs):
        aum_calculator = AUMCalculator(aum_save_dir, compressed=False)

        pseudolabled_data = predict_unlabeled(best_model, ds_unlabeled)

        train_ssl_with_aum(pt_teacher_checkpoint, ds_train, pseudolabled_data, sup_epochs, aum_calculator, ls, sup_batch_size, num_labels)
        aum_calculator.finalize()
        aum_values_df = pd.read_csv(os.path.join(aum_save_dir, 'aum_values.csv'))

        aum_values = aum_values_df['aum'].to_list()
        aum_values.sort()
        median_aum_id = int(float(len(aum_values))* 0.5)
        median_aum_value = aum_values[median_aum_id]
        high_aum_ids, low_aum_ids = [], []

        for i, row in aum_values_df.iterrows():
            if row['aum'] > median_aum_value:
                high_aum_ids.append(int(row['sample_id']))
            else:
                low_aum_ids.append(int(row['sample_id']))

        print("Low aum : ", len(low_aum_ids))
        print("High aum : ", len(high_aum_ids))


        low_aum_data = pseudolabled_data.get_subset_dataset(low_aum_ids)
        high_aum_data = pseudolabled_data.get_subset_dataset(high_aum_ids)

        ds_unlabeled_low = CustomDataset(low_aum_data.text_list, low_aum_data.labels, ds_unlabeled.tokenizer, labeled=True)
        ds_unlabeled_high = CustomDataset(high_aum_data.text_list, high_aum_data.labels, ds_unlabeled.tokenizer, labeled=True)

        #------------------------------------------------------
        # Next SSL after aum calculation and with mixup
        #------------------------------------------------------

        best_f1_overall, best_f1 = train_ssl_no_aum_with_mixup(pt_teacher_checkpoint, ds_train, validation_dataloader, 
                                    ds_unlabeled_low, ds_unlabeled_high, sup_epochs, ls, model_dir, sup_batch_size, best_f1_overall, best_f1, num_labels)



    copy_cfg.return_dict = True
    model = BertModel(pt_teacher_checkpoint, num_labels)
    model = multigpu(model)
    #model.to(device)
    state_dict = torch.load("/home/b/bharanibala/datasets/humaid_data/data/" + model_dir + "/pytorch_model.bin")
    model.load_state_dict(state_dict)

    f1_macro_test, loss_test, ece_metric = evaluate(model, test_dataloader, loss_fn, unsup_batch_size, num_labels)
    logger.info ("Test macro F1 based on best validation f1 : {}".format(f1_macro_test))

    logger_dict["Best ST+AumMixup model"] = {}
    logger_dict["Best ST+AumMixup model"]["F1 before temp scaling"] = str(f1_macro_test)
    logger_dict["Best ST+AumMixup model"]["ECE before temp scaling"] = str(ece_metric)


    #logger_dict["Best ST+AumMixup model"]["T before temp scaling"] = str(model.T.detach().cpu().numpy()[0])

    if temp_scaling:
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-02)

        for epoch in range(20):
            for data in tqdm(validation_dataloader):
                cuda_tensors = {key: data[key].to(
                        device) for key in data if key not in ['idx', 'weights']}
                optimizer.zero_grad()
  
                result = model(cuda_tensors['input_ids'], cuda_tensors.get('token_type_ids'), cuda_tensors['attention_mask'], True)

                loss = loss_fn(result.logits, cuda_tensors['lbl'])
                loss.backward()
                optimizer.step()


        f1_macro_test, loss_test, ece_metric = evaluate(model, test_dataloader, loss_fn, unsup_batch_size, num_labels, True)

        logger_dict["Best ST+AumMixup model"]["F1 after temp scaling"] = str(f1_macro_test)
        logger_dict["Best ST+AumMixup model"]["ECE after temp scaling"] = str(ece_metric)

        #logger_dict["Best ST+AumMixup model"]["T  after temp scaling"] = str(model.T.detach().cpu().numpy()[0])

    print(json.dumps(logger_dict, indent=4))
    with open("/home/b/bharanibala/datasets/humaid_data/data/" + model_dir +"/"+ results_file + '.txt','w') as fp:
        fp.write(json.dumps(logger_dict, indent=4))