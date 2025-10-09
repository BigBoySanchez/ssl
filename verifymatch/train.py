import argparse
import csv
import os
import sys
import json
import pickle
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pyparsing as pp
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from transformers import (  
    get_constant_schedule, 
    get_constant_schedule_with_warmup, 
    get_cosine_schedule_with_warmup, 
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup
)

from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer

from torch.distributions.distribution import Distribution
from tqdm import tqdm
from torch.distributions import Categorical
from itertools import cycle

from datasets import load_from_disk, load_dataset
import time

# Pick the largest value the platform allows
SAFE_LIMIT = 2**31 - 1  # max signed 32-bit int
csv.field_size_limit(min(sys.maxsize, SAFE_LIMIT))

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0, help='CUDA device')
parser.add_argument('--model', type=str, help='pre-trained model (bert-base-uncased, roberta-base)')
parser.add_argument('--task', type=str, help='task name (SNLI, MNLI, QQP, TwitterPPDB, SWAG, HellaSWAG)')
parser.add_argument('--max_seq_length', type=int, default=256, help='max sequence length')
parser.add_argument('--ckpt_path', type=str, help='model checkpoint path')
parser.add_argument('--output_path', type=str, help='model output path')
parser.add_argument('--train_path', type=str, help='train dataset path')
parser.add_argument('--dev_path', type=str, help='dev dataset path')
parser.add_argument('--test_path', type=str, help='test dataset path')
parser.add_argument('--epochs', type=int, default=3, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0., help='weight decay')
parser.add_argument('--label_smoothing', type=float, default=-1., help='label smoothing \\alpha')
parser.add_argument('--max_grad_norm', type=float, default=1., help='gradient clip')
parser.add_argument('--do_train', action='store_true', default=False, help='enable training')
parser.add_argument('--do_evaluate', action='store_true', default=False, help='enable evaluation')
parser.add_argument('--warmup_steps',type=int, default=0)
parser.add_argument('--gradient_accumulation_steps',default=1)
parser.add_argument('--labeled_train_path', type=str, help='labeled train dataset path')
parser.add_argument('--unlabeled_train_path', type=str, help='unlabeled train dataset path')
parser.add_argument('--ssl',action='store_true',help='Semi-supervised learning')
parser.add_argument('--mixup',action='store_true')
parser.add_argument('--pseudo_label_by_normalized',action='store_true')
parser.add_argument('--same_domain_unlabeled',action='store_true')
parser.add_argument('--th', type=float, default=0.7)
parser.add_argument('--sharpening',action='store_true',default=True)
parser.add_argument('--T',type=float,default=0.1)
parser.add_argument('--seed',type=int,default=int(time.time()))
parser.add_argument('--rand_mixup',action='store_true')
parser.add_argument('--mixup_loss_weight',type=float, default=1.)
parser.add_argument('--consistency',action='store_true')
parser.add_argument('--high_mixup',action='store_true',default=False)
parser.add_argument('--multigpus',action='store_true')
parser.add_argument('--unlabeled_batch_size',type=int,default=32)
args = parser.parse_args()
print(args)


assert args.task in ('SNLI', 'MNLI', 'QQP', 'TwitterPPDB', 'SWAG', 'HellaSWAG', 'SICK','RTE','FEVER','HANS','CrisisMMDINF', 'HumAID')
assert args.model in ('bert-base-uncased', 'roberta-base', 'bert-large-uncased')
if args.task in ('HumAID'):
    n_classes = 10
elif args.task in ('SNLI', 'MNLI','SICK','FEVER','HANS'):
    n_classes = 3
elif args.task in ('QQP', 'TwitterPPDB','RTE','CrisisMMDINF'):
    n_classes = 2
elif args.task in ('SWAG', 'HellaSWAG'):
    n_classes = 1

def cuda(tensor):
    """Places tensor on CUDA device."""
    if args.multigpus:
        return tensor.cuda()
    else:
        return tensor.to(args.device)


def load(dataset, batch_size, shuffle):
    """Creates data loader with dataset and iterator options."""

    return DataLoader(dataset, batch_size, shuffle=shuffle)


def adamw_params(model):
    """Prepares pre-trained model parameters for AdamW optimizer."""

    no_decay = ['bias', 'LayerNorm.weight']
    params = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        },
    ]
    return params


def encode_pair_inputs(sentence1, sentence2):
    """
    Encodes pair inputs for pre-trained models using the template
    [CLS] sentence1 [SEP] sentence2 [SEP]. Used for SNLI, MNLI, QQP, and TwitterPPDB.
    Returns input_ids, segment_ids, and attention_mask.
    """

    inputs = tokenizer.encode_plus(
        sentence1, sentence2, add_special_tokens=True, max_length=args.max_seq_length
    )
    input_ids = inputs['input_ids']
    if args.model == 'bert-base-uncased' or args.model == 'bert-large-uncased':
        segment_ids = inputs['token_type_ids']
    else:
        segment_ids = [0]*len(inputs['input_ids'])
    attention_mask = [1]*len(inputs['input_ids'])#inputs['attention_mask']
    padding_length = args.max_seq_length - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length
    segment_ids += [0] * padding_length
    attention_mask += [0] * padding_length
    for input_elem in (input_ids, segment_ids, attention_mask):
        assert len(input_elem) == args.max_seq_length
    return (
        cuda(torch.tensor(input_ids)).long(),
        cuda(torch.tensor(segment_ids)).long(),
        cuda(torch.tensor(attention_mask)).long(),
    )


def encode_mc_inputs(context, start_ending, endings):
    """
    Encodes multiple choice inputs for pre-trained models using the template
    [CLS] context [SEP] ending_i [SEP] where 0 <= i < len(endings). Used for
    SWAG and HellaSWAG. Returns input_ids, segment_ids, and attention_masks.
    """
    all_input_ids = []
    all_segment_ids = []
    all_attention_masks = []
    for ending in endings:
        inputs = tokenizer.encode_plus(
            context, start_ending+" " + ending, add_special_tokens=True, max_length=args.max_seq_length
        )
        input_ids = inputs['input_ids']
        if args.model == 'bert-base-uncased' or args.model == 'bert-large-uncased':
            segment_ids = inputs['token_type_ids']
        else:
            segment_ids = [0] * len(inputs['input_ids'])
        attention_mask = inputs['attention_mask']
        padding_length = args.max_seq_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_length
        segment_ids += [0] * padding_length
        attention_mask += [0] * padding_length
        for input_elem in (input_ids, segment_ids, attention_mask):
            assert len(input_elem) == args.max_seq_length
        all_input_ids.append(input_ids)
        all_segment_ids.append(segment_ids)
        all_attention_masks.append(attention_mask)
    return (
        cuda(torch.tensor(all_input_ids)).long(),
        cuda(torch.tensor(all_segment_ids)).long(),
        cuda(torch.tensor(all_attention_masks)).long(),
    )


def encode_label(label):
    """Wraps label in tensor."""

    return cuda(torch.tensor(label)).long()

class HumAIDProcessor:
    """Data loader for HumAID."""

    def __init__(self):
        self.label_map = {'not_humanitarian': 0, 'requests_or_urgent_needs': 1, 
                          'rescue_volunteering_or_donation_effort': 2, 'infrastructure_and_utility_damage': 3, 
                          'missing_or_found_people': 4, 'displaced_people_and_evacuations': 5, 
                          'sympathy_and_support': 6, 'injured_or_dead_people': 7, 
                          'caution_and_advice': 8, 'other_relevant_information': 9}

    def valid_inputs(self, sentence1, label):
        return len(sentence1) > 0 and label in self.label_map

    def load_samples(self, path):
        samples = []
        ds = load_dataset("csv", data_files=path, split="train", delimiter="\t")

        for ex in ds:
            try:
                guid = ex["tweet_id"]
                sentence1 = ex["tweet_text"]
                label = ex["class_label"]
                if self.valid_inputs(sentence1, label):
                    label = int(self.label_map[label])
                    samples.append((sentence1, "", label, guid))
            except:
                pass

        return samples

class CrisisMMDINFProcessor:
    """
    Loads CrisisMMD informative classification from a Hugging Face
    saved-to-disk directory (e.g., .../crisismmd2inf_dataset/train).

    It emits 4-tuples that your TextDataset expects for *pair* classification:
      (sentence1, sentence2, label_int, guid)
    where:
      - sentence1 = tweet_text
      - sentence2 = event_name  <-- added per request
      - label_int ∈ {0,1} (not_informative=0, informative=1)
      - guid = tweet_id (fallback to image_id if missing)

    NOTE: The pair gets encoded as:
      [CLS] sentence1 [SEP] sentence2 [SEP]
    which your encode_pair_inputs() already implements. No other code changes needed.
    """

    def __init__(self, label_field="label_text"):
        # Prefer text-only supervision for a text model; change to "label_text_image"
        # if you want the joint (text+image) supervision target instead.
        self.label_field = label_field
        # Map string labels -> ints. If your stored labels are already ints, we’ll cast below.
        self.label_map = {"not_informative": 0, "informative": 1}

    def valid_inputs(self, s1, s2, label):
        # s1 must exist and label must be 0/1; s2 (event) can be empty string if missing.
        return (s1 is not None) and (len(s1) > 0) and (label in (0, 1))

    def _to_int_label(self, raw):
        # Accept ints (0/1) or strings ("informative"/"not_informative")
        if isinstance(raw, (int, float)) and int(raw) in (0, 1):
            return int(raw)
        return self.label_map.get(str(raw))

    def load_samples(self, path):
        # `path` points to a split folder produced by Dataset.save_to_disk()
        ds = load_from_disk(path)

        samples = []
        for ex in ds:
            # sentence1: tweet text
            s1 = ex.get("tweet_text")

            # sentence2: event context (can be "")
            # s2 = ex.get("event_name") or ""
            s2 = ""

            # choose label source (prefer text-only; fallback to overall label)
            raw_label = ex.get(self.label_field)
            if raw_label is None:
                raw_label = ex.get("label")

            label = self._to_int_label(raw_label)

            # stable-ish ID for debugging/traceability
            guid = ex.get("tweet_id", [])

            if self.valid_inputs(s1, s2, label):
                samples.append((s1, s2, label, guid))

        return samples

class FEVERProcessor:
    """Data loader for QQP."""

    def __init__(self):
        self.label_map = {"SUPPORTS":0, "REFUTES":1, "NOT ENOUGH INFO":2}

    def valid_inputs(self, sentence1, sentence2, label):
        return len(sentence1) > 0 and len(sentence2) > 0 and label in self.label_map

    def load_samples(self, path):
        samples = []
        # guid_check = []
        # label_check = []
        #new_file = open("./data/FEVER/sym_test_v1.txt","w")
        #new_file = open("./data/FEVER/train.jsonl","w")
        #new_data = []
        df = pd.read_json(path, lines=True)
        for i, (_, line) in enumerate(df.iterrows()):
            #line['id'] = line['id']#int(i)
            #new_data.append(dict(line))
            if "unique_id" in line:
                guid = line["unique_id"]
            else:
                guid = line["id"]

            sentence1 = line["claim"]
            try:
                sentence2 = line["evidence"]
                label = line["gold_label"]
            except:
                sentence2 = line["evidence_sentence"]
                label = line["label"]
            label = self.label_map[label]
            label = int(label)
            samples.append((sentence1, sentence2, label, guid))
        return samples

class SNLIProcessor:
    """Data loader for SNLI."""

    def __init__(self):
        self.label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

    def valid_inputs(self, sentence1, sentence2, label):
        return len(sentence1) > 0 and len(sentence2) > 0 and label in self.label_map

    def load_samples(self, path):
        samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # skip header
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    guid = row[0]
                    sentence1 = row[7]
                    sentence2 = row[8]
                    label = row[-1]
                    if self.valid_inputs(sentence1, sentence2, label):
                        label = self.label_map[label]
                        samples.append((sentence1, sentence2, label, guid))
                except:
                    pass
        return samples

class SNLITESTProcessor:
    """Data loader for SNLI."""

    def __init__(self):
        self.label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

    def valid_inputs(self, sentence1, sentence2, label):
        return len(sentence1) > 0 and len(sentence2) > 0 and label in self.label_map

    def load_samples(self, path):
        samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # skip header
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    # guid = row[0]
                    sentence1 = row[4]
                    sentence2 = row[7]
                    label = row[2]
                    if self.valid_inputs(sentence1, sentence2, label):
                        label = self.label_map[label]
                        samples.append((sentence1, sentence2, label, []))
                except:
                    pass
        return samples

##HANS
class HANSProcessor(SNLIProcessor):
    """Data loader for MNLI."""

    def load_samples(self, path):
        samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # skip header
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                sentence1 = row[5]
                sentence2 = row[6]
                label = row[0]
                if label == 'non-entailment':
                    label = 'contradiction'
                label = self.label_map[label]
                samples.append((sentence1, sentence2, label, []))
        return samples



class MNLIProcessor(SNLIProcessor):
    """Data loader for MNLI."""

    def load_samples(self, path):
        samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # skip header
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    sentence1 = row[8]
                    sentence2 = row[9]
                    label = row[-1]
                    if self.valid_inputs(sentence1, sentence2, label):
                        label = self.label_map[label]
                        samples.append((sentence1, sentence2, label, []))
                except:
                    pass
        return samples

class MNLITESTProcessor:
    """Data loader for MNLI."""
    def __init__(self):
        self.label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

    def valid_inputs(self, sentence1, sentence2, label):
        return len(sentence1) > 0 and len(sentence2) > 0 and label in self.label_map

    def load_samples(self, path):
        samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # skip header
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    # ## Stress Test Eval
                    # sentence1 = row[1]
                    # sentence2 = row[2]
                    # label = row[0]

                    # # RTE
                    # if label == 'contradiction':
                    #     label = 0
                    # elif label == 'neutral':
                    #     label = 0 
                    # else:
                    #     label = 1
                    # samples.append((sentence1, sentence2, label, []))
                    sentence1 = row[5]
                    sentence2 = row[6]
                    label = row[0]
                    if self.valid_inputs(sentence1, sentence2, label):
                        label = self.label_map[label]
                        samples.append((sentence1, sentence2, label, []))
                except:
                    pass
        return samples


class SICKProcessor():
    def __init__(self):
        self.label_map = {'entailment': 0, 'contradiction': 2, 'neutral': 1}
        self.label_list = [0,1,2]

    def valid_inputs(self, sentence1, sentence2, label):
        return len(sentence1) > 0 and len(sentence2) > 0 and label in self.label_list


    def load_samples(self, path):
        samples = []
        e,c,n = 0,0,0
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # skip header
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                #try:
                guid = row[0]
                sentence1 = row[1]
                sentence2 = row[2]
                try:
                    label = int(row[3])
                except:
                    label = self.label_map[row[3]]
                if self.valid_inputs(sentence1, sentence2, label):
                    samples.append((sentence1, sentence2, label, guid))
                # except:
                #     pass
        return samples



class RTEProcessor():
    def __init__(self):
        self.label_map = {'entailment': 1, 'not_entailment':0}
        
    def valid_inputs(self, sentence1, sentence2, label):
        return len(sentence1) > 0 and len(sentence2) > 0 and label in self.label_map


    def load_samples(self, path):
        samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # skip header
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                #try:
                guid = row[0]
                sentence1 = row[1]
                sentence2 = row[2]
                label = row[3]
                if self.valid_inputs(sentence1, sentence2, label):
                    label = int(self.label_map[label])
                    samples.append((sentence1, sentence2, label, guid))
                # except:
                #     pass
        return samples


class QQPProcessor:
    """Data loader for QQP."""

    def valid_inputs(self, sentence1, sentence2, label):
        return len(sentence1) > 0 and len(sentence2) > 0 and label in ('0', '1')

    def load_samples(self, path):
        samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # skip header
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    guid = row[0]
                    sentence1 = row[3]
                    sentence2 = row[4]
                    label = row[5]
                    if self.valid_inputs(sentence1, sentence2, label):
                        label = int(label)
                        samples.append((sentence1, sentence2, label, guid))
                except:
                    pass
        return samples


class TwitterPPDBProcessor:
    """Data loader for TwittrPPDB."""

    def valid_inputs(self, sentence1, sentence2, label):
        return len(sentence1) > 0 and len(sentence2) > 0 and label != 3 
    
    def load_samples(self, path):
        samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    sentence1 = row[0]
                    sentence2 = row[1]
                    label = eval(row[2])[0]
                    if self.valid_inputs(sentence1, sentence2, label):
                        label = 0 if label < 3 else 1
                        samples.append((sentence1, sentence2, label, []))
                except:
                    pass
        return samples

class SWAGProcessor:
    """Data loader for SWAG."""

    def load_samples(self, path):
        samples = []
        file = open(path,"r")
        data = file.read().strip().split("\n")
        for item in data:
            row =  pp.commaSeparatedList.parseString(item).asList()
            try:
                guid = row[0]
                context = row[4]
                start_ending = row[5]
                endings = row[7:11]
                label = int(row[-1])
                samples.append((context, start_ending, endings, label, guid))
            except:
                pass
        return samples



class HellaSWAGProcessor:
    """Data loader for HellaSWAG."""

    def load_samples(self, path):
        samples = []
        with open(path) as f:
            desc = f'loading \'{path}\''
            for line in f:
                try:
                    line = line.rstrip()
                    input_dict = json.loads(line)
                    context = input_dict['ctx_a']
                    start_ending = input_dict['ctx_b']
                    endings = input_dict['endings']
                    label = input_dict['label']
                    samples.append((context, start_ending, endings, label, []))
                except:
                    pass
        return samples


def select_processor():
    """Selects data processor using task name."""

    return globals()[f'{args.task}Processor']()



def smoothing_label(target, smoothing):
    """Label smoothing"""
    _n_classes = n_classes if args.task not in ('SWAG', 'HellaSWAG') else 4
    confidence = 1. - smoothing
    smoothing_value = smoothing / (_n_classes - 1)
    one_hot = cuda(torch.full((_n_classes,), smoothing_value))
    model_prob = one_hot.repeat(target.size(0), 1)
    model_prob.scatter_(1, target.unsqueeze(1), confidence)
    return model_prob


class TextDataset(Dataset):
    """
    Task-specific dataset wrapper. Used for storing, retrieving, encoding,
    caching, and batching samples.
    """

    def __init__(self, path, processor, num_instances=None, augment=False):
        # print(path)
        # if path is not None:
        self.samples = processor.load_samples(path)
        self.unlabeled = False
        self.cache = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        res = self.cache.get(i, None)
        if res is None:
            sample = self.samples[i]
            if args.task in ('SNLI', 'MNLI', 'QQP', 'MRPC', 'TwitterPPDB','SICK','RTE','FEVER','HANS','CrisisMMDINF', 'HumAID'): # and not self.unlabeled:
                sentence1, sentence2, label, guid = sample
                input_ids, segment_ids, attention_mask = encode_pair_inputs(
                    sentence1, sentence2
                )
                label_id = encode_label(label)
                res = ((input_ids, segment_ids, attention_mask, guid, [sentence1+' [SEP] ' +sentence2]), label_id)
            elif args.task in ('SWAG', 'HellaSWAG'):
                if self.unlabeled:
                    context, ending_start, endings = sample
                    guid = -1
                else:
                    context, ending_start, endings, label, guid = sample
                input_ids, segment_ids, attention_mask = encode_mc_inputs(
                    context, ending_start, endings
                )
                label_id = encode_label(label)
                res = ((input_ids, segment_ids, attention_mask, guid), label_id)
            self.cache[i] = res
        return res


class Model(nn.Module):
    """Pre-trained model for classification."""

    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(args.model)
        if args.model in ('bert-base-uncased', 'roberta-base'):
            self.classifier = nn.Linear(768, n_classes)
        elif args.model in ('bert-large-uncased', 'roberta-large-uncased'):
            self.classifier = nn.Linear(1024,n_classes)
        if args.task in ('SWAG', 'HellaSWAG'):
            self.n_choices = -1

    def forward(self, input_ids, segment_ids, attention_mask, unlabeled=False):
        # On SWAG and HellaSWAG, collapse the batch size and
        # choice size dimension to process everything at once
        if args.task in ('SWAG', 'HellaSWAG'):
            n_choices = input_ids.size(1)
            self.n_choices = n_choices
            input_ids = input_ids.view(-1, input_ids.size(-1))
            segment_ids = segment_ids.view(-1, segment_ids.size(-1))
            attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        transformer_params = {
            'input_ids': input_ids,
            'token_type_ids': (
                segment_ids if args.model in ('bert-base-uncased','bert-large-uncased') else None
            ),
            'attention_mask': attention_mask,
        }
        transformer_outputs = self.model(**transformer_params)
        #if args.consistency_learning or args.noisy_label:
        if args.ssl:
            return transformer_outputs
        else:
            if args.task in ('SWAG', 'HellaSWAG'):
                pooled_output = transformer_outputs[1]
                logits = self.classifier(pooled_output)
                logits = logits.view(-1, self.n_choices)
            else:
                cls_output = transformer_outputs[0][:, 0]
                logits = self.classifier(cls_output)
            return logits


def smoothing_label(target, smoothing):
    """Label smoothing"""
    _n_classes = n_classes if args.task not in ('SWAG', 'HellaSWAG') else 4
    confidence = 1. - smoothing
    smoothing_value = smoothing / (_n_classes - 1)
    one_hot = cuda(torch.full((_n_classes,), smoothing_value))
    model_prob = one_hot.repeat(target.size(0), 1)
    model_prob.scatter_(1, target.unsqueeze(1), confidence)
    return model_prob


prev_mean1 = cuda(torch.tensor(1.,dtype=torch.float))#, cuda(torch.tensor(1.,dtype=torch.float))
prev_std1 = cuda(torch.tensor(1.,dtype=torch.float))#, cuda(torch.tensor(1.,dtype=torch.float))

def train(d1,d2=None,aug=None,epoch=0):
    """Fine-tunes pre-trained model on training set."""
    global prev_mean1, prev_std1
    model.train()
    train_loss = 0.
    if args.ssl:
        d1_loader = load(d1,args.batch_size,True)
        d2_loader = tqdm(load(d2,args.unlabeled_batch_size,False))
        optimizer = AdamW(adamw_params(model),lr=args.learning_rate,eps=1e-8)
        alpha = 0.4
        lam = np.random.beta(alpha,alpha)
        high_count, low_count = 0,0
        high_per_iter, low_per_iter = [], []
        discard_per_iter = []
        unlabeled_not_used_count = 0
        total_num = []
        average = []
        discard_ratio = args.batch_size*0.1
        pbar = tqdm(total=len(d2_loader))
        pbar2 = tqdm(total=len(d2_loader))
        pbar3 = tqdm(total=len(d2_loader))
        pbar4 = tqdm(total=len(d2_loader))


        folder = args.output_path.split("_")[2].replace(".json",'')
        if not os.path.exists(folder):
            os.mkdir(folder)
        unlabeled_not_used = open("./"+folder+"/"+args.task + "_" +str(epoch)+"_dataitself_discard_per_iter_dueToVerification_verifyMatch_bs"+str(args.batch_size)+".txt","w",encoding="utf-8")
        average_tracking = open("./"+folder+"/"+args.task + "_" +str(epoch)+"_avgConf_verifyMatch_bs"+str(args.batch_size)+".txt","w",encoding="utf-8")
        confidence_tracking = open("./"+folder+"/"+args.task + "_" +str(epoch)+"_Conf_verifyMatch_bs"+str(args.batch_size)+".txt","w",encoding="utf-8")
        
        discard_info = open("./"+folder+"/"+args.task + "_" +str(epoch)+"_datanumber_discard_per_iter_dueToVerification_verifyMatch_bs"+str(args.batch_size)+".txt","w",encoding="utf-8")
        low_file = open("./"+folder+"/"+args.task + "_" +str(epoch) + "_low_verfiyMatch_bs"+str(args.batch_size)+".txt","w",encoding="utf-8")
        high_file = open("./"+folder+"/"+args.task + "_" +str(epoch) + "_high_verifyMatch_bs"+str(args.batch_size)+".txt","w",encoding="utf-8")
        for i, (dataset1, dataset2) in enumerate(zip(cycle(d1_loader),d2_loader)):
            optimizer.zero_grad()
            inputs1, labels1 = dataset1
            inputs2, true_labels2 = dataset2
            guid = list(inputs1[3])
            original_unlabeled = inputs2[4][0]
            discard_count = 0 
            smoothing_val = 0.3

            if args.task in ('SNLI','MNLI','SICK','RTE','FEVER','QQP','CrisisMMDINF', 'HumAID'):
                output1 = model(inputs1[0],inputs1[1],inputs1[2], unlabeled = False)[0]
                output2 = model(inputs2[0],inputs2[1],inputs2[2], unlabeled = True)[0]
                #logits1 = model.classifier(output1[:,0])
            elif args.task in ('SWAG'):
                output1 = model(inputs1[0],inputs1[1],inputs1[2], unlabeled = False)[1]
                output2 = model(inputs2[0],inputs2[1],inputs2[2], unlabeled = True)[1]
                #logits1 = model.classifier(output1)
            #if args.task in ('SWAG'): 
            #    logits1 = logits1.view(-1, model.n_choices)

            if args.multigpus:
                logits1 = model.module.classifier(output1[:,0])
            else:
                logits1 = model.classifier(output1[:,0])
            loss1 = criterion(logits1,labels1)

            if output1.shape[0] != output2.shape[0]:
                min_idx = min(output1.shape[0],output2.shape[0])
                output1 = output1[:min_idx,:]
                output2 = output2[:min_idx,:]
                true_labels2 = true_labels2[:min_idx]

            if args.pseudo_label_by_normalized:
                ## Moment injection to unlabeled features
                unlabeled_mean = torch.mean(output2,dim=1) 
                labeled_mean = torch.mean(output1,dim=1)
                unlabeled_std = torch.std(output2,dim=1)
                labeled_std = torch.std(output1,dim=1)
                output1 = output1[:,0]
                output2 = output2[:,0]
                
                output2_perturb = (output2 - unlabeled_mean)/unlabeled_std *labeled_std + labeled_mean
                logits2 = model.classifier(output2_perturb)
                if args.task in ('SWAG'): logits2 = logits2.view(-1,model.n_choices)
            else:
                output2_perturb = output2[:,0]
                if args.multigpus:
                    logits2 = model.module.classifier(output2[:,0])
                else:
                    logits2 = model.classifier(output2[:,0])

            # output1 = output1[:,0]
            output2 = output2[:,0]


            ## Pseudo Label Generation
            if args.task in ('SWAG'): logits2 = logits2.view(-1,model.n_choices)
            tmp_labels2 = F.softmax(logits2,dim=-1)
            if args.sharpening:
                tmp_labels2 = tmp_labels2**(1/args.T)
                tmp_labels2 = tmp_labels2 / tmp_labels2.sum(dim=1, keepdim=True)
            verifier_prob, verifier_label = torch.max(tmp_labels2,dim=-1)
            original_prob, original_idx, original_true_labels2 = verifier_prob, verifier_label, true_labels2

            mismatch_outputs, mismatch_labels =  cuda(torch.zeros([int(output2.shape[0]), 768])), cuda(torch.tensor([[0. for _ in range(n_classes)] for _ in range(int(output2.shape[0]))]))#,cuda(torch.tensor([0 for _ in range(int(output2.shape[0]))]))#cuda(torch.zeros([int(output2.shape[0]/2), n_classes]))
            filtered_outputs, filtered_labels = cuda(torch.zeros([int(output2.shape[0]), 768])), cuda(torch.tensor([0 for _ in range(int(output2.shape[0]))]))#cuda(torch.zeros([int(output2.shape[0]/2), n_classes]))
            filtered_prob = cuda(torch.zeros_like(verifier_prob))
            filtered_label = cuda(torch.zeros_like(verifier_label))
            usage_check = cuda(torch.tensor([-1 for _ in range(0,output2.shape[0])]))
            mismatch_idx, filtered_idx = 0,0
            discard_count = 0
            for output_idx in range(verifier_prob.shape[0]):
                if verifier_label[output_idx] == true_labels2[output_idx]:
                    filtered_outputs[filtered_idx] = output2[output_idx]
                    filtered_labels[filtered_idx] = true_labels2[output_idx]
                    filtered_prob[filtered_idx] = verifier_prob[output_idx]
                    usage_check[output_idx] = 1
                    filtered_idx += 1
                else:
                    discard_count += 1
                    mismatch_outputs[mismatch_idx] = output2[output_idx]
                    mismatch_labels[mismatch_idx,true_labels2[output_idx]] = tmp_labels2[output_idx][true_labels2[output_idx]]
                    mismatch_idx += 1
            mismatch_outputs = mismatch_outputs[:mismatch_idx]
            mismatch_labels = mismatch_labels[:mismatch_idx]
            select_idx = torch.randperm(mismatch_outputs.shape[0])
            labeled4MisMatched = output1[select_idx]
            labeled4MisMatchedLabels = labels1[select_idx]
            labeled4MisMatchedLabelsOneHot = smoothing_label(labeled4MisMatchedLabels,smoothing_val)
            
            discardMixUp = labeled4MisMatched * lam + mismatch_outputs * (1-lam)
            discardMixUpLabels = labeled4MisMatchedLabelsOneHot * lam + mismatch_labels * (1-lam)
            if args.multigpus:
                discardMixUp = model.module.classifier(discardMixUp)
            else:
                discardMixUp = model.classifier(discardMixUp)
            discardMixUpLoss = torch.mean(torch.sum(-discardMixUpLabels * torch.log_softmax(discardMixUp, dim=-1), dim=-1))
            
            output2 = filtered_outputs[:filtered_idx]
            true_labels2 = filtered_labels[:filtered_idx]
            verifier_prob = filtered_prob[:filtered_idx]
            verifier_label = filtered_labels[:filtered_idx]

            for u_idx, u in enumerate(usage_check):
                if u == -1:
                    unlabeled_not_used.write(str(original_unlabeled[u_idx])+"\t"+ str(original_true_labels2[u_idx].data.tolist())+"\t"+str(original_idx[u_idx].data.tolist())+"\n")
            discard_per_iter.append(discard_count)
            avg_prob = torch.mean(verifier_prob)
            average_tracking.write(str(avg_prob.data.tolist())+"\n")
            confidence_tracking.write(str(verifier_prob.data.tolist())+"\n")
            ## When using median model confidence
            #avg_prob = torch.median(prob)
            ## When using fixed high threshold value
            # avg_prob = 0.9
            low_output = cuda(torch.zeros([int(output2.shape[0]), 768])) 
            high_output = cuda(torch.zeros([int(output2.shape[0]), 768]))
            high_true_labels, low_true_labels = cuda(torch.tensor([0 for _ in range(int(verifier_label.shape[0]))])),cuda(torch.tensor([0 for _ in range(int(verifier_label.shape[0]))]))#cuda(torch.zeros([int(idx.shape[0]/2)])),cuda(torch.zeros([int(idx.shape[0]/2)]))
            low_idx, high_idx, c_idx = 0,0,0
            high_inputs, low_inputs = [],[]
            for k in range(0,output2.shape[0]):
                if verifier_prob[k] >= avg_prob:
                    high_output[high_idx] = output2[k]
                    high_true_labels[high_idx] = verifier_label[k]#cuda(torch.tensor(idx[k].data.tolist(),dtype=torch.int64))#true_labels2[k]#idx[k]#true_labels2[k]
                    high_file.write(str(original_unlabeled[k])+"\t" +str(verifier_prob[k].data.tolist())+"\t"+ str(verifier_label[k].data.tolist())+"\t" + str(true_labels2[k].data.tolist())+"\n")
                    high_count += 1
                    high_idx += 1
                else:
                    low_output[low_idx] = output2[k]
                    low_true_labels[low_idx] = verifier_label[k]#true_labels2[k]
                    low_file.write(str(original_unlabeled[k])+"\t" +str(verifier_prob[k].data.tolist())+"\t"+ str(verifier_label[k].data.tolist())+"\t" + str(true_labels2[k].data.tolist())+"\n")
                    low_count += 1
                    low_idx += 1
            high_true_labels = high_true_labels[:high_idx]
            high_output = high_output[:high_idx]
            low_true_labels = low_true_labels[:low_idx]
            low_output = low_output[:low_idx]
            
            if args.mixup:
                if args.rand_mixup:
                    select_idx = torch.randperm(int(output2.shape[0]/2))
                    to_be_mixed_output = output2[select_idx]
                    to_be_mixed_label = labels2[select_idx]
                    rand_high_idx = 0
                    for i in range(output2.shape[0]):
                        if i not in select_idx:
                            high_output[rand_high_idx] = output2[i]
                            high_labels[rand_high_idx] = labels2[i]
                            rand_high_idx += 1
                else:
                    if args.high_mixup:
                        select_idx = torch.randperm(high_output.shape[0])
                        to_be_mixed_output = high_output
                        to_be_mixed_label = high_labels
                    else:
                        low_labels = smoothing_label(low_true_labels,smoothing_val)
                        select_idx = torch.randperm(low_output.shape[0])
                        to_be_mixed_output = low_output
                        to_be_mixed_label = low_labels
                    output1 = output1[select_idx]
                    labels1 = labels1[select_idx]
                    labels1_onehot = smoothing_label(labels1,smoothing_val)
                    mixup_output = output1 * lam + to_be_mixed_output * (1-lam)
                    mixup_label = labels1_onehot * lam + to_be_mixed_label * (1-lam)
                if args.multigpus:
                    high_logits = model.module.classifier(high_output)
                else:
                    high_logits = model.classifier(high_output)
                high_labels = smoothing_label(high_true_labels,smoothing_val)
                loss2 = torch.mean(torch.sum(-high_labels*torch.log_softmax(high_logits,dim=-1),dim=-1))# *checkterm)
                if args.multigpus:
                    mixup_output = model.module.classifier(mixup_output)
                else:
                    mixup_output = model.classifier(mixup_output)
                if args.task in ('SWAG'): mixup_output = mixup_output.view(-1, model.n_choices)
                mixup_loss = torch.mean(torch.sum(-mixup_label * torch.log_softmax(mixup_output, dim=-1), dim=-1))
                if low_labels.shape[0] == 0:
                    mixup_loss = cuda(torch.tensor(0.))
                if mismatch_labels.shape[0] == 0:
                    discardMixUpLoss = cuda(torch.tensor(0.))
                if high_labels.shape[0] == 0:
                    loss2 = cuda(torch.tensor(0.))
                loss = loss1 + loss2 +  args.mixup_loss_weight*mixup_loss + discardMixUpLoss
            else:
                if args.multigpus:
                    high_logits = model.module.classifier(high_output)
                else:
                    high_logits = model.classifier(high_output)
                high_labels = smoothing_label(high_true_labels,0.3)
                loss2 = torch.mean(torch.sum(-high_labels*torch.log_softmax(high_logits,dim=-1),dim=-1))# *checkterm)
                
                loss = loss1 + loss2 + discardMixUpLoss
                mixup_loss = torch.tensor(0.)
            
            train_loss += loss.item()
            pbar.set_description(f"supervised loss = {(loss1.item())/(i+1):.6f}")
            pbar.update(1)
            # d1_loader.set_description(f"mixup train loss = {(mixup_loss.item() / (i+1)):.6f}")
            pbar2.set_description(f"mixup train loss = {(mixup_loss.item() / (i+1)):.6f}")
            pbar2.update(1)
            d2_loader.set_description(f'unsupervised loss = {loss2.item()/(i+1):.10f}')

            pbar3.set_description(f"total train loss = {(train_loss / (i+1)):.6f}")
            pbar3.update(1)
            pbar4.set_description(f"Discard Mixup Loss = {discardMixUpLoss.item()/(i+1):.6f}")
            pbar4.update(1)
            loss.backward()
            if args.max_grad_norm > 0.:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
        
        discard_info.write(str(discard_per_iter)+"\n")
        discard_info.write(str(total_num)+'\n')
        discard_info.write(str(sum(discard_per_iter)/len(discard_per_iter))+"\n")
        discard_info.write(str(unlabeled_not_used_count)+"\n")
        return train_loss / (len(d1_loader)+len(d2_loader))
    else:
        train_loader = tqdm(load(d1, args.batch_size, True))
        optimizer = AdamW(adamw_params(model), lr=args.learning_rate, eps=1e-8)
        for i, dataset in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, labels = dataset
            guid = inputs[3]
            logit = model(inputs[0],inputs[1],inputs[2])
            prob = F.softmax(logit,dim=-1)
            max_prob,pred_label = torch.max(prob,dim=-1)
            loss = criterion(logit,labels)
            train_loss += loss.item()
            train_loader.set_description(f'train loss = {(train_loss / (i+1)):.6f}')
            loss.backward()
            if args.max_grad_norm > 0.:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
        return train_loss / len(train_loader)


def evaluate(dataset):
    """Evaluates pre-trained model on development set."""

    model.eval()
    eval_loss = 0.
    eval_acc = 0. 
    y_true, y_pred = [], []
    eval_loader = tqdm(load(dataset, args.batch_size, False))
    for i, dataset in enumerate(eval_loader):
        with torch.no_grad():
            inputs, labels = dataset
            output = model(inputs[0],inputs[1],inputs[2])
            if args.ssl:
                if args.task in ('SNLI','MNLI','SICK','RTE','FEVER','QQP','HANS','CrisisMMDINF', 'HumAID'):
                    if args.multigpus:
                        output = model.module.classifier(output[0][:,0])
                    else:
                        output = model.classifier(output[0][:, 0])
                elif args.task in ('SWAG'):
                    output = model.classifier(output[1])
                    output = output.view(-1,model.n_choices)

            for j in range(output.size(0)):
                y_pred.append(output[j].argmax().item())
                y_true.append(labels[j].item())
            loss = criterion(output,labels)
        eval_loss += loss.item()
        eval_loader.set_description(f'eval loss = {(eval_loss / (i+1)):.6f}')
    eval_acc = accuracy_score(y_true, y_pred) * 100.
    return eval_loss / len(eval_loader), eval_acc


model = cuda(Model())
if args.multigpus:
    model = nn.DataParallel(model)
processor = select_processor()
tokenizer = AutoTokenizer.from_pretrained(args.model)
criterion = nn.CrossEntropyLoss()

if args.ssl:
    d1 = TextDataset(args.labeled_train_path,processor)
    d2 = TextDataset(args.unlabeled_train_path,processor)
    print(f'labeled train samples = {len(d1)}')
    print(f'unlabeled train samples = {len(d2)}')
else:
    if args.train_path:
        train_dataset = TextDataset(args.train_path, processor)
        print(f'train samples = {len(train_dataset)}')
if args.dev_path:
    dev_dataset = TextDataset(args.dev_path, processor)
    print(f'dev samples = {len(dev_dataset)}')
if args.test_path:
    test_dataset = TextDataset(args.test_path, processor)
    print(f'test samples = {len(test_dataset)}')


if args.task == 'MNLI':
    test_processor = MNLITESTProcessor()
    # test_processor = globals()[f'MNLITESTProcessor']()
    m_path = "./data/MNLI/multinli_0.9_dev_matched.txt"
    match_dataset = TextDataset(m_path, test_processor)
    mm_path = "./data/MNLI/multinli_0.9_dev_mismatched.txt"
    mismatch_dataset = TextDataset(mm_path,test_processor)


if args.do_train:
    print()
    print('*** training ***')
    best_acc = -float('inf')
    for epoch in range(1, args.epochs + 1):
        if args.ssl:
            train_loss = train(d1=d1, d2=d2, epoch=epoch)
        else:
            train_loss = train(d1=train_dataset, epoch=epoch)
        eval_loss, eval_acc = evaluate(dev_dataset)
        if eval_acc > best_acc:
            best_acc = eval_acc
            torch.save(model.state_dict(), args.ckpt_path)
        print(
            f'epoch = {epoch} | '
            f'train loss = {train_loss:.6f} | '
            f'eval loss = {eval_loss:.6f} | '
            f'eval acc = {eval_acc:.6f} '
        )



if args.do_evaluate:
    if not os.path.exists(args.ckpt_path):
        raise RuntimeError(f'\'{args.ckpt_path}\' does not exist')
    
    print()
    print('*** evaluating ***')

    output_dicts = []
    model.load_state_dict(torch.load(args.ckpt_path))
    model.eval()

    # b/c mnli has two test sets
    if args.task == 'MNLI':
        match_loader = tqdm(load(match_dataset,args.batch_size,False))
        mmatch_loader = tqdm(load(mismatch_dataset,args.batch_size,False))

        for i, (inputs, label) in enumerate(match_loader):
            with torch.no_grad():
                #if args.consistency_learning or args.noisy_label:
                if args.ssl:
                    output = model(inputs[0],inputs[1],inputs[2])
                    if args.task in ('SNLI','MNLI','SICK','RTE'):
                        if args.multigpus:
                            logits = model.module.classifier(output[0][:,0])
                        else:
                            logits = model.classifier(output[0][:,0])
                    elif args.task in ('SWAG'):
                        logits = model.classifier(output[1])
                        logits = logits.view(-1,model.n_choices)
                else:
                    logits = model(inputs[0],inputs[1],inputs[2])
                for j in range(logits.size(0)):
                    probs = F.softmax(logits[j], -1)
                    output_dict = {
                        'index': args.batch_size * i + j,
                        'true': label[j].item(),
                        'pred': logits[j].argmax().item(),
                        'conf': probs.max().item(),
                        'logits': logits[j].cpu().numpy().tolist(),
                        'probs': probs.cpu().numpy().tolist(),
                    }
                    output_dicts.append(output_dict)

        print(f'writing outputs of matched...')

        write_path = args.output_path.replace(".json","")
        write_path = write_path + "_m.json"
        with open(write_path, 'w+') as f:
            for i, output_dict in enumerate(output_dicts):
                output_dict_str = json.dumps(output_dict)
                f.write(f'{output_dict_str}\n')

        y_true = [output_dict['true'] for output_dict in output_dicts]
        y_pred = [output_dict['pred'] for output_dict in output_dicts]
        y_conf = [output_dict['conf'] for output_dict in output_dicts]

        accuracy = accuracy_score(y_true, y_pred) * 100.
        f1 = f1_score(y_true, y_pred, average='macro') * 100.
        confidence = np.mean(y_conf) * 100.

        results_dict = {
            'accuracy': accuracy_score(y_true, y_pred) * 100.,
            'macro-F1': f1_score(y_true, y_pred, average='macro') * 100.,
            'confidence': np.mean(y_conf) * 100.,
        }
        for k, v in results_dict.items():
            print(f'{k} = {v}')

        for i, (inputs, label) in enumerate(mmatch_loader):
            with torch.no_grad():
                #if args.consistency_learning or args.noisy_label:
                if args.ssl:
                    output = model(inputs[0],inputs[1],inputs[2])
                    if args.task in ('SNLI','MNLI','SICK','RTE','CrisisMMDINF', 'HumAID'):

                        if args.multigpus:
                            logits = model.module.classifier(output[0][:,0])
                        else:
                            logits = model.classifier(output[0][:,0])
                    elif args.task in ('SWAG'):
                        logits = model.classifier(output[1])
                        logits = logits.view(-1,model.n_choices)
                else:
                    logits = model(inputs[0],inputs[1],inputs[2])
                for j in range(logits.size(0)):
                    probs = F.softmax(logits[j], -1)
                    output_dict = {
                        'index': args.batch_size * i + j,
                        'true': label[j].item(),
                        'pred': logits[j].argmax().item(),
                        'conf': probs.max().item(),
                        'logits': logits[j].cpu().numpy().tolist(),
                        'probs': probs.cpu().numpy().tolist(),
                    }
                    output_dicts.append(output_dict)

        print(f'writing outputs mismatched...')

        write_path = args.output_path.replace(".json","")
        write_path = write_path + "_mm.json"
        with open(write_path, 'w+') as f:
            for i, output_dict in enumerate(output_dicts):
                output_dict_str = json.dumps(output_dict)
                f.write(f'{output_dict_str}\n')

        y_true = [output_dict['true'] for output_dict in output_dicts]
        y_pred = [output_dict['pred'] for output_dict in output_dicts]
        y_conf = [output_dict['conf'] for output_dict in output_dicts]

        accuracy = accuracy_score(y_true, y_pred) * 100.
        f1 = f1_score(y_true, y_pred, average='macro') * 100.
        confidence = np.mean(y_conf) * 100.

        results_dict = {
            'accuracy': accuracy_score(y_true, y_pred) * 100.,
            'macro-F1': f1_score(y_true, y_pred, average='macro') * 100.,
            'confidence': np.mean(y_conf) * 100.,
        }
        for k, v in results_dict.items():
            print(f'{k} = {v}')

        
    test_loader = tqdm(load(test_dataset, args.batch_size, False))
    for i, (inputs, label) in enumerate(test_loader):
        with torch.no_grad():
            if args.ssl:
                output = model(inputs[0],inputs[1],inputs[2])
                if args.task in ('SNLI','MNLI','SICK','RTE','FEVER','QQP','HANS','CrisisMMDINF', 'HumAID'):
                    if args.multigpus:
                        logits = model.module.classifier(output[0][:,0])
                    else:
                        logits = model.classifier(output[0][:,0])
                elif args.task in ('SWAG'):
                    logits = model.classifier(output[1])
                    logits = logits.view(-1,model.n_choices)
            else:
                logits = model(inputs[0],inputs[1],inputs[2])
            
            for j in range(logits.size(0)):
                # reduce 3-class logits to 2-class logits for HANS
                if args.task == 'HANS':
                    new_logits = cuda(torch.tensor([logits[j][0],logits[j][1]+logits[j][2]]))
                else:
                    new_logits = logits[j]
                probs = F.softmax(new_logits, -1)
                output_dict = {
                    'id': inputs[3][j].item() if isinstance(inputs[3][j], torch.Tensor) else inputs[3][j],
                    'true': label[j].item(),
                    'pred': new_logits.argmax().item(),
                    'conf': probs.max().item(),
                    'logits': logits[j].cpu().numpy().tolist(),
                    'probs': probs.cpu().numpy().tolist(),
                }
                output_dicts.append(output_dict)

    print(f'writing outputs to \'{args.output_path}\'')

    with open(args.output_path, 'w+') as f:
        for i, output_dict in enumerate(output_dicts):
            output_dict_str = json.dumps(output_dict)
            f.write(f'{output_dict_str}\n')

    csv_path = args.output_path.replace(".json", ".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "gold", "pred"])
        for output_dict in output_dicts:
            writer.writerow([output_dict["id"], output_dict["true"], output_dict["pred"]])

    print(f"[csv] wrote predictions to {csv_path}")

    y_true = [output_dict['true'] for output_dict in output_dicts]
    y_pred = [output_dict['pred'] for output_dict in output_dicts]
    y_conf = [output_dict['conf'] for output_dict in output_dicts]

    accuracy = accuracy_score(y_true, y_pred) * 100.
    f1 = f1_score(y_true, y_pred, average='macro') * 100.
    confidence = np.mean(y_conf) * 100.

    results_dict = {
        'accuracy': accuracy_score(y_true, y_pred) * 100.,
        'macro-F1': f1_score(y_true, y_pred, average='macro') * 100.,
        'confidence': np.mean(y_conf) * 100.,
    }
    for k, v in results_dict.items():
        print(f'{k} = {v}')
