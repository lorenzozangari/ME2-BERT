import pandas as pd
from transformers import AutoTokenizer
import numpy as np
from .data_loader import ME2Data
import os
from .split import train_test_val_split_stratified
import ast
from pathlib import Path
import torch
import json
from sklearn.cluster import KMeans
import pandas as pd
import re
import emoji

_MORAL_FOUNDATIONS =  {'care':'harm',
                        'fairness':'cheating', 
                        'loyalty':'betrayal', 
                        'authority':'subversion', 
                        'purity':'degradation'}

_SOURCE_DOMAIN = 0.0
_TARGET_DOMAIN = 1.0

_EMOTION_COLUMNS = ['anticipation', 'trust', 'disgust', 'joy', 'optimism', 'surprise', 'love', 'anger', 'sadness', 'pessimism', 'fear', 'no emotion']


_HOURGLASS = {
'pleasantness1': ['joy'],
'pleasantness2': ['sadness'],
'attention1': ['anticipation'],
'attention2': ['surprise'],
'sensitivity1': ['anger'],
'sensitivity2': ['fear'],
'aptitude1': ['trust'],
'aptitude2': ['disgust']
}

_DIMENSION_MAP = {
    'pleasantness1': 0,
    'pleasantness2': 1,
    'attention1': 2,
    'attention2': 3,
    'sensitivity1': 4,
    'sensitivity2': 5,
    'aptitude1': 6,
    'aptitude2': 7,
    'no_emotion': 8 
}

def load_raw_data(path='./e2mocase_full.csv'):
    return pd.read_csv(path, sep='\t')


def load(args, clean_data=False):
    df = load_raw_data(args.data_path)
    df['event'] = df['event'].apply(ast.literal_eval)
    moral_columns = list(_MORAL_FOUNDATIONS.keys()) 
    moral_foundations = _MORAL_FOUNDATIONS
    emotion_columns = _EMOTION_COLUMNS
    df = df.reset_index()
    
    if moral_columns is not None and moral_foundations is not None:
        for key, value in moral_foundations.items():
            df[key] = (df[key] + df[value])/2 # Incorporate the polarities of the MFT
    
    # If the paragraphs contains events it belong to the target domain; otherwise, it belongs to the source domain
    df['domain'] = _TARGET_DOMAIN
    df['domain'] = df.apply(lambda row: _SOURCE_DOMAIN if len(row['event']) <= 0 else row['domain'], axis=1)
    
    # Automatic set the number of classes for moral value prediction
    if args.mf_classes == -1:
        args.mf_classes = len(moral_columns)
        print(f'Setting number of MF classes to {args.mf_classes}')
        
    assert args.mf_classes == len(moral_columns), "Mismatch between number of moral labels and data!"
    
    
    # build labels for contrastive learning
    pos, emo_labels = build_labels(df[emotion_columns], 'emo_label')
    df[emo_labels] = pos
    
    # build text featurers
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model,
                                                use_fast=True,
                                                local_files_only=False)
        
    def enrich_input(row):
        if len(row['event'])>0:
            textual_representation = ""
            for i, event in enumerate(row['event']):
                mention = event['mention']
                if event['entities'] is not None and len(event['entities'])>0:
                    entities = ", ".join([f'{entity} with role {role}' for entity, role in event['entities'].items()])
                    textual_representation += f"Event {i}: The mention is '{mention}', among : {entities}."
                        
                else:
                    textual_representation += f"Event {i}: The mention is '{mention}', with no entities involved.\n"
                
            return textual_representation
        else:
            return row['text']

        
    df['e_text'] = df.apply(enrich_input, axis=1)
    if clean_data:
        print('Cleaning text....')
        df['e_text'] = df['e_text'].apply(clean_text)
    
    
    encodings = df['e_text'].apply(tokenizer,
                                truncation=not args.no_truncation,
                                max_length=args.max_seq_len,
                                padding=args.padding).tolist()

    
    df['input_tokenized'] = encodings
    
    dataset_dict = train_test_val_split_stratified(df)
    
    datasets = {}
    for k, v in dataset_dict.items():
        datasets[k] = ME2Data(v['input_tokenized'].values, v[moral_columns].values, v['domain'].values, v[emo_labels].values)
    if args.save_data:
        s_path = os.path.join('./data','enc') 
        Path(s_path).mkdir(parents=True, exist_ok=True)
        torch.save(datasets,os.path.join(s_path, f"model_{args.seed}.pt"))
        print(f'Data saved to {s_path}')
    
    return datasets




def build_labels(emo_df, label='emo_label'):

    df = emo_df.copy()

    def calculate_hourglass(row, mapping):
        scores = {dimension: 0 for dimension in mapping.keys()}
        
        for dimension, emotions in mapping.items():
            scores[dimension] = sum(row.get(emotion, 0) for emotion in emotions)
        
        return scores

    hourglass_mapping = _HOURGLASS
    scores_list = df.apply(lambda row: calculate_hourglass(row, hourglass_mapping), axis=1)
    scores_df = pd.DataFrame(scores_list.tolist(), index=df.index)
    
    def normalize_dimension(df, dimension):
        min_score = df[dimension].min()
        max_score = df[dimension].max()
        
        def normalize_score(score, min_score, max_score):
            if max_score - min_score == 0:
                return 0
            return (score - min_score) / (max_score - min_score)
        
        return df[dimension].apply(lambda x: normalize_score(x, min_score, max_score))

    for dimension in hourglass_mapping.keys():
        scores_df[dimension] = normalize_dimension(scores_df, dimension)
    
    dimension_map = _DIMENSION_MAP

    def dominant_dimension(row):
        max_value = row.max()
        if max_value == 0:
            return 'no_emotion'
        return row.idxmax()
    
    def map_labels(new_labels):
        new_labels = np.copy(new_labels.values)  
        new_labels[(new_labels == 0) | (new_labels == 1)] = 0
        new_labels[(new_labels == 2) | (new_labels == 3)] = 1
        new_labels[(new_labels == 4) | (new_labels == 5)] = 2
        new_labels[(new_labels == 6) | (new_labels == 7)] = 3
        new_labels[new_labels == 8] = 4
        return new_labels
   
    df[label] = scores_df.apply(dominant_dimension, axis=1)
    
    df[label] = df[label].map(dimension_map)
    
    return map_labels(df[label]), label



def remove_urls(text):
    return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

def substitute_mentions(text):
    return re.sub(r'@\w+', '@user', text)

def remove_hashtags(text):
    return re.sub(r'#\w+', '', text)

def substitute_emojis(text):
    return emoji.demojize(text)

def remove_non_ascii(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

def clean_text(text):
    text = remove_urls(text)
    text = substitute_mentions(text)
    text = remove_hashtags(text)
    text = substitute_emojis(text)
    text = remove_non_ascii(text)
    return text




