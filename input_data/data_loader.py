import torch
from torch.utils.data import Dataset



class ME2Data(Dataset):

    def __init__(self, encodings, mf_labels=None, domain_labels=None, emotion_labels=None):

        self.encodings = encodings
        self.mf_labels = mf_labels
        self.domain_labels = domain_labels
        self.feat_embed = None
        self.emotion_labels = emotion_labels
        
    def __getitem__(self, idx):
        
        item = {
            key: torch.tensor(val, dtype=torch.long)
            for key, val in self.encodings[idx].items()
        }
        
        item['length'] = item['attention_mask'].sum().item()
        
        if self.mf_labels is not None:
            item['mf_labels'] = torch.tensor(self.mf_labels[idx],
                                             dtype=torch.float)
        if self.domain_labels is not None:
        
            item['domain_labels'] = torch.tensor(self.domain_labels[idx],
                                                 dtype=torch.long)
        
        if self.emotion_labels is not None:
            item['emotion_labels'] = torch.tensor(self.emotion_labels[idx],
                                              dtype=torch.float)

                
        return item, idx
    

    def __len__(self):
        return len(self.encodings)
