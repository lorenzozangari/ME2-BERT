import torch
import transformers
from transformers import AutoModel
from .grad_rev_fn import ReverseLayerF
from .modules import FFClassifier,AutoEncoder, GatedCombination
transformers.logging.set_verbosity_error() 

class ME2Bert(torch.nn.Module):


    def __init__(self,
                 pretrained_dir: str,
                 n_mf_classes: int,
                 n_domain_classes: int,
                 dropout_rate: float, latent_dim: int =128,
                 has_trans: bool = False,
                 has_adv: bool =True,  has_gate: bool = True):

        super(ME2Bert, self).__init__()
        self.n_mf_classes = n_mf_classes
        self.n_domain_classes = n_domain_classes
        self.has_trans = has_trans
        self.has_adv = has_adv
        self.feature = AutoModel.from_pretrained(pretrained_dir)
        self.bert_frozen =  AutoModel.from_pretrained(pretrained_dir)
        for param in self.bert_frozen.parameters():
            param.requires_grad = False

        self.embedding_dim = self.feature.embeddings.word_embeddings.embedding_dim
        self.emotion_dim = 5
        self.gated_combination = GatedCombination(embedding_dim=self.embedding_dim) if has_gate else None
        
        self.trans_module = AutoEncoder(self.embedding_dim, 256, latent_dim) if self.has_trans else None
        
        
        if self.has_adv:
            initial_dim = self.embedding_dim + self.n_domain_classes + self.emotion_dim
        else:
            initial_dim = self.embedding_dim + self.emotion_dim
        
        self.mf_classifier = FFClassifier( initial_dim, latent_dim, self.n_mf_classes, dropout_rate)
    
        self.domain_classifier = FFClassifier(self.embedding_dim, latent_dim,
                                              self.n_domain_classes,
                                              dropout_rate)

    def gen_feature_embeddings(self, input_ids, att_mask):

        feature = self.feature(input_ids=input_ids, attention_mask=att_mask)
        
        return feature.last_hidden_state, feature.pooler_output

    def forward(self,
                input_ids,
                att_mask,
                domain_labels,
                lambda_domain,
                adv=True, emotion_labels=None, emotion_weights=None, no_gate=False, CLS=True):
       
        last_hidden, pooler_output = self.gen_feature_embeddings(
            input_ids, att_mask)
        
        with torch.no_grad():
            frozen_output =  self.bert_frozen(input_ids=input_ids, attention_mask=att_mask)
        
        if not CLS or pooler_output is None:
            input_mask_expanded = att_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            pooler_output = (last_hidden * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)

            frozen_output = (frozen_output.last_hidden_state * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)
        else:
            frozen_output = frozen_output.pooler_output
            
        device = pooler_output.device
        latent_output = None
        rec_embeddings = None
        if self.has_trans:
            rec_embeddings = pooler_output
            latent_output, pooler_output = self.trans_module(rec_embeddings) 
            if not no_gate:
                gated_output = self.gated_combination(frozen_output, pooler_output)
            else:
                gated_output = pooler_output
        
        domain_feature = torch.nn.functional.one_hot(
            domain_labels, num_classes=self.n_domain_classes).squeeze(1)
        emotion_features = None        
        if emotion_labels is not None:
            emotion_features = torch.nn.functional.one_hot(emotion_labels.long(), num_classes=self.emotion_dim).squeeze(1)
        if emotion_weights is not None:
            emotion_features = emotion_weights
        if  self.has_adv and emotion_features is not None:
            emotion_features = emotion_features[:gated_output.shape[0],:]
            class_output = torch.cat((gated_output, domain_feature, emotion_features), dim=1)
        
        elif self.has_adv and emotion_features is None:
            emotion_features = torch.zeros(gated_output.shape[0], self.emotion_dim).to(device)
            class_output = torch.cat((gated_output, domain_feature, emotion_features), dim=1)
        
        else:
            class_output = torch.cat((gated_output, torch.zeros(gated_output.shape[0],self.emotion_dim).to(device)), dim=1)
        

        class_output = self.mf_classifier(class_output)
        
        domain_output = None
        if adv:
            domain_output = self.domain_classifier(
                ReverseLayerF.apply(gated_output, lambda_domain))
        
        return {
            'class_output': class_output,
            'domain_output': domain_output,
            'rec_embed': rec_embeddings,
            'transformed_output': pooler_output,
            'latent_output': latent_output
        }
