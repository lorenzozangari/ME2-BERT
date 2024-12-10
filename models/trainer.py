import numpy as np
import math
import logging

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from models.model import ME2Bert
from .modules import AutoEncoderLoss

from utils.util import evaluate
from pytorch_metric_learning import losses, distances, miners
import os
from tqdm import tqdm
from pytorch_metric_learning.reducers import MeanReducer



class NReducer(MeanReducer):
    def __init__(self, negative_weight=1.5):
        super().__init__()
        self.negative_weight = negative_weight
    
    def pos_pair_reduction(self, losses, indices_tuple):
        return super().pos_pair_reduction(losses, indices_tuple)
    
    def neg_pair_reduction(self, losses, indices_tuple):
        return super().neg_pair_reduction(losses * self.negative_weight, indices_tuple)

class ME2BertTrainer:

    def __init__(self, dataset, args):
        
        self.datasets = dataset
        self.args = args
        n_domain_classes = 2
        latent_dim = 128
        if  self.args.num_no_adv<=self.args.n_epoch:
            print(f'Using adversarial learning after {self.args.num_no_adv} epoch/s')
        
        self.model = ME2Bert(self.args.pretrained_model, self.args.mf_classes, n_domain_classes, self.args.dropout, latent_dim, self.args.transformation, self.args.num_no_adv<=self.args.n_epoch, has_gate=not self.args.no_gate)   
        self.checkpoint = None
        
        self.create_optimizer_and_scheduler_adversarial_training()
        self.init_epoch = -1
        
        self.transformation = args.transformation
        self.model_embedding_dim = self.model.embedding_dim
        self.model = self.model.to(self.args.device)
        self.lambda_con = args.lambda_con 
        for p in self.model.parameters():
            p.requires_grad = True


    def create_optimizer_and_scheduler_adversarial_training(self, checkpoint=None):
 
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.args.lr)
        
        if checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        
        def lr_lambda(epoch: int):
            if epoch >= self.args.num_no_adv:
                tot_epochs_for_calc = self.args.n_epoch - \
                    self.args.num_no_adv
                if tot_epochs_for_calc == 0: return 1
                epoch_for_calc = epoch - self.args.num_no_adv
                p = epoch_for_calc / tot_epochs_for_calc
                decay_factor = 1 / \
                    ((1 + self.args.alpha * p) ** self.args.beta)
            else:
                decay_factor = 1

            return decay_factor

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

    def update_lambda_domain(self, epoch, i_dataloader):
        if epoch >= self.args.num_no_adv:
            tot_steps_for_calc = (
                self.args.n_epoch -
                self.args.num_no_adv) * self.len_dataloader
            curr_steps = float(i_dataloader +
                               (epoch - self.args.num_no_adv) *
                               self.len_dataloader)
            p = curr_steps / tot_steps_for_calc

            return 2 / (1 + math.exp(-self.args.gamma * p)) - 1
        else:
            return 0.0
        
    def update_lambda_con(self, epoch, i_dataloader):
        gamma = self.args.gamma
        min_eps = 5e-2
        if epoch >= self.args.num_no_adv:
            tot_steps_for_calc = (
                self.args.n_epoch -
                self.args.num_no_adv) * self.len_dataloader
            curr_steps = float(i_dataloader +
                            (epoch - self.args.num_no_adv) *
                            self.len_dataloader)
            p = curr_steps / tot_steps_for_calc

            return max(math.exp(-gamma * p), min_eps)
        else:
            return 1.0


    def compute_batch_size(self):
        n_devices = 1
        n_gpus = 1
        
        t_train_batch_size = round(
            self.args.batch_size * len(self.datasets['t_train']) /
            len(self.datasets['s_train']) / n_devices) * n_devices
        
        if t_train_batch_size <= n_devices:
            t_train_batch_size = 2 * n_gpus
            s_train_batch_size = math.ceil(
                len(self.datasets['s_train']) /
                (len(self.datasets['t_train']) // t_train_batch_size))
        else:
            s_train_batch_size = self.args.batch_size

        logging.debug(
            's_train size: %d, s_train batch size: %d, t_train size: %d, t_train batch size: %d'
            % (len(self.datasets['s_train']), s_train_batch_size,
               len(self.datasets['t_train']), t_train_batch_size))

        return s_train_batch_size, t_train_batch_size

    def print_loss(self, loss):
        try:
            return loss.item()
        except:
            return loss

    def compute_weights(self):
        data_name1 = 's_train'
        data_name2 = 't_train'
        n_pos = np.sum(self.datasets[data_name1].mf_labels, axis=0).reshape(-1) + np.sum(self.datasets[data_name2].mf_labels, axis=0).reshape(-1)
        n_pos = np.where(n_pos == 0, 1, n_pos)
        len_labels = len(self.datasets[data_name1].mf_labels) + len(self.datasets[data_name2].mf_labels) 
        weights = (len_labels - n_pos) / n_pos
        weights = torch.tensor(weights, dtype=torch.float)
        return weights

    def compute_domain_adapt_loss(self, data, epoch, mf_loss=True):
        is_adv = (epoch >= self.args.num_no_adv)

        outputs = self.model(data['input_ids'],
                             data['attention_mask'],
                             data['domain_labels'],
                             self.args.lambda_domain,
                             adv=is_adv, emotion_labels=data['emotion_labels'], no_gate=self.args.no_gate)

        loss = 0.0
        loss_domain = 0.0
        if is_adv:
            loss_domain = self.loss_fn_domain(outputs['domain_output'],
                                              data['domain_labels'].squeeze())
            loss +=  loss_domain
        loss_mf = 0.0
        if mf_loss:
            loss_mf = self.loss_fn_mf(outputs['class_output'],
                                      data['mf_labels'])
            loss += loss_mf
            
        loss_trans = 0.0
        if self.transformation:
            loss_trans = self.args.lambda_trans * \
                self.loss_fn_trans(outputs['transformed_output'], outputs['rec_embed'] , outputs['latent_output'])
               
        return loss, loss_mf, loss_domain, loss_trans, outputs['latent_output']
    
    
    
    
    def compute_contrastive_loss(self, miner, loss_con, emb, labels, domain_labels, source_domain_label=0, target_domain_label=1):
        source_mask = domain_labels == source_domain_label
        target_mask = domain_labels == target_domain_label
        source_emb, source_labels = emb[source_mask], labels[source_mask]
        target_emb, target_labels = emb[target_mask], labels[target_mask]
        
        intra_source_hard_pairs = miner(source_emb, source_labels)
        intra_target_hard_pairs = miner(target_emb, target_labels)
        
        intra_loss = loss_con(source_emb, source_labels, intra_source_hard_pairs) + \
                    loss_con(target_emb, target_labels, intra_target_hard_pairs)
        
        return intra_loss
    
    
    def train(self):
        f1 = self.train_con_adv()
        return f1

  

    def train_con_adv(self):

        s_train_batch_size, t_train_batch_size = self.compute_batch_size()
        
        def collate_fn(batch):
            device = self.args.device
                # Sort batch by text length
            batch.sort(key=lambda x: x[0]['length'], reverse=True)
            
            # Group batch elements into separate lists
            idxs = [item[1] for item in batch]
            batch = [item[0] for item in batch]
            inputs_ids = [item['input_ids'] for item in batch]
            attention_mask = [item['attention_mask'] for item in batch]
            mf_labels = None
            if 'mf_labels' in batch[0]:
                mf_labels = torch.stack([item['mf_labels'] for item in batch]).to(device)
                
            domain_labels = None
            if 'domain_labels' in batch[0]:
                domain_labels = torch.stack([item['domain_labels'] for item in batch]).to(device)
            
            emotion_labels = None
            if 'emotion_labels' in batch[0]:
                emotion_labels = torch.stack([item['emotion_labels'] for item in batch]).to(device)
            
            max_len = max([len(ids) for ids in inputs_ids])

            # Pad to uniform length
            padded_input_ids = torch.stack([torch.cat([ids.to(device), torch.zeros(max_len - len(ids), dtype=torch.long).to(device)]) for ids in inputs_ids])
            padded_attention_mask = torch.stack([torch.cat([mask.to(device), torch.zeros(max_len - len(mask), dtype=torch.long).to(device)]) for mask in attention_mask])

            # Stack other components
            inputs_ids = padded_input_ids.to(device)
            attention_mask = padded_attention_mask.to(device)
            return {
                'input_ids': inputs_ids,
                'attention_mask': attention_mask,
                'mf_labels': mf_labels,
                'domain_labels': domain_labels,
                'emotion_labels': emotion_labels
            }, torch.tensor(idxs)


        self.dataloaders = {}
        self.dataloaders['s_train'] = DataLoader(
            dataset=self.datasets['s_train'],
            batch_size=s_train_batch_size,
            shuffle=True,
            drop_last=True, collate_fn=collate_fn)
         
        self.dataloaders['t_train'] = DataLoader(
            dataset=self.datasets['t_train'],
            batch_size=t_train_batch_size,
            shuffle=True,
            drop_last=True, collate_fn=collate_fn)
        self.len_dataloader = min(len(self.dataloaders['s_train']),
                                  len(self.dataloaders['t_train']))
    
        if self.args.weighted_loss:
            self.loss_fn_mf = torch.nn.BCEWithLogitsLoss(
            pos_weight=self.compute_weights()).to(self.args.device)  # moral loss
        else:
            self.loss_fn_mf = torch.nn.BCEWithLogitsLoss().to(self.args.device)
        
        if self.args.contrastive:
            negative_weight =1.5
            reducer = NReducer(negative_weight=negative_weight)
            self.miner = miners.MultiSimilarityMiner()
            
        self.loss_fn_domain = torch.nn.CrossEntropyLoss().to(
            self.args.device) # domain loss
        
        self.loss_fn_trans = AutoEncoderLoss(self.args.device).to(self.args.device) if self.args.transformation else None                           


        logging.debug('lambda_trans = %f' %
                      (self.args.lambda_trans))

        # training

        if self.checkpoint:
            best_accu = self.checkpoint['best_accu']
            best_epoch = self.checkpoint['best_epoch']
        else:
            best_accu = 0.0
            best_epoch = 0

        mn = self.args.pretrained_model.split('/')[0].split('-')[0]
        for epoch in range(self.init_epoch+1, self.args.n_epoch):

            self.model.train()

            is_adv = (epoch >= self.args.num_no_adv)

            data_source_iter = iter(self.dataloaders['s_train'])
            if is_adv or self.args.contrastive:
                data_target_iter = iter(self.dataloaders['t_train'])
            for i in tqdm(range(self.len_dataloader), desc=f'Epoch {epoch}'):
                # update lambda_domain
                self.args.lambda_domain = self.update_lambda_domain(
                    epoch, i)
                print(
                    '\r epoch: %d, [iter: %d / all %d], lambda_domain = %f' %
                    (epoch, i + 1, self.len_dataloader,
                     self.args.lambda_domain))

                # get data
                data_source, _ = next(data_source_iter)
                for k, v in data_source.items():
                    if data_source[k] != None:
                        data_source[k] = data_source[k].to(self.args.device)

                if is_adv or self.args.contrastive:
                    data_target, _ = next(data_target_iter)
                    for k, v in data_target.items():
                        if data_target[k] != None:
                            data_target[k] = data_target[k].to(self.args.device)

                self.model.zero_grad()

                
                # training model using source data
                s_loss, s_loss_mf, s_loss_domain, s_loss_trans, s_emb_trans = self.compute_domain_adapt_loss(
                    data_source, epoch, mf_loss=True)
                loss = s_loss.clone()
                # training model using target data
                t_loss, t_loss_mf, t_loss_domain, t_loss_trans = 0.0, 0.0, 0.0, 0.0
                loss_con, lambda_con = 0.0, 0.0
                if is_adv:
                    t_loss, t_loss_mf, t_loss_domain, t_loss_trans, t_emb_trans = self.compute_domain_adapt_loss(
                        data_target, epoch, mf_loss=False)
                    loss = loss + (s_train_batch_size /
                                t_train_batch_size) * t_loss + s_loss_trans + (s_train_batch_size / t_train_batch_size) * t_loss_trans
            
                    if self.args.contrastive:
                        emb = torch.cat((s_emb_trans, t_emb_trans), dim=0)
                        label = torch.cat((data_source['emotion_labels'], data_target['emotion_labels']), dim=0)
                        domain_label = torch.cat((torch.zeros(s_emb_trans.shape[0]), torch.ones(t_emb_trans.shape[0])))
                        lambda_mar = self.update_lambda_con(epoch, i)
                        self.args.temperature = lambda_mar
                        self.loss_con = losses.TripletMarginLoss(distance= distances.CosineSimilarity(), margin=self.args.temperature, reducer=reducer)#margin=0.1) #NTXentLoss(temperature=self.args.temperature, distance = distances.CosineSimilarity())
                        loss_con = lambda_mar* self.args.lambda_con * self.compute_contrastive_loss(self.miner, self.loss_con, emb, label, domain_label)
                        loss = loss +  loss_con
                        
                elif self.args.contrastive:
                        _, t_loss_mf,_, t_loss_trans, t_emb_trans = self.compute_domain_adapt_loss(
                        data_target, epoch, mf_loss=False)
                        
                        loss  += (s_train_batch_size /
                                t_train_batch_size) * (t_loss_mf + t_loss_trans) 
                       
                        emb = torch.cat((s_emb_trans, t_emb_trans), dim=0)
                        label = torch.cat((data_source['emotion_labels'], data_target['emotion_labels']), dim=0)
                        
                        domain_label = torch.cat((torch.zeros(s_emb_trans.shape[0]), torch.ones(t_emb_trans.shape[0])))
                        lambda_mar = self.update_lambda_con(epoch, i)
                        self.args.temperature = lambda_mar
                        self.loss_con = losses.TripletMarginLoss(distance= distances.CosineSimilarity(), margin=self.args.temperature, reducer=reducer)#margin=0.1) #NTXentLoss(temperature=self.args.temperature, distance = distances.CosineSimilarity())
                        loss_con = lambda_con * self.compute_contrastive_loss(self.miner, self.loss_con, emb, label, domain_label)
 
                        loss = loss +  loss_con
                
                loss.backward()
                self.optimizer.step()

                # print loss values
                print(    '\r epoch: %d, [iter: %d / all %d], total_loss: %.5f, total_s_loss: %.5f, total_t_loss: %.5f\ns_loss_mf: %.5f, t_loss_mf: %.5f, s_loss_domain: %.5f, s_loss_trans: %.5f\nt_loss_domain: %.5f, t_loss_trans: %.5f, loss_con: %.5f, lambda_con: %.5f'
                    % (epoch, i + 1, self.len_dataloader,
                       self.print_loss(loss), self.print_loss(s_loss),
                       (s_train_batch_size / t_train_batch_size) *
                       self.print_loss(t_loss), self.print_loss(s_loss_mf), self.print_loss(t_loss_mf),
                       self.print_loss(s_loss_domain),
                       self.print_loss(s_loss_trans),
                       self.print_loss(t_loss_domain),
                       self.print_loss(t_loss_trans),
                       self.print_loss(loss_con), self.print_loss(self.args.temperature)))
                
                del loss
                
            
            print(f'\nepoch: {epoch}')
            logging.info(f'\nepoch: {epoch}')
            
            with torch.no_grad():
                accu,_,_ = evaluate(self.datasets['s_val'], self.datasets['t_val'] if 't_val' in self.datasets else None,
                                self.args.batch_size,
                                model=self.model,
                                is_adv=is_adv, device=self.args.device, no_gate=self.args.no_gate)
                print('Macro F1 of the dataset: %f' % (accu))
                logging.info('Macro F1 of the dataset: %f' % (accu))

                if accu > best_accu and epoch >= self.args.num_epoch_save:
                    best_accu = accu
                    best_epoch = epoch
                    print('Saving best model : Epoch :{:.3f} - Accuracy {:.3f}'.format(best_epoch, best_accu))
                    logging.info('Saving best model : Epoch :{:.3f} - Accuracy {:.3f}'.format(best_epoch, best_accu))
                    torch.save(self.model,
                                os.path.join(self.args.output_dir, f'best_model_{self.args.seed}_{mn}.pth'))
                    torch.save(self.model.state_dict(),
                                os.path.join(self.args.output_dir, f'best_model_weights_{self.args.seed}_{mn}.pth'))

                print('Saving checkpoint....')    
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_accu': best_accu, 'best_epoch':best_epoch
                        }, os.path.join(self.args.checkpoint_dir, f'checkpoint_{self.args.seed}_{mn}.pt'))
            
            # update lr scheduler
            self.scheduler.step()

        
        logging.info('============ Training Summary ============= \n')
        
        logging.info('Best Macro F1 of the VAL dataset: %f' %
                    (best_accu))
        logging.info(f'Best Macro F1 at epoch {best_epoch}')
        logging.info('Corresponding model was save in ' +
                    os.path.join(self.args.output_dir, f'best_model_{self.args.seed}_{mn}.pth'))
        

        return best_accu