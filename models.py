from pModules.utils import DatasetHandler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np 

from time import time
from tqdm import tqdm
import pickle as pk

class BahdanauAttention(nn.Module) :
    def __init__(self, hidden_size) :
        super(BahdanauAttention, self).__init__()
        self.Wh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Ws = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V = nn.Linear(hidden_size, 1, bias=False)

        nn.init.xavier_uniform_(self.Wh.weight)
        nn.init.xavier_uniform_(self.Ws.weight)

    def forward(self, H, S, mask=None, only_A=False, activation='softmax') :
        activation = F.softmax if activation=='softmax' else torch.sigmoid

        Wh = self.Wh(H).unsqueeze(-2)
        Ws = self.Ws(S).unsqueeze(-3)
        A_score = self.V(torch.tanh(Wh + Ws)).squeeze(-1)
        if mask is not None :
            A_score = A_score.masked_fill(mask, float('-inf'))
        A_dist = activation(A_score, dim=-1)
        
        if not only_A :
            rSeq = torch.bmm(A_dist, H)
            return rSeq
        else :
            return A_score, A_dist

class NovelNet(nn.Module) :
    features = ['intro', 'read', 'real_read', 'read_duration', 'time_diff', 'novel_count', 'temporal_gaps']

    def __init__(self, datasetHandler:DatasetHandler, 
                 mode:str='base',
                 novel_embedding_size:int=128, 
                 feature_embeddung_size:int=32, 
                 gru_hidden_size:int=128):
     
        super(NovelNet, self).__init__()
        self.datasetHandler = datasetHandler
        self.mode = mode
        self.optimizer = None
        self.weight_penalty = 12

        self.novels_vocab_size = datasetHandler.novels_vocab_size
        self.n_items_of_features = datasetHandler.n_items_of_features
        self.novel_embedding_size = novel_embedding_size
        self.feature_embeddung_size = feature_embeddung_size
        self.gru_hidden_size = gru_hidden_size

        self.novel_embedding = nn.Embedding(self.novels_vocab_size, novel_embedding_size, padding_idx=0)
        self.feature_embeddings = nn.ModuleDict()
        for feature in NovelNet.features : 
            feature_emb = nn.Embedding(self.n_items_of_features[feature]+1, feature_embeddung_size, padding_idx=0)
            self.feature_embeddings[feature] = feature_emb

        gru_input_size = novel_embedding_size + len(NovelNet.features) * feature_embeddung_size
        self.gru_sequence_enc = nn.GRU(gru_input_size, int(gru_hidden_size/2), bidirectional=True, batch_first=True)
        self.gru_sequence_enc.all_weights[0][0] = nn.init.xavier_uniform
        self.gru_sequence_enc.all_weights[0][1] = nn.init.xavier_uniform

        self.newNovelAttention = BahdanauAttention(gru_hidden_size,gru_hidden_size,gru_hidden_size)
        self.consumedAttention = BahdanauAttention(gru_hidden_size,gru_hidden_size,gru_hidden_size)

    def forward(self, input_data, train_flag:bool=True, drop_out:bool=False) :
        batch_size = input_data['novel_ids'].shape[0]
        encodedNovels = self._get_encodedNovels(input_data['novel_ids'])

        pad_mask = input_data['novel_ids'].eq(0).unsqueeze(1)
        newNovel_mask = torch.bmm((input_data['novel_ids']>0).float().unsqueeze(1), encodedNovels).squeeze(1)
        consumedNovel_mask = input_data['novel_last_position_mask'].eq(0).unsqueeze(1)

        embedded_novels = self.novel_embedding(input_data['novel_ids'])
        embedded_features = [embedded_novels]
        for feature in NovelNet.features :
            embedded_feature = self.feature_embeddings[feature](input_data[feature])
            embedded_features.append(embedded_feature)
        sequences = torch.cat(embedded_features, dim=-1)

        if drop_out : 
            sequences = F.dropout(sequences, p=0.5, training=train_flag)

        hidden = torch.zeros(2, batch_size, self.gru_hidden_size//2)
        encoded_seq, bidirection_info = self.gru_sequence_enc(sequences, hidden)

        if drop_out : 
            encoded_seq = F.dropout(encoded_seq, p=0.5, training=train_flag)
            bidirection_info = F.dropout(bidirection_info, p=0.5, training=train_flag)

        bidirection_info = bidirection_info.transpose(0, 1).reshape(batch_size, -1).unsqueeze(1)
        
        rSeq = self.newNovelAttention(encoded_seq, bidirection_info, mask=pad_mask)
        newNovel_score = torch.matmul(rSeq.squeeze(1), self.novel_embedding.weight.T)

        newNovel_score = newNovel_score.masked_fill(newNovel_mask.bool(), float('-inf'))
        newNovel_prob = F.softmax(newNovel_score, dim=-1)

        consumedNovel_A_score, consumedNovel_A_dist = self.consumedAttention(encoded_seq, bidirection_info, mask=consumedNovel_mask, only_A=True)
        consumedNovel_A_dist_sigmoid = torch.sigmoid(consumedNovel_A_score)
        if self.mode == 'base' :
            consumedNovel_A_dist = consumedNovel_A_dist if train_flag else consumedNovel_A_dist_sigmoid    

        consumedNovel_prob = torch.bmm(consumedNovel_A_dist, encodedNovels).squeeze(1)

        switch = 0 if self.mode == 'base' else 1
        prob = ((0.1**switch)*newNovel_prob) + ((0.9**switch)*consumedNovel_prob)
        
        return prob, consumedNovel_A_dist_sigmoid

    def _get_encodedNovels(self, novel_ids) :
        batch_size, batch_len = novel_ids.size()
        vocab_size = self.novels_vocab_size
    
        encodedNovels = torch.zeros(batch_size, batch_len, vocab_size)
        encodedNovels.scatter_(2, novel_ids.unsqueeze(2), 1.)
        encodedNovels.requires_grad=False

        return encodedNovels 

    def fit(self, lr=0.001, epochs=1, batch_size=10, drop_out=False, 
            batch_shuffle=True, auto_save=True, print_acc=False) :
        optimizer = optim.Adam(self.parameters(), lr=lr) 
        datasetLoader = self.datasetHandler.callDatasetLoader(batch_size=batch_size, shuffle=batch_shuffle)
        for epoch in range(epochs) :
            batch_loss, batch_acc = 0, 0
            iteration = 0
            start_time = time()
            for data in tqdm(datasetLoader) : 
                prob, consumedNovel_A_dist_sigmoid = self(data, drop_out=drop_out)

                Y = data['target_novel_id'].reshape(-1)

                if self.mode == 'modified' :
                    loss = F.cross_entropy(prob, Y, ignore_index=0)
                    if print_acc :
                        preds = prob.argmax(axis=-1).reshape(-1)
                        acc = sum(preds==Y)/len(preds)
                else :
                    A_listwise = (prob+1e-8).log()
                    loss_listwise = F.nll_loss(A_listwise, Y, ignore_index=0)
                    
                    novels_mask = data['novel_last_position_mask']
                    valid_label_flag = (data['weight_label'] != -100).reshape((-1, 1))

                    A_pointwise = consumedNovel_A_dist_sigmoid.reshape((-1, 1))
                    weight_label = (data['weight_label'] * novels_mask).reshape((-1, 1))
    
                    valid_A_pointwise = A_pointwise[valid_label_flag]
                    valid_weight_label = weight_label[valid_label_flag]

                    weight_loss = nn.BCELoss()(valid_A_pointwise, valid_weight_label)
                    loss_pointwise = (weight_loss * self.weight_penalty)            

                    loss = loss_listwise + loss_pointwise
                    if print_acc :
                        acc = self.basePredict(prob, consumedNovel_A_dist_sigmoid, Y, data['novel_ids'])

                batch_loss += loss
                batch_acc += acc if print_acc else 0
                iteration += 1

                optimizer.zero_grad()   
                loss.backward()
                optimizer.step()

            end_time = time()
            time_cost = end_time - start_time
            batch_loss /= iteration
            batch_acc /= iteration

            if not print_acc :
                print(f"epoch.{epoch} :: loss.{batch_loss}, time.{time_cost}")
            else : 
                if (not epoch) and (self.mode=='base') :
                    print(f"train-acc of baseModel is not correct")
                print(f"epoch.{epoch} :: loss.{batch_loss}, acc.{batch_acc}, time.{time_cost}")  

            if auto_save :
                with open(f'modelLog/{self.mode}Model+epoch{epoch}.p', 'wb') as f :
                    pk.dump(self, f)
                    
    def evaluation(self, testsetHandler:DatasetHandler, method:str='MRR@1', batch_size:int=0) :
        testsetLoader = testsetHandler.callDatasetLoader(batch_size=batch_size)
        accuracy = 0
        for iter, testset in enumerate(tqdm(testsetLoader)) :
            prob, consumedNovel_A_dist_sigmoid = self(testset, train_flag=False)
            Y = testset['target_novel_id'].reshape(-1)
            
            if method == 'MRR@1' :
                preds = prob.argmax(axis=-1).reshape(-1)
                accuracy += sum(preds==Y)/len(preds)

            else :
                crit = int(method.split('@')[-1])
                preds = prob.argsort(axis=-1, descending=True)
                acc = 0
                for l_idx, pred in enumerate(preds) :
                    for idx in range(crit) :
                        if pred[idx] == Y[l_idx] :
                            acc += 1/(idx+1)
                            break
                accuracy += acc/len(preds)
        
        print(f"{self.mode}Model Test Evaluation:: {accuracy/(iter+1)}")

    def basePredict(self, prob, consumedNovel_A_dist_sigmoid, Y, novel_ids) :

        encodedNovels = self._get_encodedNovels(novel_ids)
        consumedNovel_prob = torch.bmm(consumedNovel_A_dist_sigmoid, encodedNovels).squeeze(1)
        
        l_max_prob_indices = prob.argmax(axis=-1).reshape(-1,1)
        row_indices = np.arange(prob.size(0)).reshape(-1, 1)
        listwise_max_probs = prob[row_indices, l_max_prob_indices]

        p_max_prob_indices = consumedNovel_prob.argmax(axis=-1).reshape(-1,1)
        row_indices = np.arange(consumedNovel_prob.size(0)).reshape(-1,1)
        pointwise_max_probs = consumedNovel_prob[row_indices, p_max_prob_indices]
        preds = list()
        for i in range(len(listwise_max_probs)) :
            if listwise_max_probs[i] > pointwise_max_probs[i] :
                preds.append(l_max_prob_indices[i])
            else : 
                preds.append(p_max_prob_indices[i])

        preds = torch.tensor(preds).reshape(-1)
        acc = sum(preds==Y)/len(preds)
        return acc
