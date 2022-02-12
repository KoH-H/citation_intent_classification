# -*t coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import AutoModel
import numpy as np
import torch.nn.functional as F
from math import sqrt


class CNNBert(nn.Module):
    def __init__(self, emb_size):
        super(CNNBert, self).__init__()
        filter_sizes = [2, 3, 4]
        num_filters = 4
        self.conv1 = nn.ModuleList([nn.Conv2d(3, num_filters, (K, emb_size)) for K in filter_sizes])
        # self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, 768)

    def forward(self, bert_in):
        x = (bert_in[7], bert_in[9], bert_in[12])
        x = torch.stack(x, dim=1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        # x = self.dropout(x)
        x = self.fc(x)
        return x


class AttentionLayer(nn.Module):
    def __init__(self):
        super(AttentionLayer, self).__init__()
        self.q_linear = nn.Linear(768, 768, bias=False)
        self.k_linear = nn.Linear(768, 768, bias=False)
        self.v_linear = nn.Linear(768, 768, bias=False)
        self._norm_fact = 1 / sqrt(768)

    def forward(self, b_in, mask):
        q = self.q_linear(b_in)
        k = self.k_linear(b_in)
        v = self.v_linear(b_in)

        mask = mask.unsqueeze(2)
        attention = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact
        attention = attention.masked_fill(mask == 0, float('-inf'))
        attention = F.softmax(attention, dim=-1)
        attention = torch.bmm(attention, v)
        pre = attention[:, 0, :]
        return pre


    # mask = mask.unsqueeze(2)
    # att_w = att_w.masked_fill(mask == 0, float('-inf'))
    # att_w = F.softmax(att_w, dim=1)


class NumModel(nn.Module):
    def __init__(self, name, temp=0.2):
        super(NumModel, self).__init__()
        self.model = AutoModel.from_pretrained(name)
        self.temp = temp
        self.fc1 = nn.Linear(768 * 3, 768)
        self.fc = nn.Linear(768, 6)
        self.drop = nn.Dropout(0.3)
        self.labelfc = nn.Linear(14, 192)
        self.labelfc2 = nn.Linear(192, 384)

        self.au_task_fc1 = nn.Linear(768, 5)

        self.ori_word_atten = nn.Linear(768, 384)
        self.ori_tanh = nn.Tanh()
        self.ori_word_weight = nn.Linear(384, 1, bias=False)

        self.re_word_atten = nn.Linear(768, 384)
        self.re_tanh = nn.Tanh()
        self.re_word_weight = nn.Linear(384, 1, bias=False)

        self.des_word_atten = nn.Linear(768, 384)
        self.des_tanh = nn.Tanh()
        self.des_word_weight = nn.Linear(384, 1, bias=False)

    def generate_sen_pre(self, sen, tp):
        bert_output = self.model(sen['input_ids'], attention_mask=sen['attention_mask'], output_hidden_states=True)
        sen = self.get_sen_att(sen, bert_output, tp, sen['attention_mask'])
        return sen


    # original
    def forward(self, x1, **kwargs):
        input_ids = x1['input_ids']
        attention_mask = x1['attention_mask']
        bert_output = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        ori_sen_pre = self.get_sen_att(x1, bert_output, 'ori', attention_mask)
        if self.training:
            onum2vec = self.labelfc(kwargs['t_one'])
            onum2vec = self.labelfc2(onum2vec)

            # Obtain the representation vector for the classification learning branch
            r_ids = kwargs['r_sen']['input_ids']
            r_attention_mask = kwargs['r_sen']['attention_mask']
            r_bert_output = self.model(r_ids, attention_mask=r_attention_mask, output_hidden_states=True)
            re_sen_pre = self.get_sen_att(kwargs['r_sen'], r_bert_output, 're', r_attention_mask)

            rnum2vec = self.labelfc(kwargs['r_one'])
            rnum2vec = self.labelfc2(rnum2vec)

            # Get the representation vector for the auxiliary task
            s_ids = kwargs['s_sen']['input_ids']
            s_attention_mask = kwargs['s_sen']['attention_mask']
            s_bert_output = self.model(s_ids, attention_mask=s_attention_mask, output_hidden_states=True)
            ausec_sen_pre = self.get_sen_att(kwargs['s_sen'], s_bert_output, 'ori', s_attention_mask)

            ori_sen_pre = torch.cat((ori_sen_pre, onum2vec), dim=1)
            re_sen_pre = torch.cat((re_sen_pre, rnum2vec), dim=1)

            ori_sen_pre = self.drop(ori_sen_pre)
            re_sen_pre = self.drop(re_sen_pre)
            # Splice the representation vectors of both branches
            mixed_feature = 2 * torch.cat((kwargs['l'] * ori_sen_pre, (1 - kwargs['l']) * re_sen_pre), dim=1)
            main_output = self.fc1(self.drop(mixed_feature))
            main_output = self.fc(main_output)
            au_output1 = self.au_task_fc1(self.drop(ausec_sen_pre))
            return main_output, au_output1
        re_sen_pre = self.get_sen_att(x1, bert_output, 're', attention_mask)
        mixed_feature = torch.cat((ori_sen_pre, re_sen_pre), dim=1)
        mixed_feature = self.fc1(mixed_feature)
        mixed_feature = self.fc(mixed_feature)
        return mixed_feature


    def get_alpha(self, word_mat, data_type, mask):
        if data_type == 'ori':
            # representation learning  attention
            att_w = self.ori_word_atten(word_mat)
            att_w = self.ori_tanh(att_w)
            att_w = self.ori_word_weight(att_w)
        elif data_type == 'des':
            att_w = self.des_word_atten(word_mat)
            att_w = self.des_tanh(att_w)
            att_w = self.des_word_weight(att_w)
        else:
            # classification learning  attention
            att_w = self.re_word_atten(word_mat)
            att_w = self.re_tanh(att_w)
            att_w = self.re_word_weight(att_w)

        mask = mask.unsqueeze(2)
        att_w = att_w.masked_fill(mask == 0, float('-inf'))
        att_w = F.softmax(att_w, dim=1)
        return att_w

    # Get useful words vectors
    def get_word(self, sen, bert_output, mask):
        input_t2n = sen['input_ids'].cpu().numpy()
        sep_location = np.argwhere(input_t2n == 103)
        sep_location = sep_location[:, -1]

        # loc[0:size:2] 按步长取值

        select_index = list(range(sen['length'][0]))
        select_index.remove(0)  # 删除cls
        lhs = bert_output.last_hidden_state
        res = bert_output.hidden_states[8]
        relength = []
        recomposing = []
        mask_recomposing = []
        for i in range(lhs.shape[0]):
            select_index_f = select_index.copy()
            relength.append(sep_location[i] - 1)
            select_index_f.remove(sep_location[i])
            select_row = torch.index_select(lhs[i], 0,
                                            index=torch.LongTensor(select_index_f).to(sen['input_ids'].device))
            select_mask = torch.index_select(mask[i], 0,
                                             index=torch.LongTensor(select_index_f).to(sen['input_ids'].device))
            recomposing.append(select_row)
            mask_recomposing.append(select_mask)
        matrix = torch.stack(recomposing)
        mask = torch.stack(mask_recomposing)
        return matrix, mask

    # Get the representation vector after calculating the attention mechanism
    def get_sen_att(self, sen, bert_output, data_type, mask):
        word_mat, select_mask = self.get_word(sen, bert_output, mask)
        word_mat = self.drop(word_mat)
        att_w = self.get_alpha(word_mat, data_type, select_mask)
        word_mat = word_mat.permute(0, 2, 1)
        sen_pre = torch.bmm(word_mat, att_w).squeeze(2)
        return sen_pre

    def imix(self, input, alpha, share_lam=False):
        if not isinstance(alpha, (list, tuple)):
            alpha = [alpha, alpha]
        beta = torch.distributions.beta.Beta(*alpha)
        randind = torch.randperm(input.shape[0], device=input.device)
        if share_lam:
            lam = beta.sample().to(device=input.device)
            lam = torch.max(lam, 1. - lam)
            lam_expanded = lam
        else:
            lam = beta.sample([input.shape[0]]).to(device=input.device)
            lam = torch.max(lam, 1. - lam)
            lam_expanded = lam.view([-1] + [1] * (input.dim() - 1))
        output = lam_expanded * input + (1. - lam_expanded) * input[randind]
        return output, randind, lam