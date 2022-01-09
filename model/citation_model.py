# -*t coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import AutoModel
import numpy as np
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, name, temp=0.2):
        super(Model, self).__init__()
        self.model = AutoModel.from_pretrained(name)
        self.temp = temp
        self.fc1 = nn.Linear(768 * 2, 768)
        self.mix_fc = nn.Linear(768, 6)
        self.mix_fc1 = nn.Linear(768, 6)
        self.fc = nn.Linear(768, 6)
        self.drop = nn.Dropout(0.5)

        self.au_task_fc1 = nn.Linear(768, 5)

        self.ori_word_atten = nn.Linear(768, 384)
        self.ori_tanh = nn.Tanh()
        self.ori_word_weight = nn.Linear(384, 1, bias=False)

        self.re_word_atten = nn.Linear(768, 384)
        self.re_tanh = nn.Tanh()
        self.re_word_weight = nn.Linear(384, 1, bias=False)

    # Calculate the weight of each word
    def get_alpha(self, word_mat, data_type, mask):
        if data_type == 'ori':
            # representation learning  attention
            att_w = self.ori_word_atten(word_mat)
            att_w = self.ori_tanh(att_w)
            att_w = self.ori_word_weight(att_w)
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

    # original
    # def forward(self, x1, **kwargs):
    #     input_ids = x1['input_ids']
    #     batch_size = input_ids.shape[0]
    #     attention_mask = x1['attention_mask']
    #     bert_output = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
    #     ori_sen_pre = self.get_sen_att(x1, bert_output, 'ori', attention_mask)
    #
    #     if self.training:
    #         # Obtain the representation vector for the classification learning branch
    #         r_ids = kwargs['r_sen']['input_ids']
    #         r_attention_mask = kwargs['r_sen']['attention_mask']
    #         r_bert_output = self.model(r_ids, attention_mask=r_attention_mask, output_hidden_states=True)
    #         re_sen_pre = self.get_sen_att(kwargs['r_sen'], r_bert_output, 're', r_attention_mask)
    #         # Get the representation vector for the auxiliary task
    #         s_ids = kwargs['s_sen']['input_ids']
    #         s_attention_mask = kwargs['s_sen']['attention_mask']
    #         s_bert_output = self.model(s_ids, attention_mask=s_attention_mask, output_hidden_states=True)
    #         ausec_sen_pre = self.get_sen_att(kwargs['s_sen'], s_bert_output, 'ori', s_attention_mask)
    #
    #         ori_sen_pre = self.drop(ori_sen_pre)
    #         re_sen_pre = self.drop(re_sen_pre)
    #         # Splice the representation vectors of both branches
    #         mixed_feature = 2 * torch.cat((kwargs['l'] * ori_sen_pre, (1 - kwargs['l']) * re_sen_pre), dim=1)
    #         main_output = self.fc1(self.drop(mixed_feature))
    #         main_output = self.fc(main_output)
    #         au_output1 = self.au_task_fc1(self.drop(ausec_sen_pre))
    #         return main_output, au_output1
    #     re_sen_pre = self.get_sen_att(x1, bert_output, 're', attention_mask)
    #     mixed_feature = torch.cat((ori_sen_pre, re_sen_pre), dim=1)
    #     mixed_feature = self.fc1(mixed_feature)
    #     mixed_feature = self.fc(mixed_feature)
    #     return mixed_feature

    # forward for i-mix
    # def forward(self, x1, **kwargs):
    #     input_ids = x1['input_ids']
    #     batch_size = input_ids.shape[0]
    #     attention_mask = x1['attention_mask']
    #     bert_output = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
    #     ori_sen_pre = self.get_sen_att(x1, bert_output, 'ori', attention_mask)
    #
    #     if self.training:
    #         ori_sen_pre_mix = self.drop(ori_sen_pre)
    #         # for i-mix
    #         bert_output_imix = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
    #         ori_sen_pre_imix = self.get_sen_att(x1, bert_output_imix, 'ori', attention_mask)
    #         ori_sen_pre_imix = self.drop(ori_sen_pre_imix)
    #         # bert_output_imix = self.drop(bert_output_imix)
    #         ori_sen_pre_mix, labels_aux, lam = self.imix(ori_sen_pre_mix, kwargs['mix_alpha'])
    #         tem_ori_pre = torch.cat([ori_sen_pre_mix, ori_sen_pre_imix], dim=0)
    #         tem_ori_pre = self.mix_fc(tem_ori_pre)
    #
    #         tem_ori_pre = nn.functional.normalize(tem_ori_pre, dim=1)
    #         bert_output_mix, bert_output_imix = tem_ori_pre[:batch_size], tem_ori_pre[batch_size:]
    #         mix_logits = bert_output_mix.mm(bert_output_imix.t())
    #         mix_logits /= self.temp
    #         mix_labels = torch.arange(batch_size, dtype=torch.long).cuda()
    #         # mix_loss = (lam * criterion(mix_logits, mix_labels) + (1. - lam) * criterion(mix_logits, labels_aux)).mean()
    #
    #
    #         # Obtain the representation vector for the classification learning branch
    #         r_ids = kwargs['r_sen']['input_ids']
    #         r_attention_mask = kwargs['r_sen']['attention_mask']
    #         r_bert_output = self.model(r_ids, attention_mask=r_attention_mask, output_hidden_states=True)
    #         re_sen_pre = self.get_sen_att(kwargs['r_sen'], r_bert_output, 're', r_attention_mask)
    #         # Get the representation vector for the auxiliary task
    #         s_ids = kwargs['s_sen']['input_ids']
    #         s_attention_mask = kwargs['s_sen']['attention_mask']
    #         s_bert_output = self.model(s_ids, attention_mask=s_attention_mask, output_hidden_states=True)
    #         ausec_sen_pre = self.get_sen_att(kwargs['s_sen'], s_bert_output, 'ori', s_attention_mask)
    #
    #         ori_sen_pre = self.drop(ori_sen_pre)
    #         re_sen_pre = self.drop(re_sen_pre)
    #         # Splice the representation vectors of both branches
    #         mixed_feature = 2 * torch.cat((kwargs['l'] * ori_sen_pre, (1 - kwargs['l']) * re_sen_pre), dim=1)
    #         main_output = self.fc1(self.drop(mixed_feature))
    #         main_output = self.fc(main_output)
    #         au_output1 = self.au_task_fc1(self.drop(ausec_sen_pre))
    #         return main_output, au_output1, mix_logits, mix_labels, labels_aux, lam
    #     re_sen_pre = self.get_sen_att(x1, bert_output, 're', attention_mask)
    #     mixed_feature = torch.cat((ori_sen_pre, re_sen_pre), dim=1)
    #     mixed_feature = self.fc1(mixed_feature)
    #     mixed_feature = self.fc(mixed_feature)
    #     return mixed_feature

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

    # for feature space aug

    def generate_hidden_mean(self, pre, label):
        unique_label = torch.unique(label)
        mean_dict = dict()
        for i in range(unique_label.shape[0]):
            idx_t2n = label.numpy()
            index = np.argwhere(idx_t2n == unique_label[i].item())
            index = torch.tensor(index).squeeze(1).to(device=pre.device)
            select_vector = pre.index_select(0, index)
            mean_value = torch.mean(select_vector, 0)
            mean_dict[unique_label[i].item()] = mean_value
        return mean_dict

    # def forward(self, x1, **kwargs):
    #
    #     input_ids = x1['input_ids']
    #     attention_mask = x1['attention_mask']
    #     bert_output = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
    #     ori_sen_pre = self.get_sen_att(x1, bert_output, 'ori', attention_mask)
    #
    #     if self.training:
    #         # Obtain the representation vector for the classification learning branch
    #         r_ids = kwargs['r_sen']['input_ids']
    #         r_attention_mask = kwargs['r_sen']['attention_mask']
    #         r_bert_output = self.model(r_ids, attention_mask=r_attention_mask, output_hidden_states=True)
    #         re_sen_pre = self.get_sen_att(kwargs['r_sen'], r_bert_output, 're', r_attention_mask)
    #
    #         ori_mean = self.generate_hidden_mean(ori_sen_pre, kwargs['ori_label'])
    #         re_mean = self.generate_hidden_mean(re_sen_pre, kwargs['re_label'])
    #         re_sen_pre = None
    #         for i in range(ori_sen_pre.shape[0]):
    #             gen_example = ori_sen_pre[i] - ori_mean[kwargs['ori_label'][i].item()] + re_mean[kwargs['re_label'][i].item()]
    #             if re_sen_pre is None:
    #                 re_sen_pre = gen_example.unsqueeze(0)
    #             else:
    #                 re_sen_pre = torch.cat([re_sen_pre, gen_example.unsqueeze(0)], 0)
    #         # Get the representation vector for the auxiliary task
    #         s_ids = kwargs['s_sen']['input_ids']
    #         s_attention_mask = kwargs['s_sen']['attention_mask']
    #         s_bert_output = self.model(s_ids, attention_mask=s_attention_mask, output_hidden_states=True)
    #         ausec_sen_pre = self.get_sen_att(kwargs['s_sen'], s_bert_output, 'ori', s_attention_mask)
    #
    #         ori_sen_pre = self.drop(ori_sen_pre)
    #         re_sen_pre = self.drop(re_sen_pre)
    #         # Splice the representation vectors of both branches
    #         mixed_feature = 2 * torch.cat((kwargs['l'] * ori_sen_pre, (1 - kwargs['l']) * re_sen_pre), dim=1)
    #         main_output = self.fc1(self.drop(mixed_feature))
    #         main_output = self.fc(main_output)
    #         au_output1 = self.au_task_fc1(self.drop(ausec_sen_pre))
    #         return main_output, au_output1
    #     re_sen_pre = self.get_sen_att(x1, bert_output, 're', attention_mask)
    #     mixed_feature = torch.cat((ori_sen_pre, re_sen_pre), dim=1)
    #     mixed_feature = self.fc1(mixed_feature)
    #     mixed_feature = self.fc(mixed_feature)
    #     return mixed_feature

    # for imix and space aug
    def generate_new_example(self, ori_sen_pre, ori_mean, re_mean, ori_label, re_label):
        re_sen_pre = None
        for i in range(ori_sen_pre.shape[0]):
            gen_example = ori_sen_pre[i] - ori_mean[ori_label[i].item()] + re_mean[
                re_label[i].item()]
            if re_sen_pre is None:
                re_sen_pre = gen_example.unsqueeze(0)
            else:
                re_sen_pre = torch.cat([re_sen_pre, gen_example.unsqueeze(0)], 0)
        return re_sen_pre

    # def forward(self, x1, **kwargs):
    #
    #     input_ids = x1['input_ids']
    #     attention_mask = x1['attention_mask']
    #     batch_size = input_ids.shape[0]
    #     bert_output = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
    #     ori_sen_pre = self.get_sen_att(x1, bert_output, 'ori', attention_mask)
    #
    #     if self.training:
    #         # Get the representation vector for the auxiliary task
    #         s_ids = kwargs['s_sen']['input_ids']
    #         s_attention_mask = kwargs['s_sen']['attention_mask']
    #         s_bert_output = self.model(s_ids, attention_mask=s_attention_mask, output_hidden_states=True)
    #         ausec_sen_pre = self.get_sen_att(kwargs['s_sen'], s_bert_output, 'ori', s_attention_mask)
    #
    #
    #
    #         # for i-mix
    #         bert_output_imix = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
    #         ori_sen_pre_imix = self.get_sen_att(x1, bert_output_imix, 'ori', attention_mask)
    #
    #
    #         ori_sen_pre_mix = self.drop(ori_sen_pre)
    #         ori_sen_pre_imix = self.drop(ori_sen_pre_imix)
    #
    #         ori_imix_mean1 = self.generate_hidden_mean(ori_sen_pre_mix, kwargs['ori_label'])
    #         ori_imix_mean2 = self.generate_hidden_mean(ori_sen_pre_imix, kwargs['ori_label'])
    #
    #         # Obtain the representation vector for the classification learning branch
    #         r_ids = kwargs['r_sen']['input_ids']
    #         r_attention_mask = kwargs['r_sen']['attention_mask']
    #         r_bert_output = self.model(r_ids, attention_mask=r_attention_mask, output_hidden_states=True)
    #         re_sen_pre = self.get_sen_att(kwargs['r_sen'], r_bert_output, 're', r_attention_mask)
    #         re_mean = self.generate_hidden_mean(re_sen_pre, kwargs['re_label'])
    #         re_sen_pre1 = self.generate_new_example(ori_sen_pre_mix, ori_imix_mean1, re_mean, kwargs['ori_label'], kwargs['re_label'])
    #         re_sen_pre2 = self.generate_new_example(ori_sen_pre_imix, ori_imix_mean2, re_mean, kwargs['ori_label'], kwargs['re_label'])
    #
    #
    #         ori_sen_pre_mix = self.drop(ori_sen_pre_mix)
    #         ori_sen_pre_imix = self.drop(ori_sen_pre_imix)
    #         re_sen_pre1 = self.drop(re_sen_pre1)
    #         re_sen_pre2 = self.drop(re_sen_pre2)
    #
    #         # Splice the representation vectors of both branches
    #         mixed_feature1 = 2 * torch.cat((kwargs['l'] * ori_sen_pre_mix, (1 - kwargs['l']) * re_sen_pre1), dim=1)
    #         mixed_feature2 = 2 * torch.cat((kwargs['l'] * ori_sen_pre_imix, (1 - kwargs['l']) * re_sen_pre2), dim=1)
    #
    #         mixed_feature1, labels_aux, lam = self.imix(mixed_feature1, kwargs['mix_alpha'])
    #
    #         temp = torch.cat([mixed_feature1, mixed_feature2], dim=0)
    #
    #         main_output = self.fc1(temp)
    #         main_output = self.fc(main_output)
    #
    #         main_output = nn.functional.normalize(main_output, dim=1)
    #         main_output1, main_output2 = main_output[:batch_size], main_output[batch_size:]
    #         mix_logits = main_output1.mm(main_output2.t())
    #         mix_logits /= self.temp
    #         mix_labels = torch.arange(batch_size, dtype=torch.long).cuda()
    #
    #         au_output1 = self.au_task_fc1(self.drop(ausec_sen_pre))
    #         return au_output1, mix_logits, mix_labels, labels_aux, lam
    #     re_sen_pre = self.get_sen_att(x1, bert_output, 're', attention_mask)
    #     mixed_feature = torch.cat((ori_sen_pre, re_sen_pre), dim=1)
    #     mixed_feature = self.fc1(mixed_feature)
    #     mixed_feature = self.fc(mixed_feature)
    #     return mixed_feature


    # def forward(self, x1, **kwargs): # left imix & right space aug
    #     input_ids = x1['input_ids']
    #     batch_size = input_ids.shape[0]
    #     attention_mask = x1['attention_mask']
    #     bert_output = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
    #     ori_sen_pre = self.get_sen_att(x1, bert_output, 'ori', attention_mask)
    #
    #     if self.training:
    #         ori_sen_pre_mix = self.drop(ori_sen_pre)
    #         # for i-mix
    #         bert_output_imix = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
    #         ori_sen_pre_imix = self.get_sen_att(x1, bert_output_imix, 'ori', attention_mask)
    #         ori_sen_pre_imix = self.drop(ori_sen_pre_imix)
    #         # bert_output_imix = self.drop(bert_output_imix)
    #
    #
    #         ori_sen_pre_mix, labels_aux, lam = self.imix(ori_sen_pre_mix, kwargs['mix_alpha'])
    #         tem_ori_pre = torch.cat([ori_sen_pre_mix, ori_sen_pre_imix], dim=0)
    #         tem_ori_pre = self.mix_fc(tem_ori_pre)
    #
    #         tem_ori_pre = nn.functional.normalize(tem_ori_pre, dim=1)
    #         bert_output_mix, bert_output_imix = tem_ori_pre[:batch_size], tem_ori_pre[batch_size:]
    #         mix_logits = bert_output_mix.mm(bert_output_imix.t())
    #         mix_logits /= self.temp
    #         mix_labels = torch.arange(batch_size, dtype=torch.long).cuda()
    #
    #         # Obtain the representation vector for the classification learning branch
    #         r_ids = kwargs['r_sen']['input_ids']
    #         r_attention_mask = kwargs['r_sen']['attention_mask']
    #         r_bert_output = self.model(r_ids, attention_mask=r_attention_mask, output_hidden_states=True)
    #         re_sen_pre = self.get_sen_att(kwargs['r_sen'], r_bert_output, 're', r_attention_mask)
    #
    #         ori_mean = self.generate_hidden_mean(ori_sen_pre_mix, kwargs['ori_label'])
    #         re_mean = self.generate_hidden_mean(re_sen_pre, kwargs['re_label'])
    #         re_sen_pre = None
    #         for i in range(ori_sen_pre_mix.shape[0]):
    #             gen_example = ori_sen_pre_mix[i] - ori_mean[kwargs['ori_label'][i].item()] + re_mean[
    #                 kwargs['re_label'][i].item()]
    #             if re_sen_pre is None:
    #                 re_sen_pre = gen_example.unsqueeze(0)
    #             else:
    #                 re_sen_pre = torch.cat([re_sen_pre, gen_example.unsqueeze(0)], 0)
    #
    #         # Get the representation vector for the auxiliary task
    #         s_ids = kwargs['s_sen']['input_ids']
    #         s_attention_mask = kwargs['s_sen']['attention_mask']
    #         s_bert_output = self.model(s_ids, attention_mask=s_attention_mask, output_hidden_states=True)
    #         ausec_sen_pre = self.get_sen_att(kwargs['s_sen'], s_bert_output, 'ori', s_attention_mask)
    #
    #         ori_sen_pre = self.drop(ori_sen_pre_mix)
    #         re_sen_pre = self.drop(re_sen_pre)
    #         # Splice the representation vectors of both branches
    #         mixed_feature = 2 * torch.cat((kwargs['l'] * ori_sen_pre, (1 - kwargs['l']) * re_sen_pre), dim=1)
    #         main_output = self.fc1(self.drop(mixed_feature))
    #         main_output = self.fc(main_output)
    #         au_output1 = self.au_task_fc1(self.drop(ausec_sen_pre))
    #         return main_output, au_output1, mix_logits, mix_labels, labels_aux, lam
    #     re_sen_pre = self.get_sen_att(x1, bert_output, 're', attention_mask)
    #     mixed_feature = torch.cat((ori_sen_pre, re_sen_pre), dim=1)
    #     mixed_feature = self.fc1(mixed_feature)
    #     mixed_feature = self.fc(mixed_feature)
    #     return mixed_feature

    # space aug for imix
    def forward(self, x1, **kwargs):
        input_ids = x1['input_ids']
        batch_size = input_ids.shape[0]
        attention_mask = x1['attention_mask']
        bert_output = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        ori_sen_pre = self.get_sen_att(x1, bert_output, 'ori', attention_mask)

        if self.training:
            ori_sen_pre_mix = self.drop(ori_sen_pre)
            # for i-mix
            # bert_output_imix = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            # ori_sen_pre_imix = self.get_sen_att(x1, bert_output_imix, 'ori', attention_mask)
            # ori_sen_pre_imix = self.drop(ori_sen_pre_imix)
            # bert_output_imix = self.drop(bert_output_imix)

            r_ids = kwargs['r_sen']['input_ids']
            r_attention_mask = kwargs['r_sen']['attention_mask']
            r_bert_output = self.model(r_ids, attention_mask=r_attention_mask, output_hidden_states=True)
            re_sen_pre = self.get_sen_att(kwargs['r_sen'], r_bert_output, 're', r_attention_mask)
            re_sen_pre_mix = self.drop(re_sen_pre)

            ori_mean = self.generate_hidden_mean(ori_sen_pre_mix, kwargs['ori_label'])
            re_mean = self.generate_hidden_mean(re_sen_pre_mix, kwargs['re_label'])

            new_re_sen_pre, new_ori_sen_pre = None, None
            for i in range(ori_sen_pre_mix.shape[0]):
                gen_re_example = ori_sen_pre_mix[i] - ori_mean[kwargs['ori_label'][i].item()] + re_mean[
                    kwargs['re_label'][i].item()]
                gen_ori_example = re_sen_pre_mix[i] - re_mean[kwargs['re_label'][i].item()] + ori_mean[kwargs['ori_label'][i].item()]
                if new_re_sen_pre is None:
                    new_re_sen_pre = gen_re_example.unsqueeze(0)
                else:
                    new_re_sen_pre = torch.cat([new_re_sen_pre, gen_re_example.unsqueeze(0)], 0)

                if new_ori_sen_pre is None:
                    new_ori_sen_pre = gen_ori_example.unsqueeze(0)
                else:
                    new_ori_sen_pre = torch.cat([new_ori_sen_pre, gen_ori_example.unsqueeze(0)], 0)



            ori_sen_pre_mix, labels_aux, lam = self.imix(ori_sen_pre_mix, kwargs['mix_alpha'])
            re_sen_pre_mix, re_label_aux, re_lam = self.imix(re_sen_pre_mix, kwargs['mix_alpha'])

            tem_ori_pre = torch.cat([ori_sen_pre_mix, new_ori_sen_pre], dim=0)
            tem_re_pre = torch.cat([re_sen_pre_mix, new_re_sen_pre], dim=0)

            tem_ori_pre = self.mix_fc(tem_ori_pre)
            tem_re_pre = self.mix_fc1(tem_re_pre)

            tem_ori_pre = nn.functional.normalize(tem_ori_pre, dim=1)
            tem_re_pre = nn.functional.normalize(tem_re_pre, dim=1)
            bert_output_mix, bert_output_imix = tem_ori_pre[:batch_size], tem_ori_pre[batch_size:]
            re_bert_output_mix, re_bert_output_imix = tem_re_pre[:batch_size], tem_re_pre[batch_size:]
            mix_logits = bert_output_mix.mm(bert_output_imix.t())
            re_mix_logits = re_bert_output_mix.mm(re_bert_output_imix.t())
            mix_logits /= self.temp
            re_mix_logits /= self.temp
            mix_labels = torch.arange(batch_size, dtype=torch.long).cuda()
            re_mix_labels = torch.arange(batch_size, dtype=torch.long).cuda()

            # Obtain the representation vector for the classification learning branch
            # r_ids = kwargs['r_sen']['input_ids']
            # r_attention_mask = kwargs['r_sen']['attention_mask']
            # r_bert_output = self.model(r_ids, attention_mask=r_attention_mask, output_hidden_states=True)
            # re_sen_pre = self.get_sen_att(kwargs['r_sen'], r_bert_output, 're', r_attention_mask)
            #
            # ori_mean = self.generate_hidden_mean(ori_sen_pre_mix, kwargs['ori_label'])
            # re_mean = self.generate_hidden_mean(re_sen_pre, kwargs['re_label'])
            # re_sen_pre = None
            # for i in range(ori_sen_pre_mix.shape[0]):
            #     gen_example = ori_sen_pre_mix[i] - ori_mean[kwargs['ori_label'][i].item()] + re_mean[
            #         kwargs['re_label'][i].item()]
            #     if re_sen_pre is None:
            #         re_sen_pre = gen_example.unsqueeze(0)
            #     else:
            #         re_sen_pre = torch.cat([re_sen_pre, gen_example.unsqueeze(0)], 0)

            # Get the representation vector for the auxiliary task
            s_ids = kwargs['s_sen']['input_ids']
            s_attention_mask = kwargs['s_sen']['attention_mask']
            s_bert_output = self.model(s_ids, attention_mask=s_attention_mask, output_hidden_states=True)
            ausec_sen_pre = self.get_sen_att(kwargs['s_sen'], s_bert_output, 'ori', s_attention_mask)

            ori_sen_pre = self.drop(ori_sen_pre)
            re_sen_pre = self.drop(re_sen_pre)
            # Splice the representation vectors of both branches
            mixed_feature = 2 * torch.cat((kwargs['l'] * ori_sen_pre, (1 - kwargs['l']) * re_sen_pre), dim=1)
            main_output = self.fc1(self.drop(mixed_feature))
            main_output = self.fc(main_output)
            au_output1 = self.au_task_fc1(self.drop(ausec_sen_pre))
            return main_output, au_output1, mix_logits, mix_labels, labels_aux, lam, re_mix_logits, re_mix_labels, re_label_aux, re_lam
        re_sen_pre = self.get_sen_att(x1, bert_output, 're', attention_mask)
        mixed_feature = torch.cat((ori_sen_pre, re_sen_pre), dim=1)
        mixed_feature = self.fc1(mixed_feature)
        mixed_feature = self.fc(mixed_feature)
        return mixed_feature
