# forward for i-mix
def i_mix(self, x1, **kwargs):
    input_ids = x1['input_ids']
    batch_size = input_ids.shape[0]
    attention_mask = x1['attention_mask']
    bert_output = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
    ori_sen_pre = self.get_sen_att(x1, bert_output, 'ori', attention_mask)

    if self.training:
        ori_sen_pre_mix = self.drop(ori_sen_pre)
        # for i-mix
        bert_output_imix = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        ori_sen_pre_imix = self.get_sen_att(x1, bert_output_imix, 'ori', attention_mask)
        ori_sen_pre_imix = self.drop(ori_sen_pre_imix)
        # bert_output_imix = self.drop(bert_output_imix)
        ori_sen_pre_mix, labels_aux, lam = self.imix(ori_sen_pre_mix, kwargs['mix_alpha'])
        tem_ori_pre = torch.cat([ori_sen_pre_mix, ori_sen_pre_imix], dim=0)
        tem_ori_pre = self.mix_fc(tem_ori_pre)

        tem_ori_pre = nn.functional.normalize(tem_ori_pre, dim=1)
        bert_output_mix, bert_output_imix = tem_ori_pre[:batch_size], tem_ori_pre[batch_size:]
        mix_logits = bert_output_mix.mm(bert_output_imix.t())
        mix_logits /= self.temp
        mix_labels = torch.arange(batch_size, dtype=torch.long).cuda()
        # mix_loss = (lam * criterion(mix_logits, mix_labels) + (1. - lam) * criterion(mix_logits, labels_aux)).mean()


        # Obtain the representation vector for the classification learning branch
        r_ids = kwargs['r_sen']['input_ids']
        r_attention_mask = kwargs['r_sen']['attention_mask']
        r_bert_output = self.model(r_ids, attention_mask=r_attention_mask, output_hidden_states=True)
        re_sen_pre = self.get_sen_att(kwargs['r_sen'], r_bert_output, 're', r_attention_mask)
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
        return main_output, au_output1, mix_logits, mix_labels, labels_aux, lam
    re_sen_pre = self.get_sen_att(x1, bert_output, 're', attention_mask)
    mixed_feature = torch.cat((ori_sen_pre, re_sen_pre), dim=1)
    mixed_feature = self.fc1(mixed_feature)
    mixed_feature = self.fc(mixed_feature)
    return mixed_feature


def space_aug(self, x1, **kwargs):

    input_ids = x1['input_ids']
    attention_mask = x1['attention_mask']
    bert_output = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
    ori_sen_pre = self.get_sen_att(x1, bert_output, 'ori', attention_mask)

    if self.training:
        # Obtain the representation vector for the classification learning branch
        r_ids = kwargs['r_sen']['input_ids']
        r_attention_mask = kwargs['r_sen']['attention_mask']
        r_bert_output = self.model(r_ids, attention_mask=r_attention_mask, output_hidden_states=True)
        re_sen_pre = self.get_sen_att(kwargs['r_sen'], r_bert_output, 're', r_attention_mask)

        ori_mean = self.generate_hidden_mean(ori_sen_pre, kwargs['ori_label'])
        re_mean = self.generate_hidden_mean(re_sen_pre, kwargs['re_label'])
        re_sen_pre = None
        for i in range(ori_sen_pre.shape[0]):
            gen_example = ori_sen_pre[i] - ori_mean[kwargs['ori_label'][i].item()] + re_mean[kwargs['re_label'][i].item()]
            if re_sen_pre is None:
                re_sen_pre = gen_example.unsqueeze(0)
            else:
                re_sen_pre = torch.cat([re_sen_pre, gen_example.unsqueeze(0)], 0)
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
        return main_output, au_output1
    re_sen_pre = self.get_sen_att(x1, bert_output, 're', attention_mask)
    mixed_feature = torch.cat((ori_sen_pre, re_sen_pre), dim=1)
    mixed_feature = self.fc1(mixed_feature)
    mixed_feature = self.fc(mixed_feature)
    return mixed_feature
