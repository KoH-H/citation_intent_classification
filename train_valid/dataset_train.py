# -*- coding: utf-8 -*-
from utils.util import *
from utils.dataload import load_data
from train_valid.dataset_valid import dataset_valid
import time
from sklearn.metrics import f1_score

import torch.nn.functional as F



def compute_kl_loss(p, q,pad_mask = None):

    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    # p_loss = p_loss.sum()
    # q_loss = q_loss.sum()
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss

# keep dropout and forward twice
# logits = model(x)
#
# logits2 = model(x)

# cross entropy loss for classifier
# ce_loss = 0.5 * (cross_entropy_loss(logits, label) + cross_entropy_loss(logits2, label))
#
# kl_loss = compute_kl_loss(logits, logits2)
#
# # carefully choose hyper-parameters
# loss = ce_loss + beta * kl_loss


def dataset_train(model, token, data, criterion, optimize, n_epoch, au_weight, device, scheduler=None, model_path=None):
    model.to(device=device)
    best_val_f1, counts, tmp, best_epoch = 0, 0, 0, 0
    train_sen = data['train']['sen']
    train_tar = data['train']['tar']
    re_sen = data['reverse']['sen']
    re_tar = data['reverse']['tar']
    sec_sen = data['section']['sen']
    sec_tar = data['section']['tar']
    assert len(train_sen) <= len(sec_sen)

    for i in range(1, n_epoch + 1):
        model.train()
        avg_loss, avgori_loss, avgau_loss1,  avgre_loss = 0, 0, 0, 0
        train_ltrue_label, train_rtrue_label, train_lpre_label, train_rpre_label = [], [], [], []
        alpha = 1 - ((i - 1) / n_epoch) ** 2
        tst = time.time()

        for index, (t_sen, t_tar, r_sen, r_tar, s_sen, s_tar) in enumerate(zip(train_sen, train_tar, re_sen, re_tar,
                                                                               sec_sen, sec_tar)):
            t_sent = token(t_sen, return_tensors='pt', is_split_into_words=True, padding=True,
                           add_special_tokens=True, return_length=True)
            r_sent = token(r_sen, return_tensors='pt', is_split_into_words=True, padding=True,
                           add_special_tokens=True, return_length=True)
            s_sent = token(s_sen, return_tensors='pt', is_split_into_words=True, padding=True,
                           add_special_tokens=True, return_length=True)
            optimize.zero_grad()

            t_sent = t_sent.to(device)
            r_sent = r_sent.to(device)
            s_sent = s_sent.to(device)
            train_t_tar = torch.LongTensor(t_tar)
            train_r_tar = torch.LongTensor(r_tar)
            s_tar = torch.LongTensor(s_tar)
            main_output, au_output1 = model(t_sent, r_sen=r_sent, s_sen=s_sent, l=alpha)
            # L_o
            ori_loss = criterion(main_output, train_t_tar.to(device))
            # L_r
            re_loss = criterion(main_output, train_r_tar.to(device))
            # L_a
            au_loss = criterion(au_output1, s_tar.to(device))
            loss = alpha * (ori_loss + au_weight * au_loss) + (1 - alpha) * re_loss  # 0.05表现较好
            loss.backward()
            optimize.step()

            avg_loss += loss.item()
            avgori_loss += ori_loss.item()
            avgau_loss1 += au_loss.item()
            avgre_loss += re_loss.item()

            pre_output = torch.softmax(main_output, dim=1)
            train_value, train_location = torch.max(pre_output, dim=1)

            train_lpre_label.extend(train_location.tolist())
            train_ltrue_label.extend(t_tar)

            train_rpre_label.extend(train_location.tolist())
            train_rtrue_label.extend(r_tar)

            if (index + 1) % 10 == 0:
                print('Batch: %d \t Loss: %.4f \t Avgori_loss: %.4f \t Avgau_loss: %.4f \t Avgre_loss: %.4f' % (
                (index + 1), avg_loss / 10, avgori_loss / 10, avgau_loss1 / 10, avgre_loss / 10))
                avg_loss, avgori_loss, avgau_loss1, avgre_loss = 0, 0, 0, 0
        ten = time.time()
        print('Epoch Train time: {}'.format(ten - tst))
        l_macro_f1 = f1_score(torch.LongTensor(train_ltrue_label), torch.LongTensor(train_lpre_label), average='macro')
        l_micro_f1 = f1_score(torch.LongTensor(train_ltrue_label), torch.LongTensor(train_lpre_label), average='micro')
        r_macro_f1 = f1_score(torch.LongTensor(train_rtrue_label), torch.LongTensor(train_rpre_label), average='macro')
        r_micro_f1 = f1_score(torch.LongTensor(train_rtrue_label), torch.LongTensor(train_rpre_label), average='micro')

        print("Train".center(20, '='))
        print("Train L_macro_f1: %.4f \t L_micro_f1: %.4f \t R_macro_f1: %.4f \t R_micro_f1: %.4f"% (l_macro_f1, l_micro_f1, r_macro_f1, r_micro_f1))
        print("Train".center(20, '='))
        print('LR'.center(20, '*'))
        print('Learning Rate: {} \t Epoch'.format(optimize.param_groups[0]['lr']), i)
        print('LR'.center(20, '*'))
        scheduler.step()
        val_f1, val_micro_f1 = dataset_valid(model, token, data['val'], device, criterion=criterion)
        print("Val".center(20, '='))
        print('Epoch: %d \t macro_F1: %.4f \t micro_F1: %.4f' % (i, val_f1, val_micro_f1))
        print("Val".center(20, '='))
        if val_f1 > best_val_f1:
            print('Val F1: %.4f > %.4f Saving Model' % (val_f1, best_val_f1))
            torch.save(model.state_dict(), model_path)
            best_val_f1 = val_f1
            best_epoch = i
    return best_val_f1, best_epoch


def dataset_train_contr(model, token, data, criterion, optimize, n_epoch, au_weight, device, scheduler=None, model_path=None, beta=None):
    model.to(device=device)
    best_val_f1, counts, tmp, best_epoch = 0, 0, 0, 0
    train_sen = data['train']['sen']
    train_tar = data['train']['tar']
    re_sen = data['reverse']['sen']
    re_tar = data['reverse']['tar']
    sec_sen = data['section']['sen']
    sec_tar = data['section']['tar']
    assert len(train_sen) <= len(sec_sen)

    for i in range(1, n_epoch + 1):
        model.train()
        avg_loss, avgori_loss, avgau_loss1,  avgre_loss = 0, 0, 0, 0
        train_ltrue_label, train_rtrue_label, train_lpre_label, train_rpre_label = [], [], [], []
        alpha = 1 - ((i - 1) / n_epoch) ** 2
        tst = time.time()

        for index, (t_sen, t_tar, r_sen, r_tar, s_sen, s_tar) in enumerate(zip(train_sen, train_tar, re_sen, re_tar,
                                                                               sec_sen, sec_tar)):
            t_sent = token(t_sen, return_tensors='pt', is_split_into_words=True, padding=True,
                           add_special_tokens=True, return_length=True)
            r_sent = token(r_sen, return_tensors='pt', is_split_into_words=True, padding=True,
                           add_special_tokens=True, return_length=True)
            s_sent = token(s_sen, return_tensors='pt', is_split_into_words=True, padding=True,
                           add_special_tokens=True, return_length=True)
            optimize.zero_grad()

            t_sent = t_sent.to(device)
            r_sent = r_sent.to(device)
            s_sent = s_sent.to(device)
            train_t_tar = torch.LongTensor(t_tar)
            train_r_tar = torch.LongTensor(r_tar)
            s_tar = torch.LongTensor(s_tar)
            main_output, au_output1 = model(t_sent, r_sen=r_sent, s_sen=s_sent, l=alpha)
            main_output2, au_output12 = model(t_sent, r_sen=r_sent, s_sen=s_sent, l=alpha)
            # L_o
            ori_ce_loss = 0.5 * (criterion(main_output, train_t_tar.to(device)) + criterion(main_output2, train_t_tar.to(device)))
            ori_kl_loss = compute_kl_loss(p=main_output, q=main_output2)
            ori_loss = ori_ce_loss + beta * ori_kl_loss
            # L_r
            re_loss = criterion(main_output, train_r_tar.to(device))
            # L_a
            au_loss = criterion(au_output1, s_tar.to(device))
            loss = alpha * (ori_loss + au_weight * au_loss) + (1 - alpha) * re_loss  # 0.05表现较好
            loss.backward()
            optimize.step()

            avg_loss += loss.item()
            avgori_loss += ori_loss.item()
            avgau_loss1 += au_loss.item()
            avgre_loss += re_loss.item()

            pre_output = torch.softmax(main_output, dim=1)
            train_value, train_location = torch.max(pre_output, dim=1)

            train_lpre_label.extend(train_location.tolist())
            train_ltrue_label.extend(t_tar)

            train_rpre_label.extend(train_location.tolist())
            train_rtrue_label.extend(r_tar)

            if (index + 1) % 10 == 0:
                print('Batch: %d \t Loss: %.4f \t Avgori_loss: %.4f \t Avgau_loss: %.4f \t Avgre_loss: %.4f' % (
                (index + 1), avg_loss / 10, avgori_loss / 10, avgau_loss1 / 10, avgre_loss / 10))
                avg_loss, avgori_loss, avgau_loss1, avgre_loss = 0, 0, 0, 0
        ten = time.time()
        print('Epoch Train time: {}'.format(ten - tst))
        l_macro_f1 = f1_score(torch.LongTensor(train_ltrue_label), torch.LongTensor(train_lpre_label), average='macro')
        l_micro_f1 = f1_score(torch.LongTensor(train_ltrue_label), torch.LongTensor(train_lpre_label), average='micro')
        r_macro_f1 = f1_score(torch.LongTensor(train_rtrue_label), torch.LongTensor(train_rpre_label), average='macro')
        r_micro_f1 = f1_score(torch.LongTensor(train_rtrue_label), torch.LongTensor(train_rpre_label), average='micro')

        print("Train".center(20, '='))
        print("Train L_macro_f1: %.4f \t L_micro_f1: %.4f \t R_macro_f1: %.4f \t R_micro_f1: %.4f"% (l_macro_f1, l_micro_f1, r_macro_f1, r_micro_f1))
        print("Train".center(20, '='))
        print('LR'.center(20, '*'))
        print('Learning Rate: {} \t Epoch'.format(optimize.param_groups[0]['lr']), i)
        print('LR'.center(20, '*'))
        scheduler.step()
        val_f1, val_micro_f1 = dataset_valid(model, token, data['val'], device, criterion=criterion)
        print("Val".center(20, '='))
        print('Epoch: %d \t macro_F1: %.4f \t micro_F1: %.4f' % (i, val_f1, val_micro_f1))
        print("Val".center(20, '='))
        if val_f1 > best_val_f1:
            print('Val F1: %.4f > %.4f Saving Model' % (val_f1, best_val_f1))
            torch.save(model.state_dict(), model_path)
            best_val_f1 = val_f1
            best_epoch = i
    return best_val_f1, best_epoch