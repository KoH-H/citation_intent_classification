# -*- coding: utf-8 -*-
import json

from transformers import AutoTokenizer
import torch.optim as optim
from model.citation_model import *
from utils.scheduler import WarmupMultiStepLR
from train_valid.dataset_train import dataset_train_contr, dataset_train, dataset_train_imix, dataset_train_space, dataset_train_imix_space, \
    dataset_train_limix_rspace, dataset_train_imixspace_cross, dataset_train_limix_rspace_v2, dataset_train_labeldes_limix_rspace_v2
from train_valid.dataset_valid import dataset_valid
from utils.dataload import *
from utils.util import *
from sklearn.metrics import classification_report, confusion_matrix
import optuna
import time
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description = 'code for Citation Intent')
    parser.add_argument(
        "--mode",
        help="decide find parameters or train",
        default=None,
        type=str
    )
    # parser.add_argument(
    #     '--radio',
    #     help="proportion of training data",
    #     type=float
    # )
    args = parser.parse_args()
    return args

def run_optuna(path, dev):
    print('Run optuna')
    setup_seed(0)
    token = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    criterion = nn.CrossEntropyLoss()
    # dataset = load_data(16, reverse=True, multi=True, mul_num=2400)
    dataset = load_data(batch_size=16, radio=0.2)

    def objective(trial):
        model = Model('allenai/scibert_scivocab_uncased')
        # n_epoch = trial.suggest_int('n_epoch', 140, 170, log=True)
        n_epoch = 151
        lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
        au_weight = trial.suggest_float('au_weight', 0.001, 0.01, log=True)
        mix_w = trial.suggest_float('mix_w', 0.04, 0.1, log=True)
        # beta = trial.suggest_float('beta', 1, 10, log=True)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)
        scheduler = WarmupMultiStepLR(optimizer, [90, 110], gamma=0.1, warmup_epochs=5)
        best_model_f1, best_epoch = dataset_train_imix(model, token, dataset, criterion, optimizer, n_epoch,
                                                        au_weight, dev, mix_w, scheduler, model_path=path)

        return best_model_f1
    study = optuna.create_study(study_name='studyname', direction='maximize', storage='sqlite:///optuna.db', load_if_exists=True)
    study.optimize(objective, n_trials=10)
    print("Best_Params:{} \t Best_Value:{}".format(study.best_params, study.best_value))
    history = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    print(history)


def main_run(params, path, dev):
    setup_seed(0)
    token = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    model = Model('allenai/scibert_scivocab_uncased')
    criterion = nn.CrossEntropyLoss()
    # imix parames
    # au_weight = 0.001122
    n_epoch = 151
    # lr = 0.000583
    # mix_w = 0.022242


    # lr = 0.0001
    # au_weight = 0.007413
    # n_epoch = 151
    # mix_w = 0.05
    # dataset = load_data(16, reverse=True, multi=True, mul_num=2400)
    dataset = load_data(batch_size=16, radio=0.8)

    optimizer = optim.SGD(model.parameters(), lr=params.lr, momentum=0.9, weight_decay=2e-4)
    scheduler = WarmupMultiStepLR(optimizer, [90, 110], gamma=0.1, warmup_epochs=5)
    best_model_f1, best_epoch = dataset_train_imix(model, token, dataset, criterion, optimizer, n_epoch, params.au_weight, dev, params.mix_w,
                                                scheduler, model_path=path)
    print("best_model_f1:{} \t best_epoch:{}".format(best_model_f1, best_epoch))
    test_f1, test_micro_f1, test_true_label, test_pre_label = dataset_valid(model, token,
                                                                         dataset['test'], device,
                                                                         mode='test', path=path)
    print('Test'.center(20, '='))
    print('Test_True_Label:', collections.Counter(test_true_label))
    print('Test_Pre_Label:', collections.Counter(test_pre_label))
    print('Test macro F1: %.4f \t Test micro F1: %.4f' % (test_f1, test_micro_f1))
    print('Test'.center(20, '='))
    test_true = torch.Tensor(test_true_label).tolist()
    test_pre = torch.Tensor(test_pre_label).tolist()
    generate_submission(test_pre, 'mul_rev_val_f1_{:.5}_best_epoch_{}'.format(best_model_f1, best_epoch), test_f1)
    c_matrix = confusion_matrix(test_true, test_pre, labels=[0, 1, 2, 3, 4, 5])
    per_eval = classification_report(test_true, test_pre, labels=[0, 1, 2, 3, 4, 5])
    log_result(test_f1, best_model_f1,  c_matrix, per_eval, lr=params.lr, epoch=n_epoch, fun_name='main_multi_rev')


if __name__ == "__main__":
    args = parse_args()
    with open('params.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    args.lr = config['imix']['lr']
    args.au_weight = config['imix']['au_weight']
    args.mix_w = config['imix']['mix_w']
    tst = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    if args.mode == 'optuna':
        run_optuna('citation_mul_rev_model.pth', device)
    else:
        main_run(args, 'citation_mul_rev_model.pth', device)
    ten = time.time()
    print('Total time: {}'.format((ten - tst)))
