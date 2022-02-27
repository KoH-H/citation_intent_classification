# -*- coding: utf-8 -*-
import json
from transformers import AutoTokenizer, AutoConfig
import torch.optim as optim
# from model.citation_model import *
# from model.citation_model_num import *
from model.Model import *
# from model.cnn_bert import *
from utils.scheduler import WarmupMultiStepLR
# from train_valid.dataset_train import *
from train_valid.dataset_valid import dataset_valid
from train_valid.SupCNNTrain import *
from utils.dataload import *
from utils.util import *
from sklearn.metrics import classification_report, confusion_matrix
import optuna
import time
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(description='code for Citation Intent')
    parser.add_argument("--mode", help="decide find parameters or train", default=None, type=str)
    parser.add_argument("--dataname", help="dataname", default=None, type=str)
    parser.add_argument("--tp", help="type of params", default=None, type=str)
    parser.add_argument("--epochs", default=35, type=int)
    parser.add_argument("--bsz", default=16, type=int)
    args = parser.parse_args()
    return args


def set_optimizer(lr, model):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = WarmupMultiStepLR(optimizer, [15, 25], gamma=0.1, warmup_epochs=5)
    return optimizer, scheduler


def set_model(tp, conf):
    if tp == "supcnn":
        cnnl, cnnr = CNN(768), CNN(768)
        model = SupCNN('allenai/scibert_scivocab_uncased', config=conf, cnnl=cnnl, cnnr=cnnr)
    elif tp == "onlysup":
        model = OnlySupLoss('allenai/scibert_scivocab_uncased', config=conf)
    else:
        cnnl, cnnr = CNN(768), CNN(768)
        model = OnlyCNN('allenai/scibert_scivocab_uncased', config=conf, cnnl=cnnl, cnnr=cnnr)
    return model


def set_token(dropout=0.3):
    token = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    conf = AutoConfig.from_pretrained('allenai/scibert_scivocab_uncased')
    conf.hidden_dropout_prob = 0.3
    criterion = nn.CrossEntropyLoss()
    return token, criterion, conf


def run_optuna(params, path, dev):
    print('Run optuna')
    setup_seed(0)
    token, criterion, conf = set_token()
    dataset = load_data(params.dataname, batch_size=params.bsz, radio=0.8)

    def objective(trial):
        model = set_model(params.tp, conf)
        lr = trial.suggest_float('lr', 1e-5, 1e-4, log=True)
        auw = trial.suggest_float('auw', 0.001, 0.01, log=True)
        optimizer, scheduler = set_optimizer(lr, model=model)
        best_model_f1, best_epoch = supcnn(model, token, dataset, criterion, optimizer, params.epochs,
                                           auw, dev, scheduler, model_path=path)

        return best_model_f1
    study = optuna.create_study(study_name='studyname', direction='maximize', storage='sqlite:///optuna.db',
                                load_if_exists=True)
    study.optimize(objective, n_trials=5)
    print("Best_Params:{} \t Best_Value:{}".format(study.best_params, study.best_value))
    history = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    print(history)
    print("Train".center(30, '-'))
    args.lr = float(format(study.best_params['lr'], '.6f'))
    args.auw = float(format(study.best_params['auw'], '.6f'))
    main_run(args, 'citation_mul_rev_model.pth', device)


# "onlycnn": {
#       "lr": 0.000036,
#       "auw": 0.003202,
#       "bsz": 16,
#       "epochs": 40,
#       "mix_w": 0
#     },


def main_run(params, path, dev):
    setup_seed(0)
    token, criterion, conf = set_token()
    model = set_model(params.tp, conf)
    # n_epoch = 35
    # lr = 0.0001
    # au_weight = 0.007413
    dataset = load_data(params.dataname, batch_size=params.bsz, radio=0.8)
    optimizer, scheduler = set_optimizer(params.lr, model)
    best_model_f1, best_epoch = supcnn(model, token, dataset, criterion, optimizer, params.epochs, params.auw, dev,
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
    generate_submission(test_pre, 'mul_rev_val_f1_{:.5}_best_epoch_{}'.format(best_model_f1, best_epoch), test_f1, params.dataname)
    c_matrix = confusion_matrix(test_true, test_pre, labels=[0, 1, 2, 3, 4, 5])
    per_eval = classification_report(test_true, test_pre, labels=[0, 1, 2, 3, 4, 5])
    log_result(test_f1, best_model_f1,  c_matrix, per_eval, lr=params.lr, epoch=params.epochs, fun_name='main_multi_rev')
# 是否在输出时加上 1- dropout, 不需要加 因为在训练阶段不仅会遮盖 还会将遮盖后的放大。  https://www.zhihu.com/question/61751133
# 将SGD改为Adam


# "original": {
#       "lr": 0.000036,
#       "au_weight":  0.003202,
#       "mix_w": 0
#     },

if __name__ == "__main__":
    # act imix best params
    # {
    #     "lr": 0.0001,
    #     "au_weight": 0.007413,
    #     "mix_w": 0.05
    # }
    args = parse_args()
    tst = time.time()
    if args.mode == 'optuna':
        run_optuna(args, 'citation_mul_rev_model.pth', device)
    else:
        with open('params.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        args.lr = config[args.dataname][args.tp]['lr']
        args.auw = config[args.dataname][args.tp]['auw']
        args.bsz = config[args.dataname][args.tp]['bsz']
        args.epochs = config[args.dataname][args.tp]['epochs']
        main_run(args, 'citation_mul_rev_model.pth', device)
    ten = time.time()
    print('Total time: {}'.format((ten - tst)))


# "supcnn": {
#       "lr": 0.00005,
#       "auw": 0.005,
#       "bsz": 16,
#       "epochs": 35,
#       "mix_w": 0
#     }  0.6942
