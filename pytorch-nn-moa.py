# Modified from
# https://www.kaggle.com/utkukubilay/notebooks

import sys
sys.path.append('./iterative-stratification-master')
from iterative_stratification_master.iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from iterative_stratification_master.iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from lib.model import Model
from lib.model import train_fn, valid_fn, inference_fn
from lib.data import *
from lib.data_processing import *

import numpy as np
import random
import pandas as pd
import os
import json


from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

import torch
import torch.nn as nn
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')

def seed_everything(seed=1903):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def main():

    seed_everything(seed=1903)
    # data_loc = "../input/lish-moa/"
    # # load data
    # train_features = pd.read_csv(data_loc + 'train_features.csv')
    # train_targets_scored = pd.read_csv(data_loc + 'train_targets_scored.csv')
    # # train_targets_nonscored = pd.read_csv(data_loc +  'train_targets_nonscored.csv')
    #
    # test_features = pd.read_csv(data_loc + 'test_features.csv')
    # # sample_submission = pd.read_csv(data_loc + 'sample_submission.csv')
    #
    # target_cols = [x for x in train_targets_scored.columns.to_list() if x != 'sig_id']
    # # save the target_cols into json file
    #
    #
    # # prepare training/test data
    # data_train_x, data_test_x = one_step_processing(train_features, test_features)
    #
    # feature_cols = [c for c in data_train_x.columns.to_list() if c != 'sig_id']

    # ====
    # with open("../input/process_1/target_cols.json", "w", encoding="utf-8") as f:
    #     json.dump(target_cols, f)
    # with open("../input/process_1/feature_cols.json", "w", encoding="utf-8") as f:
    #     json.dump(feature_cols, f)
    with open("../input/process_1/target_cols.json", "r", encoding="utf-8") as f:
        target_cols = json.load(f)
    with open("../input/process_1/feature_cols.json", "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    # HyperParameters
    # ==========================
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # EPOCHS = 27
    EPOCHS = 2
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5

    EARLY_STOPPING_STEPS = 11
    EARLY_STOP = True

    num_features = len(feature_cols)
    num_targets = len(target_cols)
    # hidden_size = 1024
    hidden_size = 50
    # ============================
    def run_training(train, valid, feature_cols, target_cols, model_path, seed):
        seed_everything(seed)

        x_train, y_train = train[feature_cols].values, train[target_cols].values
        x_valid, y_valid = valid[feature_cols].values, valid[target_cols].values

        # create the dataset loader
        train_dataset = MoADataset(x_train, y_train)
        valid_dataset = MoADataset(x_valid, y_valid)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # create an model instance
        model = Model(
            num_features=num_features,
            num_targets=num_targets,
            hidden_size=hidden_size,
        )

        model.to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.05, div_factor=1.5e3,
                                                  max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(trainloader))

        loss_fn = nn.BCEWithLogitsLoss()

        early_stopping_steps = EARLY_STOPPING_STEPS
        early_step = 0

        best_loss = np.inf

        #
        train_losses = []; valid_losses = []

        for epoch in range(EPOCHS):

            train_loss = train_fn(model, optimizer, scheduler, loss_fn, trainloader, DEVICE)
            train_losses.append(train_loss)
            # print(f"FOLD: {fold}, EPOCH: {epoch}, train_loss: {train_loss}")

            valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, DEVICE)
            valid_losses.append(valid_loss)

            if epoch % 5 == 0:
                print(f"EPOCH: {epoch}, train_loss: {train_loss}, valid_loss: {valid_loss}")

            if valid_loss < best_loss:

                best_loss = valid_loss
                # oof[val_idx] = valid_preds
                torch.save(model.state_dict(), model_path)

            elif EARLY_STOP == True:
                early_step += 1
                if early_step >= early_stopping_steps:
                    break

        return {"train_losses": train_losses, "valid_losses": valid_losses}

    # =====================================================
    # run kFold (# 3) twice
    # SEED = [1903, 1881]
    SEED = [1903]

    data_processed_input = "../input/process_1/"
    prefix = "process_1"
    models_dir = "./trained_models/"

    for seed in SEED:
        NFOLDS = 5
        # step 1: do the splitting on train_entire here to generate train, valid

        # train_entire = data_train_x.merge(train_targets_scored, on="sig_id")
        # # ==========================
        # # # k-fold split
        # #
        # NFOLDS = 5
        # mskf = MultilabelStratifiedKFold(n_splits=NFOLDS, random_state=seed)
        #
        # for f, (t_idx, v_idx) in enumerate(mskf.split(X=data_train_x, y=train_targets_scored[target_cols])):
        #     train_entire.loc[v_idx, 'kfold'] = int(f)
        #
        # train_entire['kfold'] = train_entire['kfold'].astype(int)
        # # we can create 5 train-valid pairs
        # train = train_entire[train_entire["kfold"] != 4]
        # valid = train_entire[train_entire["kfold"] == 4]
        # ==========================
        # train-valid split
        # msss = MultilabelStratifiedShuffleSplit(nsplit=2, test_size=0.2,  random_state=seed)
        # for train_index, test_index in msss.split(X=data_train_x, y=train_targets_scored[target_cols]):
        #     train, valid = train_entire[train_index], train_entire[test_index]

        # ==========================

        # one can just load the post-process data here
        train_entire = pd.read_pickle(data_processed_input + "data_train_5fold_1903.pkl")
        for k_fold in np.arange(NFOLDS):
            print("runing fold ", k_fold)
            # k_fold = 4
            train = train_entire[train_entire["kfold"] != k_fold]
            valid = train_entire[train_entire["kfold"] == k_fold]
            data_test_x = pd.read_pickle(data_processed_input + "data_test_x.pkl")
            model_path = models_dir + prefix + "_model_fold_{0:d}.pth".format(k_fold)

            model_result = run_training(train, valid, feature_cols, target_cols, model_path, seed)

            # make life easier
            total_num_inst = train_entire.shape[0]
            train_loss_final = model_result["train_losses"][-1]
            valid_loss_final = model_result["valid_losses"][-1]
            entire_train_loss = train.shape[0]/total_num_inst* train_loss_final\
                                + valid.shape[0]/total_num_inst*valid_loss_final

            print("train loss: ", train_loss_final, " valid loss: ", valid_loss_final,
                  "entire loss: ", entire_train_loss)

            # --------------------- PREDICTION on test dataset---------------------
            x_test = data_test_x[feature_cols].values
            testdataset = TestDataset(x_test)
            testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)
            model = Model(
                num_features=num_features,
                num_targets=num_targets,
                hidden_size=hidden_size,
            )
            model.load_state_dict(torch.load(model_path))
            model.to(DEVICE)
            predictions = inference_fn(model, testloader, DEVICE)
            # save the prediction on test set
            pred_test = pd.DataFrame(data=predictions, columns=target_cols).fillna(0)
            pred_test = pd.concat([data_test_x[["sig_id"]], pred_test], axis=1)
            pred_result_f = models_dir + prefix + "_fold_{0:d}_prediction.csv".format(k_fold)
            pred_test.to_csv(pred_result_f, index=False)

    # merge all the results by average
    # pred_test_average = 0
    for k_fold in np.arange(NFOLDS):
        pred_result_f = models_dir + prefix + "_fold_{0:d}_prediction.csv".format(k_fold)

        pred_test = pd.read_csv(pred_result_f, header=0)
        pred_test[target_cols] = pred_test[target_cols]/NFOLDS
        if k_fold == 0:
            pred_test_average = pred_test.copy(deep=True)
        else:
            pred_test_average[target_cols] = pred_test_average[target_cols] + pred_test[target_cols]

    # pred_test_average.to_csv("submission.csv", index=False)
    # have to use this way to say the result
    sample_submission = pd.read_csv("../input/lish-moa/" + 'sample_submission.csv')
    sub = sample_submission.drop(columns=target_cols).merge(pred_test_average, on='sig_id', how='left').fillna(0)
    sub.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    main()





