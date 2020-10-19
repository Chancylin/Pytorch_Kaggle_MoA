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
import datetime

import optuna


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

def print_model_results(train_loss, valid_loss, train_len, valid_len):
    total_num_inst = train_len + valid_len
    train_loss_final = train_loss
    valid_loss_final = valid_loss
    entire_train_loss = train_len / total_num_inst * train_loss_final \
                        + valid_len / total_num_inst * valid_loss_final
    print("train loss: ", train_loss_final, " valid loss: ", valid_loss_final,
          "entire loss: ", entire_train_loss)

if __name__ == '__main__':

    def run_training(train, valid, feature_cols, target_cols, model_path, seed,
                     param_provided=None):
        seed_everything(seed)

        x_train, y_train = train[feature_cols].values, train[target_cols].values
        # create the dataset loader
        train_dataset = MoADataset(x_train, y_train)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        if valid is not None:
            x_valid, y_valid = valid[feature_cols].values, valid[target_cols].values
            valid_dataset = MoADataset(x_valid, y_valid)
            validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # create an model instance
        if param_provided is not None:
            EPOCHS = param_provided['epoch']
            hidden_size = param_provided['hidden_size']
            LEARNING_RATE = param_provided['lr']

        print("hidden_size: ", hidden_size, ", learning_rate: ", LEARNING_RATE)

        # create an model instance
        model = Model(
            num_features=num_features,
            num_targets=num_targets,
            hidden_size=hidden_size,
        )

        model.to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        # scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.05, div_factor=1.5e3,
        #                                           max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(trainloader))

        # lmbda = lambda epoch: 0.5
        # scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

        loss_fn = nn.BCEWithLogitsLoss()

        early_stopping_steps = EARLY_STOPPING_STEPS
        early_step = 0

        best_loss = np.inf

        #
        train_losses = []; valid_losses = []

        for epoch in range(EPOCHS):

            print('Epoch {}, lr {}'.format(
                epoch, optimizer.param_groups[0]['lr']))

            train_loss = train_fn(model, optimizer, scheduler, loss_fn, trainloader, DEVICE)
            train_losses.append(train_loss)

            # print(f"FOLD: {fold}, EPOCH: {epoch}, train_loss: {train_loss}")
            if valid is not None:  # only run the valid if valid set is provided
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
            else:
                if epoch % 10 == 0:
                    print(f"EPOCH: {epoch}, train_loss: {train_loss}")

        print("early stop with epoch: ", epoch)
        print(f"LAST EPOCH: {epoch}, train_loss: {train_loss}")

        if valid is None:  # when there is not valid set, save the model
            torch.save(model.state_dict(), model_path)

        return {"train_losses": train_losses, "valid_losses": valid_losses}


    def run_training_tune(trial, train, valid, feature_cols, target_cols, model_path, seed):
        seed_everything(seed)

        x_train, y_train = train[feature_cols].values, train[target_cols].values
        x_valid, y_valid = valid[feature_cols].values, valid[target_cols].values

        # create the dataset loader
        train_dataset = MoADataset(x_train, y_train)
        valid_dataset = MoADataset(x_valid, y_valid)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

        hidden_size = trial.suggest_int("hidden_size", 30, 60, step=10)
        # create an model instance
        model = Model(
            num_features=num_features,
            num_targets=num_targets,
            hidden_size=hidden_size,
        )

        model.to(DEVICE)

        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        # lr = 0.0084

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        # this is used to change the learning rate when training model
        # scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.05,
        #                                          max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(trainloader))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

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
                # now don't save the model to adapt to the use of optuna
                # torch.save(model.state_dict(), model_path)

            elif EARLY_STOP == True:
                early_step += 1
                if early_step >= early_stopping_steps:
                    break

        print("early stopping with epoch: ", epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return valid_losses[-1]

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
    if_tune = False
    # ==========================
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # EPOCHS = 27
    EPOCHS = 20
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-2
    WEIGHT_DECAY = 1e-5

    EARLY_STOPPING_STEPS = 11
    EARLY_STOP = True

    num_features = len(feature_cols)
    num_targets = len(target_cols)
    # hidden_size = 1024
    hidden_size = 50

    # ============================


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
        best_param = {}
        # for k_fold in np.arange(NFOLDS):
        for k_fold in [4]:
            print("runing fold ", k_fold)
            # k_fold = 4
            # if have tuned the parameterts, uncomment the following line to use all data for training
            #  train = train_entire; valid = None
            train = train_entire[train_entire["kfold"] != k_fold]
            valid = train_entire[train_entire["kfold"] == k_fold]

            model_path = models_dir + prefix + "_model_fold_{0:d}.pth".format(k_fold)
            model_result = None

            if if_tune:
                # optuna for hyperparameter tuning
                # ==========================
                start_time = datetime.datetime.now()

                study = optuna.create_study(direction="minimize")
                study.optimize(lambda trial: run_training_tune(trial, train, valid, feature_cols, target_cols, model_path,
                                                           seed), n_trials=5) # timeout=600
                # ==========================

                # to check the optuna optimization
                # prune is useless since we already use early_stop in our model training
                pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
                complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                print("Study statistics: ")
                print("  Number of finished trials: ", len(study.trials))
                print("  Number of pruned trials: ", len(pruned_trials))
                print("  Number of complete trials: ", len(complete_trials))

                print("Best trial:")
                trial = study.best_trial

                print("  Value: ", trial.value)

                print("  Params: ")
                for key, value in trial.params.items():
                    print("    {}: {}".format(key, value))

                best_param["kfold_" + str(k_fold)] = trial.params

                end_time = datetime.datetime.now()
                print("time elapsed: ", end_time - start_time)

                print("you may also want to save num of epoch")
            else:
                model_result = run_training(train, valid, feature_cols, target_cols,
                                            model_path, seed)

                if valid is None:
                    print("loss on the entire training data", model_result["train_losses"][-1])
                else:
                    print_model_results(model_result["train_losses"][-1], model_result["valid_losses"][-1],
                                        train.shape[0], valid.shape[0])

    # best_param + the early_stop epoch  --> best_param_with_epoch
    # best_param_with_epoch = {'kfold_0': {'hidden_size': 450, 'lr': 0.0008925929114331024, 'epoch': 30},...}
    best_param_with_epoch = None


    # Move the prediction as an independent section
    # --------------------- PREDICTION on test dataset---------------------
    data_test_x = pd.read_pickle(data_processed_input + "data_test_x.pkl")
    x_test = data_test_x[feature_cols].values
    testdataset = TestDataset(x_test)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)

    for k_fold in np.arange(NFOLDS):
        # for k_fold in [0]:

        if best_param_with_epoch is not None:
            hidden_size = best_param_with_epoch["kfold_" + str(k_fold)]['hidden_size']

        model_path = models_dir + prefix + "_model_fold_{0:d}.pth".format(k_fold)

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
    folds_to_use = np.arange(NFOLDS)
    for i_count, k_fold in enumerate(folds_to_use):
        pred_result_f = models_dir + prefix + "_fold_{0:d}_prediction.csv".format(k_fold)

        pred_test = pd.read_csv(pred_result_f, header=0)
        pred_test[target_cols] = pred_test[target_cols] / len(folds_to_use)
        if i_count == 0:
            pred_test_average = pred_test.copy(deep=True)
        else:
            pred_test_average[target_cols] = pred_test_average[target_cols] + pred_test[target_cols]

    # pred_test_average.to_csv("submission.csv", index=False)
    # have to use this way to say the result
    sample_submission = pd.read_csv("../input/lish-moa/" + 'sample_submission.csv')
    sub = sample_submission.drop(columns=target_cols).merge(pred_test_average, on='sig_id', how='left').fillna(0)
    sub.to_csv('submission.csv', index=False)





