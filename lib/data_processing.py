import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def process_data(data):
    data = pd.get_dummies(data, columns=['cp_time', 'cp_dose'])

    return data

# one-hot-encoding
def ohe(df, cols):
    return pd.get_dummies(df, prefix = cols, columns=cols, drop_first=True)


def one_step_processing(train_features, test_features):
    random_seed = 1903

    GENES = [col for col in train_features.columns if col.startswith('g-')]
    CELLS = [col for col in train_features.columns if col.startswith('c-')]

    data_gene = pd.concat([pd.DataFrame(train_features[GENES]), pd.DataFrame(test_features[GENES])])
    data_cell = pd.concat([pd.DataFrame(train_features[CELLS]), pd.DataFrame(test_features[CELLS])])

    # create scaler
    scaler_gene = MinMaxScaler()

    #  fit  the scaler to the train set
    scaler_gene.fit(data_gene[GENES])
    # transform the train and test set
    data_gene[GENES] = scaler_gene.transform(data_gene[GENES])

    scaler_cell = MinMaxScaler()
    scaler_cell.fit(data_cell[CELLS])
    data_cell[CELLS] = scaler_cell.transform(data_cell[CELLS])

    ### PCA tranformation
    n_comp = 200
    pca_gene = PCA(n_components=n_comp, random_state=random_seed)
    data_gene_pca = pca_gene.fit_transform(data_gene[GENES])

    n_comp = 4
    pca_cell = PCA(n_components=n_comp, random_state=random_seed)
    data_cell_pca = pca_cell.fit_transform(data_cell[CELLS])

    n_comp = 200
    data_gene_pca = pd.DataFrame(data_gene_pca, columns=[f'pca_G-{i}' for i in range(n_comp)])
    n_comp = 4
    data_cell_pca = pd.DataFrame(data_cell_pca, columns=[f'pca_C-{i}' for i in range(n_comp)])

    cols = ['sig_id', 'cp_type', 'cp_time', 'cp_dose']
    data_other_cols = pd.concat([train_features[cols], test_features[cols]], axis=0, ignore_index=True)
    cols_ohe = ['cp_type', 'cp_dose']
    data_other_ohe = ohe(data_other_cols, cols_ohe)
    # normalize the cp_time
    data_other_ohe["cp_time"] = data_other_ohe["cp_time"]/72.0

    final_cols = data_other_ohe.columns.to_list() + data_gene_pca.columns.to_list() + data_cell_pca.columns.to_list()

    name_dict = {}
    for i, col in enumerate(final_cols):
        name_dict[i] = col

    data_final = pd.concat([data_other_ohe, data_gene_pca, data_cell_pca], axis=1, ignore_index=True)

    data_final.rename(columns=name_dict, inplace=True)

    data_train = data_final.iloc[0:23814, :].copy(deep=True)
    data_test = data_final.iloc[23814:, :].copy(deep=True).reset_index(drop=True)

    print("after preprocessing, shape of traning data: ", data_train.shape, \
          " shape of test data:", data_test.shape)

    return data_train, data_test