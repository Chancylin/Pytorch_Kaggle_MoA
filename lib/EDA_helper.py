import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


# check the range
def check_range_distr(df, col_list):
    df_range_min = df[col_list].apply(np.min).to_frame().T
    df_range_max = df[col_list].apply(np.max).to_frame().T
    df_range = pd.concat([df_range_min, df_range_max], axis=0, ignore_index=True)

    df_interval = df_range.apply(lambda x: x[1] - x[0], axis=0).to_frame().T
    df_range = pd.concat([df_range, df_interval], axis=0, ignore_index=True)
    df_range.rename(index={0: "min", 1: "max", 2: "interval"}, inplace=True)

    return df_range


def check_range_plot(df):
    fig, ax = plt.subplots(3, 1)
    df.loc["min", :].hist(ax=ax[0])
    df.loc["max", :].hist(ax=ax[1])
    df.loc["interval", :].hist(ax=ax[2])


def check_data_dist(df, col_list, use_kde=False):
    """
    Exmaple:
        col_list = ["g-0", "g-2", "g-10", "g-100", "g-400"]
        check_data_dist(train_features, col_list)
    """
    num = len(col_list)
    fig, ax = plt.subplots(int(num / 2) + 1, 2, figsize=(12, 4 * (int(num / 2) + 1)))

    for i, col in enumerate(col_list):
        ax_row = int(i / 2);
        ax_col = i % 2;
        print(col, " row: ", ax_row, " col: ", ax_col)
        if use_kde:
            df[col].plot.kde(ax=ax[ax_row, ax_col])
        else:
            df.hist(column=col, ax=ax[ax_row, ax_col])

    plt.tight_layout()


### check null value distribution
def analyse_na_value(df, col_list):
    df_null_check = df[col_list].apply(lambda x: sum(x.isnull()))

    print("# of variables having null value: ", (df_null_check).sum())

    return df_null_check


# find the best PCA

def find_best_pca(df, comp_list=[10]):
    print("# of original features; ", df.shape[1])

    col_list = df.columns.to_list()

    for n_comp in comp_list:

        print("# PCA components: ", n_comp)
        pca_er = PCA(n_components=n_comp, random_state=1903)
        df_pca = pca_er.fit_transform(df[col_list])

        explained_var_r_all = np.cumsum(pca_er.explained_variance_ratio_)
        print("total explained variance is: ", explained_var_r_all[-1])

        if (sum(explained_var_r_all >= 0.8) < 1):
            print("cannot explain 80% variance")
        else:
            variance_80 = np.where(explained_var_r_all >= 0.8)[0][0]
            print("# of components to explain 80% vairances:", variance_80)


# target varaibels analysis
def num_label_dist(df, target_cols):
    df_num_labels = df[target_cols].apply(np.sum, axis=1)
    df_num_labels = df_num_labels.to_frame()
    df_num_labels.hist()

    for num_label in df_num_labels[0].unique():
        pert = (df_num_labels == num_label)[0].sum() / df_num_labels.shape[0]
        print("instances having all {0:d} non-zero labels: {1:5.2f} %".format(int(num_label), pert * 100))

    return df_num_labels


# ========================

def label_hit_dist(df, target_cols):
    df_label_hit = df[target_cols].apply(np.sum, axis=0)
    df_label_hit = df_label_hit.to_frame().T / df.shape[0]
    df_label_hit.loc[0, :].hist()

    print("# of labels with null value count: ", df_label_hit.isnull().sum(axis=1)[0])

    print(df_label_hit.loc[0].quantile(q=np.arange(1, 10) * 0.1))

    return df_label_hit