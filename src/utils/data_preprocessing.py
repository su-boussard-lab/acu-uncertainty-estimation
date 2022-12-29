""" This file contains the reading and preprocessing steps for the data
Author:
    Claudio Fanconi
"""
from typing import Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def preprocessing(
    feature_path: str,
    label_path: str,
    train_ids_path: str,
    test_ids_path: str,
    outcome: str,
    scale: bool = True,
    pca: bool = False,
) -> Tuple:
    """Preprocesses the data and scales it
    NOTE: data should be in CSV format with "PAT_DEID" column as patient identifier
    Args:
        feature_path (str): path from where the data is taken
        label_path (str): path from where the labels are taken
        train_ids_path (str): path to a CSV that contains the train patient DEIDs
        test_ids_path (str): path to a CSV that contains the test patient DEIDs
        outcome (str): name of the label column
        scale (bool, True): flag to scale data or not
        pca (bool, False): run Principal component analysis to reduce features
    returns:
        X_train (np.ndarray): training features [n_samples_train, d_features]
        y_train (np.ndarray): training labels [n_samples_train]
        X_test (np.ndarray): testing features [n_samples_test, d_features]
        y_test (np.ndarray): testing labels [n_samples_test]
    """

    print("Preprocessing")
    # load features, sort by PAT_ID
    feature_matrix = (
        pd.read_csv(feature_path, low_memory=False)
        .sort_values(by="PAT_DEID")
        .set_index("PAT_DEID")
        # .drop("DEMO_INDEX_PRE_CHE", axis=1)
    )

    # load outcomes, pick right one, match index
    # Make sure that outcomes labels line up with feature matrix
    outcomes = pd.read_csv(label_path).set_index("PAT_DEID")
    labels = outcomes[outcome].reindex(feature_matrix.index)

    TRAIN_IDS = pd.read_csv(train_ids_path)["PAT_DEID"]
    TEST_IDS = pd.read_csv(test_ids_path)["PAT_DEID"]

    X_train = feature_matrix.loc[TRAIN_IDS]
    X_test = feature_matrix.loc[TEST_IDS]
    y_train = labels.loc[TRAIN_IDS]
    y_test = labels.loc[TEST_IDS]

    print(
        f"There are {feature_matrix.shape[0]} patients and {feature_matrix.shape[1]} features in full feature matrix."
    )
    print(
        f"There are {X_train.shape[0]} patients and {X_train.shape[1]} features in Train set."
    )
    print(
        f"There are {X_test.shape[0]} patients and {X_test.shape[1]} features in Test set."
    )

    if scale:
        X_train, X_test = scale_data(X_train, X_test)

    if pca:
        print("Running PCA")
        pcamod = PCA(pca).fit(X_train)
        X_train = pcamod.transform(X_train)
        X_test = pcamod.transform(X_test)

    return X_train, X_test, y_train, y_test


def scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple:
    """ "Scale the data to be zero-center and unit-variance.
        NOTE: the scaler is only fitted on the training data, and then all the data is transformed,
        This is done to prevent dataleakage in the test set
    Args:
        X_train (pd.DataFrame): training tabular data that is not yet scaled.
        X_test (pd.DataFrame): testing tabular data that is not yet scaled.
    Returns:
        X_train (np.ndarray): data frame with normalized data for training set
        X_test (np.ndarray): data frame with normalized data for test set
    """
    scaler = StandardScaler()
    col_names = [c for c in X_train.columns]
    scaler.fit(X_train[col_names])
    X_train_ = scaler.transform(X_train[col_names])
    X_train[col_names] = X_train_
    X_test_ = scaler.transform(X_test[col_names])
    X_test[col_names] = X_test_

    return X_train.values, X_test.values
