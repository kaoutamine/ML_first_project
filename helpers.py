import csv
import numpy as np
import matplotlib.pyplot as plt

from implementations import mean_squared_error_gd, ridge_regression


def clean_data(data):
    data_cleaned = data

    data_cleaned[data_cleaned == -999] = np.NaN

    # replace NaN's by mean of columns
    medians = np.nanmedian(data_cleaned, axis=0)
    sq_std = np.std(data_cleaned, axis=0) ** 2
    inds = np.where(np.isnan(data_cleaned))
    data_cleaned[inds] = np.take(medians, inds[1])

    # standardize the columns
    data_cleaned = (data_cleaned - medians) / sq_std

    return data


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te


def real_positives(pred, act):
    tot = 0
    good = 0
    for p, a in zip(pred, act):
        if(a == 1.0):
            tot += 1
            if(p == 1.0):
                good += 1
    return good / tot


def real_negatives(pred, act):
    tot = 0
    good = 0
    for p, a in zip(pred, act):
        if(a == 0.0):
            tot += 1
            if(p == 0.0):
                good += 1
    return good/tot


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.
    Args:
        x: numpy array of shape (N,), N is the number of samples.
        degree: integer.

    Returns:
        poly: numpy array of shape (N,d+1)
    """
    degrees = x
    for i in range(degree):
        degree_matrix = x**(i+2)
        degrees = np.c_[degrees, degree_matrix]

    return degrees


def pairwise_column(x):
    x_aug = x
    for i in range(x.shape[1]):
        if i != 1:
            x_aug = np.c_[x_aug, np.multiply(x[:, i], x[:, 1])]
    return x_aug


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, gamma):
    """return the loss of ridge regression for a fold corresponding to k_indices

    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)


    Returns:
            ws - the parameters for each fold 
            precsions - the precision of each fold 
            best k - the k-th fold that leads to the best score 
    >>> cross_validation(np.array([1.,2.,3.,4.]), np.array([6.,7.,8.,9.]), np.array([[3,2], [0,1]]), 1, 2, 3)
    (0.019866645527597114, 0.33555914361295175)
    """

    precisions = []
    ws = []

    for k in range(k_indices.shape[0]):
        train_folds = np.delete(k_indices, k, axis=0)

        train = x[train_folds][0]
        y_train = y[train_folds][0]

        init_w = np.empty([train.shape[1], 1])

        #w, loss_train = ridge_regression(y_train, train, gamma)

        w, loss_train = mean_squared_error_gd(
            y_train, train, init_w, 20, gamma)
        test = x[k_indices[k]]
        y_test = y[k_indices[k]]

        # do the prediction on the k-th fold
        pred = test.dot(w)

        # formating the prediction

        pred = (pred)/(pred.max() - pred.min())
        pred[pred > 0] = 1
        pred[pred < 0] = -1

        # precision
        errors = np.sum(np.abs((y_test - pred)))/2
        precision = 1 - (errors / len(y_test))

        ws.append(w)

        precisions = np.append(precisions, precision)

    return ws, precisions, np.argmax(precisions)


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})
