import numpy as np
from implementations import *
from helpers import *


def main():
    DATA_TRAIN_PATH = 'train.csv'
    DATA_TEST_PATH = 'test.csv'

    print("starting data importation")
    # import the data
    y_train, tX_train, ids_train = load_csv_data(DATA_TRAIN_PATH)
    y_test, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

    # cleaning the training and test set
    tX_train_clean = clean_data(tX_train)
    tX_test_clean = clean_data(tX_test)
    # adding features with polynomial basis function
    tX_train_clean_poly = build_poly(tX_train_clean, 3)
    tX_test_clean_poly = build_poly(tX_test_clean, 3)
    # adding the pairwise multiplication
    tX_train_clean_pc = pairwise_column(tX_train_clean)
    tX_test_clean_pc = pairwise_column(tX_test_clean)
    # mixing it all together
    tX_test_clean_aug = np.c_[tX_test_clean_pc[:, 30:], tX_test_clean_poly]
    tX_train_clean_aug = np.c_[tX_train_clean_pc[:, 30:], tX_train_clean_poly]

    # trying the cross validation
    k_indices = build_k_indices(y_train, 10, 10)
    gamma = 0.1
    ws, precisions, best_k = cross_validation(
        y_train, tX_train_clean_aug, k_indices, gamma)
    # print(precisions)
    cross_val_std = np.std(precisions)
    cross_val_mean = np.mean(precisions)

    print("the average mean of cross validation is " + str(cross_val_mean) +
          "and the std is " + str(cross_val_std))

    w = ws[best_k]

    pred_test = tX_test_clean_aug.dot(w)

    # format the predictions
    pred_test = pred_test / (pred_test.max() - pred_test.min())
    pred_test[pred_test > 0] = 1
    pred_test[pred_test < 0] = -1
    print("we set everything to 1 or -1")

    create_csv_submission(ids_test, pred_test, "predictions.csv")
    print("csv successfully written")


if __name__ == '__main__':
    main()
