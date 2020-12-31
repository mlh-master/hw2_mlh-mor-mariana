import numpy as np
import pandas as pd
import seaborn as sns
import sys
import pickle
import math
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix, roc_auc_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm


#function 1 - remove and complete nan values
def rm_and_com_nan(data):
    """
        :param data: Pandas data frame
    """
    diagnosis = data['Diagnosis'].value_counts('Positive') * len(data['Diagnosis'])
    Pos = int(diagnosis[0])
    Neg = int(diagnosis[1])

    nan_idx = []  # removing all Positive subjects with 'nan' values
    for i in list(range(0, len(data['Diagnosis']))):  # for each row
        if (data.loc[i, 'Diagnosis'] == 'Positive'):  # for Positive subjects only
            new_row = data.loc[i, :].dropna()  # look for 'nan' values
            if len(new_row) < 18:
                nan_idx.append(i)
    X_correct = data.drop(nan_idx)

    for key in X_correct.keys():  # for the remaining 'nan' values (Negative subjects) - replacing missing values with real values, according to that feature's probabilities
        probs = X_correct[key].value_counts(normalize=True)  # calculate the probability of each element in the column
        new_vals = []
        for value in X_correct[key]:
            if type(value) == float and math.isnan(value):
                new_vals.append(np.random.choice(list(probs.keys()), p=list(probs.values)))  # if value=NaN -> replace it using the requested distribution
            else:
                new_vals.append(value)  # if value is numeric -> keep it
        X_correct[key] = new_vals

    return X_correct

#function 2 - compare test and train sets' features - table and age histogram
def compare_feat(X,X_train,X_test):
    """
            :param X: The data without nan values, devided to X and Y
            :param X_train,X_test: X after the splitting
    """

    relevant_features = list(X.keys())[1:17]  # selecting the relevant features to be displayed
    train_array = []
    test_array = []
    delta_array = []
    for feat in relevant_features:
        train_stats = X_train[feat].value_counts(normalize=True) * 100
        test_stats = X_test[feat].value_counts(normalize=True) * 100
        if feat == 'Family History':
            train_stats_vals = train_stats[1]
            test_stats_vals = test_stats[1]
        elif feat == 'Gender':
            train_stats_vals = train_stats['Male']
            test_stats_vals = test_stats['Male']
        else:
            train_stats_vals = train_stats['Yes']
            test_stats_vals = test_stats['Yes']
        train_array.append(train_stats_vals)
        test_array.append(test_stats_vals)
        delta_array.append(train_stats_vals - test_stats_vals)

    table = pd.DataFrame(train_array)  # creating an array for the table requested
    table.loc[:, 1] = pd.DataFrame(test_array)
    table.loc[:, 2] = pd.DataFrame(delta_array)
    relevant_features[0] = 'Gender (Male)'
    table.index = relevant_features
    table.columns = ['Train %', 'Test %', 'Delta %']
    round_table = table.astype(int)  # rounding the numbers

    return round_table

#function 3 - label vs. feature analysis
def label_analysis(X,Y):
    """
    :param X: the features' data
    :param Y: the labels

    """

    Pos_group = []
    Neg_group = []

    for i in Y.index:  # deviding the subjects to two groups based on their diagnostic
        if Y[i] == 'Positive':
            Pos_group.append(i)
        else:
            Neg_group.append(i)

    median_age = np.median(X['Age'])
    calcs = {}
    relevant_features = list(X.keys())[0:17]
    for feat in relevant_features:
        pos_yes = 0
        pos_no = 0
        neg_yes = 0
        neg_no = 0
        if feat == 'Family History':
            for j in Pos_group:
                if X[feat][j] == 1:
                    pos_yes = pos_yes + 1
                else:
                    pos_no = pos_no + 1
            for k in Neg_group:
                if X[feat][k] == 1:
                    neg_yes = neg_yes + 1
                else:
                    neg_no = neg_no + 1
        elif feat == 'Age':
            for j in Pos_group:
                if X[feat][j] >= median_age:
                    pos_yes = pos_yes + 1
                else:
                    pos_no = pos_no + 1
            for k in Neg_group:
                if X[feat][k] >= median_age:
                    neg_yes = neg_yes + 1
                else:
                    neg_no = neg_no + 1
        elif feat == 'Gender':
            for j in Pos_group:
                if X[feat][j] == 'Male':
                    pos_yes = pos_yes + 1
                else:
                    pos_no = pos_no + 1
            for k in Neg_group:
                if X[feat][k] == 'Male':
                    neg_yes = neg_yes + 1
                else:
                    neg_no = neg_no + 1
        else:
            for j in Pos_group:
                if X[feat][j] == 'Yes':
                    pos_yes = pos_yes + 1
                else:
                    pos_no = pos_no + 1
            for k in Neg_group:
                if X[feat][k] == 'Yes':
                    neg_yes = neg_yes + 1
                else:
                    neg_no = neg_no + 1
        calcs[feat] = [pos_yes, pos_no, neg_yes, neg_no]

        # create the plots
        n_groups = 2
        negative_group = (neg_yes, neg_no)
        positive_group = (pos_yes, pos_no)
        # create plot
        fig, ax = plt.subplots()
        index = np.arange(n_groups)
        bar_width = 0.15
        opacity = 0.8
        rects1 = plt.bar(index, negative_group, bar_width, alpha=opacity, color='b', label='Negative')
        rects2 = plt.bar(index + bar_width, positive_group, bar_width, alpha=opacity, color='g', label='Positive')
        plt.xlabel(feat)
        plt.ylabel('Count')
        plt.title("%s according to Diagnosis" % feat)
        if feat == 'Gender':
            plt.xticks(index + bar_width / 2, ('Male', 'Female'))
        elif feat == 'Age':
            plt.xticks(index + bar_width / 2, ('Larger than median age', 'Smaller than median age'))
        else:
            plt.xticks(index + bar_width / 2, ('Yes', 'No'))
        plt.legend()
        plt.tight_layout()
        plt.show()

    calcs_DF = pd.DataFrame(calcs, index=['pos_yes', 'pos_no', 'neg_yes', 'neg_no'])

    return calcs_DF


#function 4 - comapre all features between diagnosis
def feature_analysis(calcs_DF):
    """
    
    :param calcs_DF: data frame containing the values of "yes" and "no" answers for each feature, according to the diagnosis 
    """
    pos_yes_group = tuple(calcs_DF.iloc[0, :])
    pos_no_group = tuple(calcs_DF.iloc[1, :])
    neg_yes_group = tuple(calcs_DF.iloc[2, :])
    neg_no_group = tuple(calcs_DF.iloc[3, :])
    relevant_features = list(calcs_DF.keys())[0:17]
    new_relevant_features = relevant_features
    new_relevant_features[0] = 'Age - larger than median'
    new_relevant_features[1] = 'Gender - Male'

    n_groups = 17
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.25
    opacity = 0.8
    rects1 = plt.bar(index, pos_yes_group, bar_width, alpha=opacity, color='b', label='Yes')
    rects2 = plt.bar(index + bar_width, pos_no_group, bar_width, alpha=opacity, color='g', label='No')
    plt.xlabel('features')
    plt.ylabel('Count')
    plt.title("Positive subjects' answers to the questioneres")
    plt.xticks(index, (new_relevant_features), fontsize=8, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    rects1 = plt.bar(index, neg_yes_group, bar_width, alpha=opacity, color='b', label='Yes')
    rects2 = plt.bar(index + bar_width, neg_no_group, bar_width, alpha=opacity, color='g', label='No')
    plt.xlabel('features')
    plt.ylabel('Count')
    plt.title("Negative subjects' answers to the questioneres")
    plt.xticks(index, (new_relevant_features), fontsize=8, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()

    relevant_features = list(calcs_DF.keys())[0:17]

    return relevant_features

#function 5 - encode the data to one hot vector
def hot_vec(X_train, X_test, y_train, y_test, relevant_features):
    """

    :param relevant_features: to be used in the model
    :param X: original data
    :param Y: original output
    :return:
    """
    X_train_hot_vec = dict((el, 0) for el in relevant_features)
    X_train_hot_vec['Family History'] = X_train['Family History']
    new_vals_list = []

    for feat in relevant_features[0:16]:
        if feat == 'Gender':
            X_train_hot_vec[feat] = 1 * (X_train[feat] == 'Male')
        elif feat == 'Age':
            for val in X_train[feat]:
                new_val = math.ceil(val / 10)
                new_vals_list.append(new_val)
        else:
            X_train_hot_vec[feat] = 1 * (X_train[feat] == 'Yes')
    X_train_hot_vec['Age'] = new_vals_list

    X_train_hot_vec_df = pd.DataFrame(X_train_hot_vec)

    Age_train_groups = pd.get_dummies(X_train_hot_vec_df['Age'])
    Age_train_groups.columns = (
    'Age 11-20', 'Age 21-30', 'Age 31-40', 'Age 41-50', 'Age 51-60', 'Age 61-70', 'Age 71-80', 'Age 81-90')
    X_train_hot_vec_df_updated = X_train_hot_vec_df.drop('Age', axis=1)
    for age in Age_train_groups.keys():
        X_train_hot_vec_df_updated[age] = Age_train_groups[age]

    X_test_hot_vec = dict((el, 0) for el in relevant_features)
    X_test_hot_vec['Family History'] = X_test['Family History']
    new_vals_list = []

    for feat in relevant_features[0:16]:
        if feat == 'Gender':
            X_test_hot_vec[feat] = 1 * (X_test[feat] == 'Male')
        elif feat == 'Age':
            for val in X_test[feat]:
                new_val = math.ceil(val / 10)
                new_vals_list.append(new_val)
        else:
            X_test_hot_vec[feat] = 1 * (X_test[feat] == 'Yes')
    X_test_hot_vec['Age'] = new_vals_list

    X_test_hot_vec_df = pd.DataFrame(X_test_hot_vec)

    Age_test_groups = pd.get_dummies(X_test_hot_vec_df['Age'])
    Age_test_groups.columns = ('Age 21-30', 'Age 31-40', 'Age 41-50', 'Age 51-60', 'Age 61-70', 'Age 71-80')
    Age_test_groups['Age 11-20'] = 0
    Age_test_groups['Age 81-90'] = 0
    X_test_hot_vec_df_updated = X_test_hot_vec_df.drop('Age', axis=1)
    for age in Age_test_groups.keys():
        X_test_hot_vec_df_updated[age] = Age_test_groups[age]
    cols = (relevant_features[1:17] + list(Age_train_groups.columns))
    X_test_hot_vec_df_updated = X_test_hot_vec_df_updated[cols]

    y_train_hv = 1 * (y_train == 'Positive')
    y_test_hv = 1 * (y_test == 'Positive')


    return [X_train_hot_vec_df_updated, X_test_hot_vec_df_updated, y_train_hv, y_test_hv]


#function 6 - tuning linear models
def tune_lin_mod(X_train_hv, y_train_hv, pen):
    """

    :param X_train_hv: X as one hot vector
    :param y_train_hv: y as one hot vector
    :param pen: penalty type
    :return:
    """

    def check_penalty(penalty='none'):
        if penalty == 'l1':
            solver = 'liblinear'
        if penalty == 'l2' or penalty == 'none':
            solver = 'lbfgs'
        return solver

    lmbda = np.array([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10])
    n_splits = 5  # 5k cross fold validation
    skf = StratifiedKFold(n_splits=n_splits, random_state=10, shuffle=True)
    max_iter = 2000

    AUC_val = np.zeros((2, len(lmbda)))

    solver = check_penalty(penalty=pen)
    scaler = StandardScaler()

    for idx, lmb in enumerate(lmbda):
        C = 1 / lmb
        log_reg = LogisticRegression(random_state=5, penalty=pen, C=C, max_iter=max_iter, solver=solver)
        with tqdm(total=n_splits, file=sys.stdout, position=0, leave=True) as pbar:
            h = 0  # index per split per lambda
            AUC_val_fold = np.zeros(n_splits)

            for train_index, val_index in skf.split(X_train_hv, y_train_hv):
                pbar.set_description('%d/%d lambda values, processed folds' % ((1 + idx), len(lmbda)))
                pbar.update()
                x_train_fold, x_val_fold = X_train_hv[train_index, :], X_train_hv[val_index, :]  # Here
                y_train_fold, y_val_fold = y_train_hv[train_index], y_train_hv[val_index]
                x_train_fold = scaler.fit_transform(x_train_fold)
                x_val_fold = scaler.transform(x_val_fold)
                log_reg.fit(x_train_fold, y_train_fold)
                y_pred_train = log_reg.predict_proba(x_train_fold)
                y_pred_val = log_reg.predict_proba(x_val_fold)
                AUC_val_fold[h] = roc_auc_score(y_val_fold, y_pred_val[:, 1])
                h += 1

            AUC_val[0, idx] = AUC_val_fold.mean()
            AUC_val[1, idx] = AUC_val_fold.std()

    plt.errorbar(np.log10(lmbda), AUC_val[0, :], yerr=AUC_val[1, :])
    plt.xlabel('$\log_{10}\lambda$')
    plt.ylabel('Test AUC')
    plt.title('AUC Vs. $\log_{10}\lambda$ for pen = %s' % pen)
    plt.show()

    AUC_val_new = list(AUC_val[0, :])
    lb_idx = AUC_val_new.index(max(AUC_val_new))
    lb_value = lmbda[lb_idx]

    return [solver, AUC_val, lb_value]

#function 7 - creating a confusion matrix and calculations
def conf_mat_and_calcs(log_reg, x, y, y_pred, y_pred_proba, model_type, pen, deg, set_type):
    """

    :param log_reg:
    :param x:
    :param y:
    :return:
    """
    plot_confusion_matrix(log_reg, x, y, cmap=plt.cm.Blues)
    plt.grid(False)
    if model_type == 'LR':
        if set_type == 'Train':
            plt.title('Train Set Confusion Matrix for LR with Pen = %s' % pen)
        elif set_type == 'Test':
            plt.title('Test Set Confusion Matrix for LR with Pen = %s' % pen)
    elif model_type == 'SVM':
        if set_type == 'Train':
            plt.title('Train Set Confusion Matrix for Non-linear SVM with Degree = %i' % deg)
        elif set_type == 'Test':
            plt.title('Test Set Confusion Matrix for Non-linear SVM with Degree = %i' % deg)

    calc_TN = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 0]
    calc_FP = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 1]
    calc_FN = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 0]
    calc_TP = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 1]

    TN = calc_TN(y, y_pred)
    FN = calc_FN(y, y_pred)
    FP = calc_FP(y, y_pred)
    TP = calc_TP(y, y_pred)
    Sp = TN / (TN + FP)
    Se = TP / (TP + FN)
    PPV = TP / (TP + FP)
    NPV = TN / (TN + FN)
    Acu = (TP + TN) / (TP + FP + TN + FN)
    F1 = 2 * (PPV * Se) / (PPV + Se)
    AUROC = roc_auc_score(y, y_pred_proba[:, 1])
    Loss = log_loss(y, y_pred_proba)

    return [Acu, F1, AUROC, Loss]


#function 8 - training linear model and calculating performance
def train_lin_mod_and_calc(x_tr, x_tst, y_train_hv, y_test_hv, pen, solver, best_lambda):
    """

    :param X_train_hv: X_train as one hot vector
    :param X_test_hv: X_test as one hot vector
    :param y_train_hv: Y_train as one hot vector
    :param y_test_hv: Y_test as one hot vector
    :param pen: penalty type
    :param solver: solver type
    :param best_lambda: best lambda selected based on the previous tuning
    :return:
    """
    max_iter = 2000

    log_reg = LogisticRegression(random_state=5, penalty=pen, C=1 / best_lambda, max_iter=max_iter, solver=solver)
    log_reg.fit(x_tr, y_train_hv)
    y_pred_test = log_reg.predict(x_tst)
    y_pred_proba_test = log_reg.predict_proba(x_tst)
    y_pred_train = log_reg.predict(x_tr)
    y_pred_proba_train = log_reg.predict_proba(x_tr)

    [Acu_train, F1_train, AUROC_train, Loss_train] = conf_mat_and_calcs(log_reg, x_tr, y_train_hv, y_pred_train, y_pred_proba_train, 'LR', pen, 0, 'Train')
    [Acu_test, F1_test, AUROC_test, Loss_test] = conf_mat_and_calcs(log_reg, x_tst, y_test_hv, y_pred_test, y_pred_proba_test, 'LR', pen, 0, 'Test')

    return [Acu_train, F1_train, AUROC_train, Loss_train, Acu_test, F1_test, AUROC_test, Loss_test]


#function 9 - tuning Non-linear models
def train_nonlin_mod_and_calc(x_tr, x_tst, y_train_hv, y_test_hv, deg):
    """

    :param x_tr: X_train as one hot vector
    :param x_tst: X_test as one hot vector
    :param y_train_hv: Y_train as one hot vector
    :param y_test_hv: Y_test as one hot vector
    :param deg: degree of the model
    :return:
    """
    n_splits = 5 #number of k folds
    skf = StratifiedKFold(n_splits=n_splits, random_state=10, shuffle=True)
    C = np.array([1, 10, 100, 1000])
    svc = SVC(probability=True)
    pipe = Pipeline(steps=[('svm', svc)])
    svm_nonlin = GridSearchCV(estimator=pipe,
                                 param_grid={'svm__kernel': ['rbf', 'poly'], 'svm__C': C, 'svm__degree': [deg],
                                             'svm__gamma': ['scale', 'auto']},
                                 scoring=['accuracy', 'f1', 'precision', 'recall', 'roc_auc'],
                                 cv=skf, refit='roc_auc', verbose=3, return_train_score=True)
    svm_nonlin.fit(x_tr, y_train_hv)
    best_svm_nonlin = svm_nonlin.best_estimator_
    print('Best parameters for this model are:')
    print(svm_nonlin.best_params_)
    y_pred_test = best_svm_nonlin.predict(x_tst)
    y_pred_proba_test = best_svm_nonlin.predict_proba(x_tst)
    y_pred_train = best_svm_nonlin.predict(x_tr)
    y_pred_proba_train = best_svm_nonlin.predict_proba(x_tr)

    [Acu_train, F1_train, AUROC_train, Loss_train] = conf_mat_and_calcs(svm_nonlin, x_tr, y_train_hv, y_pred_train, y_pred_proba_train, 'SVM', 0, deg, 'Train')
    [Acu_test, F1_test, AUROC_test, Loss_test] = conf_mat_and_calcs(svm_nonlin, x_tst, y_test_hv, y_pred_test, y_pred_proba_test, 'SVM', 0, deg, 'Test')

    return [Acu_train, F1_train, AUROC_train, Loss_train, Acu_test, F1_test, AUROC_test, Loss_test]

#function 10 - plot 2d pca data
def plt_2d_pca(X_pca,y,set_type):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    ax.scatter(X_pca[y==0, 0], X_pca[y==0, 1], color='g')
    ax.scatter(X_pca[y==1, 0], X_pca[y==1, 1], color='r')
    ax.legend(('Negative','Positive'))
    ax.plot([0], [0], "ko")
    ax.arrow(0, 0, 0, 1, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')
    ax.arrow(0, 0, 1, 0, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')
    ax.set_xlabel('$U_1$')
    ax.set_ylabel('$U_2$')
    ax.set_title('%s Set 2D PCA' % set_type)


#function 11 - tuning linear models with PCA
def tune_lin_mod_pca(X_train_hv, y_train_hv, pen, pca):
    """

    :param X_train_hv:
    :param y_train_hv:
    :param pen:
    :return:
    """

    def check_penalty(penalty='none'):
        if penalty == 'l1':
            solver = 'liblinear'
        if penalty == 'l2' or penalty == 'none':
            solver = 'lbfgs'
        return solver

    lmbda = np.array([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10])
    n_splits = 5  # 5k cross fold validation
    skf = StratifiedKFold(n_splits=n_splits, random_state=10, shuffle=True)
    max_iter = 2000

    AUC_val = np.zeros((2, len(lmbda)))

    solver = check_penalty(penalty=pen)
    scaler = StandardScaler()

    for idx, lmb in enumerate(lmbda):
        C = 1 / lmb
        log_reg = LogisticRegression(random_state=5, penalty=pen, C=C, max_iter=max_iter, solver=solver)
        with tqdm(total=n_splits, file=sys.stdout, position=0, leave=True) as pbar:
            h = 0  # index per split per lambda
            AUC_val_fold = np.zeros(n_splits)

            for train_index, val_index in skf.split(X_train_hv, y_train_hv):
                pbar.set_description('%d/%d lambda values, processed folds' % ((1 + idx), len(lmbda)))
                pbar.update()
                x_train_fold, x_val_fold = X_train_hv[train_index, :], X_train_hv[val_index, :]  # Here
                y_train_fold, y_val_fold = y_train_hv[train_index], y_train_hv[val_index]

                X_train_ = scaler.fit_transform(x_train_fold)
                X_val_ = scaler.transform(x_val_fold)
                x_train_fold = pca.fit_transform(X_train_)
                x_val_fold = pca.transform(X_val_)

                log_reg.fit(x_train_fold, y_train_fold)
                y_pred_train = log_reg.predict_proba(x_train_fold)
                y_pred_val = log_reg.predict_proba(x_val_fold)
                AUC_val_fold[h] = roc_auc_score(y_val_fold, y_pred_val[:, 1])
                h += 1

            AUC_val[0, idx] = AUC_val_fold.mean()
            AUC_val[1, idx] = AUC_val_fold.std()

    plt.errorbar(np.log10(lmbda), AUC_val[0, :], yerr=AUC_val[1, :])
    plt.xlabel('$\log_{10}\lambda$')
    plt.ylabel('Test AUC')
    plt.title('AUC Vs. $\log_{10}\lambda$ for pen = %s' % pen)
    plt.show()

    AUC_val_new = list(AUC_val[0, :])
    lb_idx = AUC_val_new.index(max(AUC_val_new))
    lb_value = lmbda[lb_idx]

    return [solver, AUC_val, lb_value]

