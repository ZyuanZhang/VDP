import os
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from DataSet import getDataSet

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


RANDOM_STATE = 42
N_BOOTSTRAPS = 100


def grid_search_xgboost(X_train, y_train):
    """
    Perform grid search to find the best hyperparameters for XGBoost.

    :param X_train: Training feature matrix.
    :param y_train: Training labels.
    :return: Best parameters from grid search.
    """
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'n_estimators': [100, 200, 300, 500, 1000],
    }

    grid_search = GridSearchCV(
        estimator=XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric='logloss',
            use_label_encoder=False,
        ),
        param_grid=param_grid,
        scoring='roc_auc',
        n_jobs=10,
        verbose=0,
        cv=5,
    )

    grid_search.fit(X_train, y_train)
    print('The best value of params is: ', grid_search.best_params_)

    return grid_search.best_params_


def evaluate_combined_model(X1_train, X2_train, y_train, X1_test, X2_test, f):
    """
    Train two XGBoost models with bootstrap bagging and combine their predicted probabilities.

    :param X1_train: Training feature matrix for model 1.
    :param X2_train: Training feature matrix for model 2.
    :param y_train: Training labels.
    :param X1_test: Test feature matrix for model 1.
    :param X2_test: Test feature matrix for model 2.
    :param f: Log file object.
    :return: combined_test_prob, y_pred
    """
    # 用训练集均值填补 X1 中的 NaN
    imputer_1 = SimpleImputer(strategy='mean')
    X1_train = imputer_1.fit_transform(X1_train)
    X1_test = imputer_1.transform(X1_test)

    # 用训练集均值填补 X2 中的 NaN
    imputer_2 = SimpleImputer(strategy='mean')
    X2_train = imputer_2.fit_transform(X2_train)
    X2_test = imputer_2.transform(X2_test)

    # 两个特征集都使用 XGBoost 进行参数搜索
    best_params_1 = grid_search_xgboost(X1_train, y_train)
    best_params_2 = grid_search_xgboost(X2_train, y_train)

    f.write("Best params for model 1: " + str(best_params_1) + "\n")
    f.write("Best params for model 2: " + str(best_params_2) + "\n")
    print("Best params found.")

    all_y_prob_1_test = []
    all_y_prob_2_test = []

    for i in range(N_BOOTSTRAPS):
        f.write("Bootstrap: " + str(i) + "\n")
        print("Bootstrap:", i)

        X1_train_bootstrap, y1_train_bootstrap = bootstrapSample(
            X1_train,
            y_train,
            seed=i,
        )
        X2_train_bootstrap, y2_train_bootstrap = bootstrapSample(
            X2_train,
            y_train,
            seed=i + 1,
        )

        # 两个模型都使用 XGBoost
        best_model_1 = XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric='logloss',
            use_label_encoder=False,
            **best_params_1,
        )
        best_model_2 = XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric='logloss',
            use_label_encoder=False,
            **best_params_2,
        )

        best_model_1.fit(X1_train_bootstrap, y1_train_bootstrap)
        best_model_2.fit(X2_train_bootstrap, y2_train_bootstrap)

        y_prob_1_test = best_model_1.predict_proba(X1_test)[:, 1]
        y_prob_2_test = best_model_2.predict_proba(X2_test)[:, 1]

        all_y_prob_1_test.append(y_prob_1_test)
        all_y_prob_2_test.append(y_prob_2_test)

    # 计算 bootstrap 预测概率平均值
    avg_y_prob_1_test = np.mean(all_y_prob_1_test, axis=0)
    avg_y_prob_2_test = np.mean(all_y_prob_2_test, axis=0)

    # 两个 XGBoost 模型的预测概率取平均作为最终组合概率
    combined_test_prob = (avg_y_prob_1_test + avg_y_prob_2_test) / 2
    y_pred = (combined_test_prob >= 0.5).astype(int)

    return combined_test_prob, y_pred


def bootstrapSample(X_train, y_train, seed):
    np.random.seed(seed)
    bootstrap_indices = np.random.choice(
        X_train.shape[0],
        X_train.shape[0],
        replace=True,
    )
    X_train_bootstrap = X_train[bootstrap_indices]
    y_train_bootstrap = y_train[bootstrap_indices]
    return X_train_bootstrap, y_train_bootstrap


def save_file(a, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for row in a:
            f.write(str(row) + '\n')


if __name__ == '__main__':
    (
        Associations_train,
        Labels_train,
        embedding_Sim_train,
        embedding_GO_train,
        Associations_test,
        embedding_Sim_test,
        embedding_GO_test,
    ) = getDataSet()

    outpath = "./Result/XGBoost_bagging"
    os.makedirs(outpath, exist_ok=True)

    with open("./log", "a", buffering=1, encoding='utf-8') as f:
        f.write("------------------------------Start combine XGBoost + XGBoost------------------------------\n")
        print("Evaluating combined model...")

        combined_test_prob, y_pred = evaluate_combined_model(
            np.array(embedding_Sim_train),
            np.array(embedding_GO_train),
            np.array(Labels_train),
            np.array(embedding_Sim_test),
            np.array(embedding_GO_test),
            f,
        )

    save_file(combined_test_prob, outpath + "/combined_test_prob.csv")
    save_file(y_pred, outpath + "/y_pred.csv")
    save_file(Associations_test, outpath + "/Associations_test.csv")