import csv
import os
import random
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    precision_recall_curve, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from DataSet import getDataSet

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def grid_search_xgboost(X_train, y_train):
    """
    Perform grid search to find the best hyperparameters for XGBoost.

    :param X_train: Training feature matrix.
    :param y_train: Training labels.
    :return: Best estimator from grid search.
    """
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'n_estimators': [100, 200, 300, 500, 1000],
    }

    grid_search = GridSearchCV(
        estimator=XGBClassifier(random_state=42),
        param_grid=param_grid,
        scoring='roc_auc',
        n_jobs=10,
        verbose=0
    )

    grid_search.fit(X_train, y_train)
    # print('The grid_search result: ', grid_search.cv_results_)
    # print('\n')
    print('The best value of params is: ', grid_search.best_params_)

    return grid_search.best_params_


def grid_search_rf(X_train, y_train):
    """
    Perform grid search to find the best hyperparameters for Random Forest.

    :param X_train: Training feature matrix.
    :param y_train: Training labels.
    :return: Best estimator from grid search.
    """
    param_grid = {
        'n_estimators': [100, 200, 300, 500, 1000],
        'max_depth': [10, 20 ,30 , 50],
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        scoring='roc_auc',
        n_jobs=10,  # Use all available cores
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    return grid_search.best_params_


def evaluate_combined_model(X1, X2, y, f, n_splits=5):
    """
    Perform k-fold cross-validation with grid search for combining two XGBoost models,
    and record PR and ROC data for each fold.

    :param X1: Feature matrix for model 1.
    :param X2: Feature matrix for model 2.
    :param y: Labels.
    :param f: Log file object.
    :param n_splits: Number of folds for cross-validation.
    :param w_values: Array of weight values for combining models.
    :return: Evaluation metrics across all folds.
    """

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies, precisions, recalls, f1_scores, aucs, specificities, best_ws, prs = [], [], [], [], [], [], [], []
    pr_data, roc_data = [], []
    fold_results = {}

    fold = 1
    for train_index, test_index in kf.split(X1, y):
        print("Fold:", str(fold))
        f.write("Fold: " + str(fold) + "\n")

        # Split train and test sets
        X1_train, X1_test = X1[train_index], X1[test_index]
        X2_train, X2_test = X2[train_index], X2[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Grid search for the best model for each feature set
        best_params_1 = grid_search_rf(X1_train, y_train)
        best_params_2 = grid_search_xgboost(X2_train, y_train)

        f.write("Best params finded. " + "\n")
        print("Best params finded. ")

        all_y_prob_1_test = []
        all_y_prob_2_test = []
        for i in range(100):
            f.write("Bootstrap: " + str(i) + "\n")
            print("Bootstrap: ", i)
            X1_train_bootstrap, y1_train_bootstrap = bootstrapSample(X1_train, y_train, i)
            X2_train_bootstrap, y2_train_bootstrap = bootstrapSample(X2_train, y_train, i+1)
            #Train model with best params and best weight on the training set
            best_model_1 = RandomForestClassifier(random_state=42, **best_params_1)
            best_model_2 = XGBClassifier(random_state=42, **best_params_2)
            best_model_1.fit(X1_train_bootstrap, y1_train_bootstrap)
            best_model_2.fit(X2_train_bootstrap, y2_train_bootstrap)

            # Combine probabilities using the best weight and evaluate on the test set
            y_prob_1_test = best_model_1.predict_proba(X1_test)[:, 1]
            y_prob_2_test = best_model_2.predict_proba(X2_test)[:, 1]

            all_y_prob_1_test.append(y_prob_1_test)
            all_y_prob_2_test.append(y_prob_2_test)

        # 计算预测结果的平均值
        avg_y_prob_1_test = np.mean(all_y_prob_1_test, axis=0)
        avg_y_prob_2_test = np.mean(all_y_prob_2_test, axis=0)

        # 最终的组合概率
        combined_test_prob = (avg_y_prob_1_test + avg_y_prob_2_test ) / 2

        y_pred = (combined_test_prob >= 0.5).astype(int)

        # Calculate evaluation metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        aucs.append(roc_auc_score(y_test, combined_test_prob))
        specificities.append(tn / (tn + fp))

        # PR and ROC data
        precision, recall, pr_thresholds = precision_recall_curve(y_test, combined_test_prob)
        fpr, tpr, roc_thresholds = roc_curve(y_test, combined_test_prob)
        prs.append(auc(recall, precision))

        pr_data.append({"fold": fold - 1, "precision": precision, "recall": recall, "thresholds": pr_thresholds})
        roc_data.append({"fold": fold - 1, "fpr": fpr, "tpr": tpr, "thresholds": roc_thresholds})
        fold_results[fold] ={
            "accuracy": accuracies[-1],
            "precision": precisions[-1],
            "recall": recalls[-1],
            "specificity": specificities[-1],
            "f1_score": f1_scores[-1],
            "auc": aucs[-1],
            "pr": prs[-1],
        }

        fold += 1

    return {
        "metrics": {
            "accuracy": np.mean(accuracies),
            "precision": np.mean(precisions),
            "recall": np.mean(recalls),
            "specificity": np.mean(specificities),
            "f1_score": np.mean(f1_scores),
            "auc": np.mean(aucs),
            "pr": np.mean(prs),
        },
        "fold_results": fold_results,
        "pr_data": pr_data,
        "roc_data": roc_data
    }


def bootstrapSample(X_train, y_train, seed):
    np.random.seed(seed)
    bootstrap_x1 = np.random.choice(X_train.shape[0], X_train.shape[0], replace=True)
    x1_train_bootstrap = X_train[bootstrap_x1]
    y1_train = y_train[bootstrap_x1]
    return x1_train_bootstrap, y1_train


def savefile(results, outpath, filename):
    df_results = pd.DataFrame(results, index=['values'])
    df_results.to_csv(outpath + "/" + filename, sep='\t')


def savefileForfold(fold_results, outpath, filename):
    # 写入 CSV 文件
    with open(outpath+'/'+filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写入表头
        writer.writerow(['Fold', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1_Score', 'AUC', "AUCPR"])
        # 写入数据
        for fold, metrics in fold_results.items():
            writer.writerow([fold] + list(metrics.values()))


def save_pr_roc_data(pr_data, roc_data, outpath):
    """Save PR and ROC data for each fold."""
    df_pr = pd.DataFrame()
    for pr in pr_data:
        df = pd.DataFrame({
            "precision": pr["precision"],
            "recall": pr["recall"],
        })
        df.rename(columns={"precision": "Precision{}".format(pr["fold"]), "recall": "Recall{}".format(pr["fold"])}, inplace=True)
        df_pr = pd.concat([df_pr, df], axis=1)

    df_roc = pd.DataFrame()
    for roc in roc_data:
        df = pd.DataFrame({
            "fpr": roc["fpr"],
            "tpr": roc["tpr"],
        })
        df.rename(columns={"fpr": "FPR{}".format(roc["fold"]), "tpr": "TPR{}".format(roc["fold"])}, inplace=True)
        df_roc = pd.concat([df_roc, df], axis=1)

    #save filew
    df_pr.to_csv(outpath + "/pr_data.csv", sep='\t')
    df_roc.to_csv(outpath + "/roc_data.csv", sep='\t')

if __name__ == '__main__':
    Associations,Labels,embedding_Sim,embedding_GO = getDataSet()

    outpath = "./Result/XGBoost_bagging"
    os.makedirs(outpath, exist_ok=True)

    with open("./log", "a", buffering=1) as f:
        f.write("------------------------------Start combine {}------------------------------\n")
        print("Evaluating combined model...")
        results = evaluate_combined_model(
            np.array(embedding_Sim),
            np.array(embedding_GO),
            np.array(Labels),
            f
        )

        # Save metrics
        savefile(results["metrics"], outpath, filename="results_combined.csv")
        savefileForfold(results["fold_results"], outpath, filename="results_combined_fold.csv")

        # Save PR and ROC data
        save_pr_roc_data(results["pr_data"], results["roc_data"], outpath)

        print("Results for combined model:", results["metrics"])
        f.write("Results for combined model: " + str(results["metrics"]) + "\n")
