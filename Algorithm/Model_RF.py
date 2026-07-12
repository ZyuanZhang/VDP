import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    precision_recall_curve, auc, roc_curve
from DataSet import getDataSet


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
        n_jobs=-1,  # Use all available cores
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print('The best value of params is: ', grid_search.best_params_)
    return grid_search.best_estimator_


def evaluate_model_with_search(X, y, n_splits=5):
    """
    Perform k-fold cross-validation with grid search for hyperparameter tuning.

    :param X: Feature matrix.
    :param y: Labels.
    :param n_splits: Number of folds for cross-validation.
    :return: Evaluation metrics (accuracy, precision, recall, F1 score, AUC) across all folds.
    """
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies, precisions, recalls, f1_scores, aucs, specificities,auprs,pr_curves,roc_curves = [], [], [], [], [], [],[],[],[]

    fold = 1
    for train_index, test_index in kf.split(X, y):
        print("Fold:",str(fold))
        fold += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Perform grid search to find the best Random Forest model
        best_model = grid_search_rf(X_train, y_train)

        # Make predictions
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        # Calculate evaluation metrics
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, average='binary'))
        recalls.append(recall_score(y_test, y_pred, average='binary'))
        f1_scores.append(f1_score(y_test, y_pred, average='binary'))
        aucs.append(roc_auc_score(y_test, y_prob))
        specificities.append(tn / (tn + fp))  # Specificity
        # Calculate precision-recall curve and AUPR
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        auprs.append(auc(recall, precision))
        pr_curves.append((precision, recall))
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_curves.append((fpr, tpr))

    return {
        "accuracy": np.mean(accuracies),
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "specificity": np.mean(specificities),
        "f1_score": np.mean(f1_scores),
        "auc": np.mean(aucs),
        "aupr": np.mean(auprs),
        "pr_curves": pr_curves,
        "roc_curves": roc_curves
    }


def savefile(results, pr_curves, roc_curves, outpath, filename):
    outpath = outpath+"-"+filename.split('.')[0]
    os.makedirs(outpath, exist_ok=True)

    # Save metrics
    df_results = pd.DataFrame(results, index=['values'])
    df_results.to_csv(outpath + "/" + filename, sep='\t')

    # Save PR curve data for each fold
    pr_curve_path = outpath + "/pr_curves/"
    os.makedirs(pr_curve_path, exist_ok=True)
    for i, (precision, recall) in enumerate(pr_curves):
        pr_df = pd.DataFrame({"precision": precision, "recall": recall})
        pr_df.to_csv(pr_curve_path + f"fold_{i + 1}_pr_curve.csv", index=False)

    # Save ROC curve data for each fold
    roc_curve_path = outpath + "/roc_curves/"
    os.makedirs(roc_curve_path, exist_ok=True)
    for i, (fpr, tpr) in enumerate(roc_curves):
        roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
        roc_df.to_csv(os.path.join(roc_curve_path, f"fold_{i + 1}_roc_curve.csv"), index=False)


if __name__ == '__main__':
    Associations,Labels,embedding_Sim,embedding_GO = getDataSet()
    print(np.array(embedding_Sim).shape)
    print(np.array(embedding_GO).shape)

    outpath = "./Result/RF"

    print("Evaluating embedding_Sim...")
    results_embedding_Sim = evaluate_model_with_search(np.array(embedding_Sim), np.array(Labels))
    pr_curves = results_embedding_Sim.pop("pr_curves")  # Extract PR curve data
    roc_curves = results_embedding_Sim.pop("roc_curves")  # Extract ROC curve data
    savefile(results_embedding_Sim, pr_curves,roc_curves, outpath, filename="embedding_Sim.csv")
    print("Results for embedding_Sim:", results_embedding_Sim)

    # print("Evaluating embedding_GO...")
    # results_embedding_GO = evaluate_model_with_search(np.array(embedding_GO), np.array(Labels))
    # pr_curves = results_embedding_GO.pop("pr_curves")  # Extract PR curve data
    # savefile(results_embedding_GO, pr_curves, outpath, filename="embedding_GO.csv")
    # print("Results for embedding_GO:", results_embedding_GO)
