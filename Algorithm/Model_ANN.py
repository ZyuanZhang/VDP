import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    precision_recall_curve, auc
from DataSet import getDataSet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def train_nn(model, optimizer, criterion, train_loader, val_loader, epochs):
    """
    Train the neural network model.

    :param model: PyTorch model.
    :param optimizer: Optimizer for training.
    :param criterion: Loss function.
    :param train_loader: DataLoader for training data.
    :param val_loader: DataLoader for validation data.
    :param epochs: Number of epochs to train.
    :return: Validation AUC for the trained model.
    """
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch.float())
            loss = criterion(outputs, y_batch.float().view(-1, 1))
            loss.backward()
            optimizer.step()

    if val_loader == None:
        return model
    # Evaluate on validation data
    model.eval()
    y_probs, y_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch.float()).squeeze()
            y_probs.extend(outputs.cpu().numpy())
            y_true.extend(y_batch.cpu().numpy())
    auc = roc_auc_score(y_true, y_probs)
    return auc

def grid_search_nn(X_train, y_train):
    """
    Perform grid search to find the best hyperparameters for the neural network.

    :param X_train: Training feature matrix.
    :param y_train: Training labels.
    :return: Best parameters and the corresponding trained model.
    """
    param_grid = {
        'hidden_size': [128, 256, 512, 1024],  #根据经验设置，防止模型过拟合或者欠拟合
        'learning_rate': [0.01, 0.001, 0.0001, 0.00001],
        'epochs': [25, 50, 75, 100]
    }

    best_auc, best_params = 0, None
    input_size = X_train.shape[1]

    # Convert data to PyTorch tensors
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    for hidden_size in param_grid['hidden_size']:
        for learning_rate in param_grid['learning_rate']:
            for epochs in param_grid['epochs']:
                print("Current params: "+'hidden_size:', hidden_size, 'learning_rate:', learning_rate, 'epochs:', epochs)
                # Initialize the model with current parameters
                model = SimpleNN(input_size=input_size, hidden_size=hidden_size, output_size=1).to(device)
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                criterion = nn.BCELoss()

                # Train the model and get AUC
                auc = train_nn(model, optimizer, criterion, train_loader, val_loader, epochs)
                if auc > best_auc:
                    best_auc = auc
                    best_params = {'hidden_size': hidden_size, 'learning_rate': learning_rate, 'epochs': epochs}

    # Retrain the best model on the entire dataset
    print('The best value of params is:', best_params)
    best_model = SimpleNN(input_size=input_size, hidden_size=best_params['hidden_size'], output_size=1).to(device)
    optimizer = optim.Adam(best_model.parameters(), lr=best_params['learning_rate'])
    criterion = nn.BCELoss()

    # Train on the entire training data
    full_train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    best_model = train_nn(best_model, optimizer, criterion, full_train_loader, None, best_params['epochs'])

    return best_model

def evaluate_model_with_search(X, y,n_splits=5):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies, precisions, recalls, f1_scores, aucs, specificities,auprs,pr_curves = [], [], [], [], [], [], [], []

    fold = 1
    for train_index, test_index in kf.split(X, y):
        print("Fold:", str(fold))
        fold += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Perform grid search to find the best model
        best_model = grid_search_nn(X_train, y_train)

        # Make predictions
        best_model.eval()
        X_test_tensor = torch.tensor(X_test).float().to(device)
        y_probs = best_model(X_test_tensor).detach().cpu().numpy()
        y_pred = (y_probs > 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        # Calculate evaluation metrics
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        aucs.append(roc_auc_score(y_test, y_probs))
        specificities.append(tn / (tn + fp))  # Specificity
        # Calculate precision-recall curve and AUPR
        precision, recall, _ = precision_recall_curve(y_test, y_probs)
        auprs.append(auc(recall, precision))
        pr_curves.append((precision, recall))

    return {
        "accuracy": np.mean(accuracies),
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "specificity": np.mean(specificities),
        "f1_score": np.mean(f1_scores),
        "auc": np.mean(aucs),
        "aupr": np.mean(auprs),
        "pr_curves": pr_curves
    }

def savefile(results, pr_curves, outpath, filename):
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


if __name__ == '__main__':
    Associations,Labels,embedding_Sim,embedding_GO = getDataSet()
    print(np.array(embedding_Sim).shape)
    print(np.array(embedding_GO).shape)

    outpath = "./Result/ANN"

    print("Evaluating embedding_Sim...")
    results_embedding_Sim = evaluate_model_with_search(np.array(embedding_Sim), np.array(Labels))
    pr_curves = results_embedding_Sim.pop("pr_curves")  # Extract PR curve data
    savefile(results_embedding_Sim, pr_curves, outpath, filename="embedding_Sim.csv")
    print("Results for embedding_Sim:", results_embedding_Sim)

    print("Evaluating embedding_GO...")
    results_embedding_GO = evaluate_model_with_search(np.array(embedding_GO), np.array(Labels))
    pr_curves = results_embedding_GO.pop("pr_curves")  # Extract PR curve data
    savefile(results_embedding_GO, pr_curves, outpath, filename="embedding_GO.csv")
    print("Results for embedding_GO:", results_embedding_GO)


