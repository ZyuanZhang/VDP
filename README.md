# Scripts for Virus Disease Predictor (VDP)

## Selection of Algorithms

| Folder | Description |
|--------|-------------|
| `./Algorithm` | Main folder for algorithm selection |
| `./Algorithm/Result` | Folder for results of algorithm selection |
| `./Algorithm/Result/Original result` | Folder for original results of algorithm selection |
| `./Algorithm/Result/RF-embedding_Sim modelA` | Folder for original results (RF-embedding_Similarity model A) |
| `./Algorithm/Result/XGB-embedding_GO modelB` | Folder for original results (XGBoost-embedding_GO model B) |
| `./Algorithm/DataSet.py` | Core script for algorithm selection, used for reading datasets |
| `./Algorithm/extractEmbbeding.py` | Core script for algorithm selection, used for encoding features |
| `./Algorithm/Model_ANN.py` | Core script for algorithm selection, used for model training and evaluation (ANN) |
| `./Algorithm/Model_RF.py` | Core script for algorithm selection, used for model training and evaluation (RF) |
| `./Algorithm/Model_SVM.py` | Core script for algorithm selection, used for model training and evaluation (SVM) |


## Data Files

| File | Description |
|------|-------------|
| `./DataNegAssociation_generateByKmeans_closestCenter.xlsx` | Negative samples |
| `./DataPosAssociation.xlsx` | Positive samples |
| `./DataVDA_interaction.csv` | Virus-disease association pairs |


## Feature Files

| Folder | Description |
|--------|-------------|
| `./Features/GenomeAndMeshSimilarty` | Genome similarity and MeSH semantic similarity |
| `./Features/GO` | Gene Ontology functional similarity |


## Independent Test Set Testing

| Folder/File | Description |
|-------------|-------------|
| `./Independent test set testing` | Main folder for independent test set testing |
| `Associations_test.xlsx` | Association pairs to be predicted |
| `DataSet.py` | Script for data extraction |
| `DataSet.xlsx` | Processed dataset for association pairs to be predicted |
| `extractEmbbeding.py` | Script for feature extraction |
| `Model_XGB_combine_bagging.py` | Script for training and evaluation (XGBoost with Bagging) |
| `Processing DataSet.py` | Script for data processing |
| `Scrape disease categories.py` | Script for scraping disease MeSH tree |
| `./Independent test set testing/Result/XGBoost_bagging` | Folder for XGBoost_Bagging results |
| `y_pred.csv` | Final predicted labels |
| `combined_test_prob.csv` | Final predicted probabilities |
| `Associations_test.csv` | Predicted association pairs |


## Cross-Validation

| Folder/File | Description |
|-------------|-------------|
| `./Test` | Main folder for cross-validation testing |
| `DataSet.py` | Script for data extraction |
| `extractEmbbeding.py` | Script for feature extraction |
| `Model_XGB_combine_bagging.py` | Script for training and evaluation (XGBoost with Bagging) |
| `Model_XGB_combine_bagging500.py` | Script for training and evaluation (XGBoost with Bagging, 500 rounds) |
| `./Test/Result/XGBoost_bagging` | Folder for XGBoost_Bagging results |