################算法选择################  
./Algorithm 算法选择文件夹  
./Algorithm/Result 算法选择结果文件夹  
./Algorithm/Result/Original result 算法选择原始结果文件夹  
./Algorithm/Result/RF-embedding_Sim modelA算法选择原始结果文件夹  
./Algorithm/Result/XGB-embedding_GO modelB算法选择原始结果文件夹  
./Algorithm/DataSet.py 算法选择核心代码，用于读取数据集  
./Algorithm/extractEmbbeding.py 算法选择核心代码，用于编码特征  
./Algorithm/Model_ANN.py 算法选择核心代码，用于模型训练评估  
./Algorithm/Model_RF.py 算法选择核心代码，用于模型训练评估  
./Algorithm/Model_SVM.py 算法选择核心代码，用于模型训练评估  

################数据################  
./DataNegAssociation_generateByKmeans_closestCenter.xlsx 负样本   
./DataPosAssociation.xlsx 正样本  
./DataVDA_interaction.csv 关联对  

################特征################  
./Features/GenomeAndMeshSimilarty 基因组和语义相似性  
./Features/GO 功能相似性    

################独立测试集测试################    
./Independent test set testing  
Associations_test.xlsx 待预测关联对    
DataSet.py 提取数据脚本    
DataSet.xlsx 待预测关联对    
extractEmbbeding.py 提取特征脚本    
Model_XGB_combine_bagging.py 训练评估脚本    
Processing DataSet.py 处理数据脚本     
Scrape disease categories.py 爬取疾病mesh树脚本   
./Independent test set testing/Result/XGBoost_bagging 结果文件  
y_pred.csv 最终预测标签  
combined_test_prob.csv  最终预测概率
Associations_test.csv  预测关联对  

################交叉验证################   
./Test  
DataSet.py  提取数据脚本  
extractEmbbeding.py  提取特征脚本  
Model_XGB_combine_bagging.py 训练评估脚本  
Model_XGB_combine_bagging500.py 训练评估脚本  
./Test/Result/XGBoost_bagging 结果文件夹  
