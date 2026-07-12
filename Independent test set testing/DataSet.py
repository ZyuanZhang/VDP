import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from extractEmbbeding import extractEmbbedingOfGO,extractEmbbedingOfSim
from itertools import product

# 这里修改为我们重新计算的特征，不用去掉四个病毒的相似性特征，因为我拿到新病毒基因组这些都是可以计算出来的。
df_ds = pd.read_csv('../Features/GenomeAndMeshSimilarty/disease_SemanticSimilarity_matrix.csv', sep='\t', index_col=0)
df_vs = pd.read_csv('../Features/GenomeAndMeshSimilarty/virus_GenomeSimilarity_matrix.csv', sep='\t', index_col=0)

df_ds_GO = pd.read_csv('../Features/GO/disease_FunctionSimilarity_matrix_Sem.csv', sep='\t', index_col=0)
df_vs_GO = pd.read_csv('../Features/GO/virus_FunctionSimilarity_matrix_Sem.csv', sep='\t', index_col=0)
df_vds_GO = pd.read_csv('../Features/GO/cross_FunctionSimilarity_matrix_Sem.csv', sep='\t', index_col=0)


def get_Kmeans_result(NegAssociation,k):
    list_embedding = []
    association_names = []
    for association in NegAssociation:
        virus, disease = int(association.split(',')[0]), association.split(',')[1]
        embedding_smi = extractEmbbedingOfSim(virus,disease,df_ds,df_vs)  #提取的特征
        list_embedding.append(embedding_smi)
        association_names.append(association)
    embeddings = np.array(list_embedding).reshape(-1, 484)  ##提取的特征的形状
    tsne = TSNE(n_components=2, random_state=42)
    X_sne = tsne.fit_transform(embeddings)
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans_result = kmeans.fit_predict(X_sne)
    result_df = pd.DataFrame({
        "Association": association_names,
        "Cluster": kmeans_result
    })

    cluster_centers = kmeans.cluster_centers_
    cluster_centers_df = pd.DataFrame(
        cluster_centers,
        columns=["Center_X", "Center_Y"]
    )
    cluster_centers_df["Cluster"] = range(k)

    # 计算每个样本到对应簇中心的欧氏距离
    distances = np.linalg.norm(X_sne - cluster_centers[kmeans_result], axis=1)  #2范数，这里即为欧氏距离

    # 找出每个簇中离簇中心最近的样本
    closest_samples_indices = []
    for i in range(k):
        # 找出簇 i 中距离簇中心最近的样本索引
        cluster_indices = np.where(kmeans_result == i)[0]
        closest_index = cluster_indices[np.argmin(distances[cluster_indices])]
        closest_samples_indices.append(closest_index)

    # 获取最近的样本对应的名称
    closest_samples = pd.DataFrame({
        "Association": [association_names[i] for i in closest_samples_indices],
        "Cluster": [kmeans_result[i] for i in closest_samples_indices],
        "Closest_Sample_Distance": [distances[i] for i in closest_samples_indices]
    })

    #所有样本对应的坐标
    df_site = pd.DataFrame(X_sne,columns=["X","Y"])
    df_site["Association"] = association_names

    return result_df,cluster_centers_df,closest_samples,df_site

def NegSampleWihtKmeanCenter():
    # df_pos = pd.read_excel('../Data/PosAssociation.xlsx')
    # df_vda = pd.read_csv('../Data/VDA_interaction.csv', sep='\t', index_col=0)
    df_vda = pd.read_csv('../newData/最终筛选出来的关联对2.csv', sep='\t')
    # 10566, 290028, 1891763, 1891767, 12475, 11292
    df_vda = df_vda[~df_vda['种id'].isin([10566,290028,1891763,1891767,12475,11292])]
    list_virus,list_disease = df_vda["种id"].tolist(),df_vda["mesh_id"].tolist()
    list_all = []
    for virus in list_virus:
        for disease in list_disease:
            list_all.append(str(virus)+','+disease)
    # 训练的正负样本和测试的正样本
    PosAssociation_test = list(df_vda.apply(lambda x: str(x['种id'])+','+x['mesh_id'], axis=1))
    df_pos = pd.read_excel('../Data/PosAssociation.xlsx')
    df_neg = pd.read_excel('../KmeaNegativeSample/NegAssociationKmeans_closestCenter_nostd_Sim.xlsx')
    PosAssociation_train = df_pos['Unnamed: 0'].tolist()
    NegAssociation_train = df_neg['Association'].tolist()
    Association = PosAssociation_test+PosAssociation_train+NegAssociation_train
    # 去除训练的正负样本和测试的正样本
    NegAssociation = list(set(list_all) - set(Association))
    print(len(PosAssociation_test)) #110
    print(len(NegAssociation)) #1827

    result_df, cluster_centers_df, closest_samples, df_site = get_Kmeans_result(NegAssociation, k=len(PosAssociation_test))
    output_path = './KmeaNegativeSample'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    result_df.to_excel(output_path + '/NegAssociationKmeans_result_nostd_Sim.xlsx')
    cluster_centers_df.to_excel(output_path + '/NegAssociationKmeans_centers_nostd_Sim.xlsx')
    closest_samples.to_excel(output_path + '/NegAssociationKmeans_closestCenter_nostd_Sim.xlsx')
    df_site.to_excel(output_path + '/NegAssociationKmeans_site_nostd_Sim.xlsx')

def getDataSet():
    # 正负样本，这里换成我们的负样本。以,分割且病毒在前的关联对形式。
    df_pos = pd.read_excel('../Data/PosAssociation.xlsx')
    df_neg = pd.read_excel('../KmeaNegativeSample/NegAssociationKmeans_closestCenter_nostd_Sim.xlsx')
    df_pos['Label'] = 1
    df_neg['Label'] = 0
    PosAssociation = df_pos['Unnamed: 0'].tolist()
    NegAssociation = df_neg['Association'].tolist()
    # 获取数据集的关联名称
    Associations_train = PosAssociation + NegAssociation
    print("训练样本")
    print(len(Associations_train))
    print(Associations_train)
    #获取数据集的标签
    Labels = df_pos['Label'].tolist()+df_neg['Label'].tolist()
    # df_train = pd.DataFrame({
    #     "Association": Associations_train,
    #     "Label": Labels
    # })
    # df_train.to_csv('./DataSet.csv',index=False)

    #获得所有特征
    list_embedding_Sim,list_embedding_GO = [],[]
    for association in Associations_train:
        virus, disease = int(association.split(',')[0]), association.split(',')[1]
        embedding_Sim = extractEmbbedingOfSim(virus,disease,df_ds,df_vs).reshape(-1, 484)
        embedding_GO = extractEmbbedingOfGO(virus,disease,df_ds_GO,df_vs_GO,df_vds_GO).reshape(-1, 484*2)
        list_embedding_Sim.append(embedding_Sim)
        list_embedding_GO.append(embedding_GO)


    list_embedding_Sim_train = np.array(list_embedding_Sim).reshape(-1, 484)
    list_embedding_GO_train = np.array(list_embedding_GO).reshape(-1, 484*2)
    Associations_train = np.array(Associations_train)
    Labels = np.array(Labels)

    #所有关联的特征  ----待预测的关联对 这里修改成我们新增的独立测试集
    df_new_association = pd.read_csv("../newData/最终筛选出来的关联对2.csv",sep="\t")
    df_new_association_negetive = pd.read_excel("./KmeaNegativeSample/NegAssociationKmeans_closestCenter_nostd_Sim.xlsx")
    # 先统计有多少行，去除行
    df_new_association = df_new_association[~df_new_association['种id'].isin([10566, 290028, 1891763, 1891767, 12475, 11292])]
    # 这个集合很重要，里面是待预测的关联对，病毒在前、疾病在后
    Associations_test = df_new_association_negetive["Association"].tolist()
    Label_Associations_test = ["No"]*len(Associations_test)
    for i, row in df_new_association.iterrows():
        Associations_test.append(str(row['种id'])+","+row['mesh_id'])
        Label_Associations_test.append("Yes")
    print("待预测的关联对：")
    print(len(Associations_test))
    print(Associations_test)
    # df_Associations_test = pd.DataFrame({"Association": Associations_test, "AssociationLabel": Label_Associations_test})
    # # Association拆分成VirusTaxID和DiseaseMeshID
    # df_Associations_test["VirusTaxID"], df_Associations_test["DiseaseMeshID"] = (
    #     df_Associations_test["Association"].str.split(',').str[0]
    #     , df_Associations_test["Association"].str.split(',').str[1])
    # df_Associations_test.drop("Association", axis=1, inplace=True)
    # print(df_Associations_test)
    # df_Associations_test.to_excel('./DataSet.xlsx', index=False)
    list_embedding_Sim, list_embedding_GO = [], []
    for association in Associations_test:
        virus, disease = int(association.split(',')[0]), association.split(',')[1]
        embedding_Sim = extractEmbbedingOfSim(virus,disease,df_ds,df_vs).reshape(-1, 484)
        embedding_GO = extractEmbbedingOfGO(virus,disease,df_ds_GO,df_vs_GO,df_vds_GO).reshape(-1, 484*2)
        list_embedding_Sim.append(embedding_Sim)
        list_embedding_GO.append(embedding_GO)
    list_embedding_Sim_test = np.array(list_embedding_Sim).reshape(-1, 484)
    list_embedding_GO_test = np.array(list_embedding_GO).reshape(-1, 484*2)
    Associations_test = np.array(Associations_test)

    return Associations_train,Labels,list_embedding_Sim_train,list_embedding_GO_train,Associations_test,list_embedding_Sim_test,list_embedding_GO_test


if __name__ == '__main__':
    # NegSampleWihtKmeanCenter()  #Negative sample clustering - Kmean
    getDataSet()
    # Associations_train,Labels,list_embedding_Sim_train,list_embedding_GO_train,Associations_test,list_embedding_Sim_test,list_embedding_GO_test = getDataSet()
    # print(Associations_train.shape)
    # print(Labels.shape)
    # print(list_embedding_Sim_train.shape)
    # print(list_embedding_GO_train.shape)
    #
    # print(Associations_test.shape)
    # print(list_embedding_Sim_test.shape)
    # print(list_embedding_GO_test.shape)
