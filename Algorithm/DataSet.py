import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from extractEmbbeding import extractEmbbedingOfGO,extractEmbbedingOfSim


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
    df_pos = pd.read_excel('../Data/PosAssociation.xlsx')
    df_vda = pd.read_csv('../Data/VDA_interaction.csv', sep='\t', index_col=0)
    list_virus,list_disease = df_vda.index.tolist(),df_vda.columns.tolist()
    list_all = []
    for virus in list_virus:
        for disease in list_disease:
            list_all.append(str(virus)+','+disease)

    PosAssociation = df_pos['Unnamed: 0'].tolist()
    NegAssociation = list(set(list_all) - set(PosAssociation))
    # print(len(PosAssociation)) #1286
    # print(len(NegAssociation)) #24154

    result_df, cluster_centers_df, closest_samples, df_site = get_Kmeans_result(NegAssociation, k=len(PosAssociation))
    output_path = '../KmeaNegativeSample'
    result_df.to_excel(output_path + '/NegAssociationKmeans_result_nostd_Sim.xlsx')
    cluster_centers_df.to_excel(output_path + '/NegAssociationKmeans_centers_nostd_Sim.xlsx')
    closest_samples.to_excel(output_path + '/NegAssociationKmeans_closestCenter_nostd_Sim.xlsx')
    df_site.to_excel(output_path + '/NegAssociationKmeans_site_nostd_Sim.xlsx')

def getDataSet():
    df_pos = pd.read_excel('../Data/PosAssociation.xlsx')
    df_neg = pd.read_excel('../KmeaNegativeSample/NegAssociationKmeans_closestCenter_nostd_Sim.xlsx')
    df_pos['Label'] = 1
    df_neg['Label'] = 0
    PosAssociation = df_pos['Unnamed: 0'].tolist()
    NegAssociation = df_neg['Association'].tolist()
    # 获取数据集的关联名称
    Associations = PosAssociation + NegAssociation
    #获取数据集的标签
    Labels = df_pos['Label'].tolist()+df_neg['Label'].tolist()

    #获得所有特征
    list_embedding_Sim,list_embedding_GO = [],[]
    for association in Associations:
        virus, disease = int(association.split(',')[0]), association.split(',')[1]
        embedding_Sim = extractEmbbedingOfSim(virus,disease,df_ds,df_vs).reshape(-1, 484)
        embedding_GO = extractEmbbedingOfGO(virus,disease,df_ds_GO,df_vs_GO,df_vds_GO).reshape(-1, 484*2)
        list_embedding_Sim.append(embedding_Sim)
        list_embedding_GO.append(embedding_GO)


    list_embedding_Sim = np.array(list_embedding_Sim).reshape(-1, 484)
    list_embedding_GO = np.array(list_embedding_GO).reshape(-1, 484*2)
    Associations = np.array(Associations)
    Labels = np.array(Labels)
    return Associations,Labels,list_embedding_Sim,list_embedding_GO


if __name__ == '__main__':
    # NegSampleWihtKmeanCenter()  #Negative sample clustering - Kmean

    Associations,Labels,list_embedding_Sim,list_embedding_GO = getDataSet()
    print(Associations.shape)
    print(Labels.shape)
    print(list_embedding_Sim.shape)
    print(list_embedding_GO.shape)
