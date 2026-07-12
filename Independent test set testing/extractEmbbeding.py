import pandas as pd
import numpy as np


def extractEmbbedingOfSim(virus, disease, df_ds, df_vs):
    vs_v = np.array(df_vs.loc[virus]).reshape(1, -1)
    # print("vs_v.shape")
    # print(vs_v.shape)
    ds_d = np.array(df_ds.loc[disease]).reshape(1, -1)
    # print("ds_d.shape")
    # print(ds_d.shape)

    vv = vs_v
    dv = ds_d

    return np.hstack((vv, dv))


def extractEmbbedingOfGO(virus,disease,df_ds,df_vs,df_vds):
    vs_v = np.array(df_vs.loc[virus]).reshape(1,-1)
    ds_d = np.array(df_ds.loc[disease]).reshape(1,-1)
    GO_v = np.array(df_vds.loc[virus]).reshape(1,-1)
    GO_d = np.array(df_vds.T.loc[disease]).reshape(1,-1)

    vv = np.hstack((vs_v,GO_v))
    dv = np.hstack((ds_d,GO_d))

    return np.hstack((vv,dv))


if __name__ == '__main__':
    # 这里修改为我们重新计算的特征，不用去掉四个病毒的相似性特征，因为我拿到新病毒基因组这些都是可以计算出来的。
    df_ds = pd.read_csv('../2Features/GenomeAndMeshSimilarty/disease_SemanticSimilarity_matrix.csv',sep='\t',index_col=0)
    df_vs = pd.read_csv('../2Features/GenomeAndMeshSimilarty/virus_GenomeSimilarity_matrix.csv',sep='\t',index_col=0)

    df_ds_GO = pd.read_csv('../2Features/GO/disease_FunctionSimilarity_matrix_Sem.csv', sep='\t', index_col=0)
    df_vs_GO = pd.read_csv('../2Features/GO/virus_FunctionSimilarity_matrix_Sem.csv', sep='\t', index_col=0)
    df_vds_GO = pd.read_csv('../2Features/GO/cross_FunctionSimilarity_matrix_Sem.csv', sep='\t', index_col=0)

    #test
    print(df_ds)
    print(df_vs)
    virus, disease = 138948, "D009461"
    v_Sim = extractEmbbedingOfSim(virus, disease,df_ds,df_vs)
    v_GO = extractEmbbedingOfGO(virus, disease,df_ds_GO,df_vs_GO,df_vds_GO)
    print(v_Sim.shape)
    print(v_GO.shape)
