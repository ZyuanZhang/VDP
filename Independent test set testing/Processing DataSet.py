import pandas as pd

df = pd.read_excel("./DataSet.xlsx")
df_myvirusdatabase = pd.read_excel(r"D:\AAA硕士\感染人数据库\final_data/all_known_infecting_human_virus_from_fengyang_with_non_mammalian_and_reversed.xlsx").rename(
    columns={"Taxid":"VirusTaxID","Genome":"RefGenome","Virus family":"VirusFamilyName","Genome type":"VirusGenomeComposition"})
print(df_myvirusdatabase)
df = pd.merge(df, df_myvirusdatabase[["VirusTaxID","RefGenome","VirusFamilyName","VirusGenomeComposition"]], on="VirusTaxID", how="left")
df = df[["VirusTaxID","RefGenome","VirusFamilyName","VirusGenomeComposition","DiseaseMeshID","AssociationLabel"]]
# 并且在AssociationLabel列前插入一列MajorDiseaseCategory
insert_pos = df.columns.get_loc("AssociationLabel")
df.insert(insert_pos, "MajorDiseaseCategory", "")
print(df.columns)
print(df)
df.to_excel("Associations_test.xlsx", index=False)