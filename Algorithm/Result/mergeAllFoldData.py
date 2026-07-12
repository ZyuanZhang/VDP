import pandas as pd
import os

inputpath = "./XGB-embedding_GO/pr_curves"
outputpath = "./XGB-embedding_GO"
filelist = os.listdir(inputpath)
df_ = pd.DataFrame()
for file in filelist:
    filepath = os.path.join(inputpath, file)
    fold = int(file.split("_")[1]) - 1
    df = pd.read_csv(filepath)
    # 为列名加上fold
    df.columns = [col[0].upper()+ col[1:] + str(fold) for col in df.columns]
    # df.columns = [col.upper()+ str(fold) for col in df.columns]
    df_ = pd.concat([df_, df], axis=1)
df_.to_csv(outputpath+"/pr_data.csv", sep='\t')