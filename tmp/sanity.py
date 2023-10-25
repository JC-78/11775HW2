import pandas as pd
import pickle
import os
import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier

# csv_file_path="data/labels/train_val.csv"

# data=[]
# labels=[]
# # train_val_df = pd.read_csv("data/labels/train_val.csv")

# for index,row in train_val_df.iterrows():
#     filename=row[0]
#     label=row[1]
#     pkl=os.path.join('data/cnn/',filename+".pkl")
pkl="LTE0MDc3MzUzOTE3NTQyMDYzOTA=.pkl"
with open(pkl,'rb') as file:
    pkl_data=pickle.load(file)
# print(len(pkl_data[0]))
print(pkl_data[0])
print(len(pkl_data[1]))

print("shape",pkl_data.shape)
