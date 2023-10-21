import pandas as pd
import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# csv_file_path="data/labels/train_val.csv"

data=[]
labels=[]
train_val_df = pd.read_csv("data/labels/train_val.csv")

for index,row in train_val_df.iterrows():
    filename=row[0]
    label=row[1]
    pkl=os.path.join('data/cnn/',filename+".pkl")
    with open(pkl,'rb') as file:
        pkl_data=pickle.load(file)
    data.append(pkl_data[1].flatten())
    labels.append(label)
print("dataset built")
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(data, labels)
print("training done")

test_data=[]
test_df=pd.read_csv("data/labels/test_for_students.csv",header=0)
for index,row in test_df.iterrows():
    filename=row[0]
    pkl=os.path.join('data/cnn/',filename+".pkl")
    with open(pkl,'rb') as file:
        pkl_data=pickle.load(file)
    test_data.append(pkl_data[1].flatten())
    
print("predicting")
pred = rf_classifier.predict(test_data)

pred=np.array(pred)
result_df=pd.DataFrame({
    'Id':test_df['Id'],
    'category':pred
})

result.to_csv('model_pred.csv',index=False)
# # List all files in the current directory
# pickle_files = [f for f in os.listdir() if f.endswith(".pkl")]

# # Read pickle files and concatenate them into a single dataframe
# arrays = []
# for file_name in pickle_files:
#     with open(file_name, 'rb') as file:
#         df = pickle.load(file)
#         print(df[1].shape)
#         arrays.append(df[1].flatten())

# combined_array = np.concatenate(arrays)

# combined_df = pd.DataFrame(combined_array)

# # Read train_val.csv

# # Print first 5 rows of combined dataframe
# print("First 5 rows of combined dataframe:")
# print(combined_df.head())

# # Print number of rows and first 5 rows of train_val.csv dataframe
# print("\nNumber of rows in train_val.csv:", len(train_val_df))
# print("\nFirst 5 rows of train_val.csv:")
# print(train_val_df.head())
