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
    pkl=os.path.join('data/cnn3d/',filename+".pkl")
    with open(pkl,'rb') as file:
        pkl_data=pickle.load(file)
    print(f"Filename: {filename}, Shape: {pkl_data[1].flatten().shape}")
    item=pkl_data[1].flatten()
    # print("item shape",item.shape)
    # reshaped_item=np.resize(item,2048)
    # print("reshaped item shape",reshaped_item.shape)
    # data.append(reshaped_item)
    #version 2
    target_length = 2048
    
    if len(item) < target_length:
        # Pad shorter arrays with zeros
        padded_item = np.pad(item, (0, target_length - len(item)))
        data.append(padded_item)
    elif len(item) > target_length:
        # Truncate longer arrays
        truncated_item = item[:target_length]
        data.append(truncated_item)
    else:
        # Keep arrays of the target length
        data.append(item)
    labels.append(label)
print("Shape of data:", np.array(data).shape)
print("Shape of labels:", np.array(labels).shape)
print("dataset built")
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(data, labels)
print("training done")

test_data=[]
test_df=pd.read_csv("data/labels/test_for_students.csv",header=0)
for index,row in test_df.iterrows():
    filename=row[0]
    pkl=os.path.join('data/cnn3d/',filename+".pkl")
    with open(pkl,'rb') as file:
        pkl_data=pickle.load(file)
    item=pkl_data[1].flatten()
    # reshaped_item=np.resize(item,2048)
    # test_data.append(reshaped_item)
    target_length = 2048
    
    if len(item) < target_length:
        # Pad shorter arrays with zeros
        padded_item = np.pad(item, (0, target_length - len(item)))
        test_data.append(padded_item)
    elif len(item) > target_length:
        # Truncate longer arrays
        truncated_item = item[:target_length]
        test_data.append(truncated_item)
    else:
        # Keep arrays of the target length
        test_data.append(item)
print("Shape of test data:", np.array(test_data).shape)
print("predicting")
pred = rf_classifier.predict(test_data)

pred=np.array(pred)
result_df=pd.DataFrame({
    'Id':test_df['Id'],
    'category':pred
})

result_df.to_csv('3d_pred.csv',index=False)