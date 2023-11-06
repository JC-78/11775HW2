import pandas as pd
import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

data=[]
labels=[]
train_val_df = pd.read_csv("11775-HW2/data/labels/train_val.csv")

for index,row in train_val_df.iterrows():
    filename=row[0]
    label=row[1]
    pkl=os.path.join('11775-HW2/data/cnn3d/',filename+".pkl")
    csv=os.path.join('11775-HW1/snf/',filename+".csv")
    if not os.path.exists(csv):  # Check if CSV file exists
        print(f"CSV file {csv} does not exist. Skipping...")
        continue
    with open(pkl,'rb') as file:
        pkl_data=pickle.load(file)
    csv_data = np.genfromtxt(csv, delimiter=',').reshape(-1, 1) 
    array1_np = np.array(pkl_data[0])
    array2_np = np.array(pkl_data[1])
    # print("array1 (pkl_data[0]) np shape is ",array1_np.shape)
    # print("array2 (pkl_data[1]) np shape is",array2_np.shape)

    # meow = array2_np.reshape(array2_np.shape[0]*array2_np.shape[2]*array2_np.shape[3],1)
    meow=array2_np.flatten()
    # print("csv_data shape is",csv_data.shape)
    # print("3d video data is ",meow.shape)
    meow=meow.reshape(-1,1)
    combined_data = np.concatenate((csv_data, meow), axis=0)
    data.append(combined_data)
    labels.append(label)
data=np.array(data)
labels=np.array(labels)
print("Shape of data:", data.shape)
print("Shape of labels:", labels.shape)
print("dataset built")
data1=data.squeeze()
labels1=labels.reshape(-1,1)
print("Shape of squeezed data: ",data1.shape)
print("Shape of labels after reshaping:", labels1.shape)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
xgb_classifier = XGBClassifier(n_estimators=100, random_state=24)

xgb_classifier.fit(data1, labels1)
rf_classifier.fit(data1, labels1)

print("training done")


print("predicting")

logits_1 = rf_classifier.predict_proba(data1)
logits_2 = xgb_classifier.predict_proba(data1)

print("Shape of logits_1:", logits_1.shape)
print("Shape of logits_2:", logits_2.shape) #(749,15)
# combined_logits = np.concatenate((logits_csv, logits_pkl), axis=1) #(749,30)
# combined_logits = np.concatenate((logits_csv, logits_pkl), axis=0)#(1498, 15)
combined_logits=logits_1+logits_2
pred = np.argmax(combined_logits, axis=1)

pred=np.array(pred)
confusion_matrix(labels1, pred)
print("Confusion Matrix:")
print(confusion_rf)
