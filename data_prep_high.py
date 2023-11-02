import pandas as pd
import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# csv="Mzg4MTU4MTQ1OTQ4MzcwOTcxMA==.csv"
# pkl="LTQyMjkzODk1NDU3MjkzOTUwMTA=.pkl"
# print("csv id extraction ",csv[:-3])
# with open(pkl,'rb') as file:
#     pkl_data=pickle.load(file)
# csv_data = np.genfromtxt(csv, delimiter=',').reshape(-1, 1) 
# print(csv_data.shape)
# array1_np = np.array(pkl_data[0])
# array2_np = np.array(pkl_data[1])
# #array2 (pkl_data[1]) np shape is (1, 768, 1, 1, 1)
# #(512, 1, 5, 6)
# print("array1 (pkl_data[0]) np shape is ",array1_np.shape)
# print("array2 (pkl_data[1]) np shape is",array2_np.shape)
# print("array2 (pkl_data[1]) flattened np shape is",array2_np.flatten().shape)

# print("printing content")
# # meow = array2_np.reshape(array2_np.shape[0]*array2_np.shape[2]*array2_np.shape[3],1)
# meow=array2_np.flatten()
# meow=meow.reshape(-1,1)
# print( "meow.shape",meow.shape) #same as flattening. But cannot always  use method above
# # cuz ValueError: cannot reshape array of size 768 into shape (1,1)
# combined_data = np.concatenate((csv_data, meow), axis=0)
# print(combined_data.shape)


train_data_pkl=[]
train_data_csv = []
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
    array2_np = np.array(pkl_data[1])
    meow=array2_np.flatten()
    meow=meow.reshape(-1,1)
    train_data_csv.append(csv_data)
    train_data_pkl.append(meow)
    labels.append(label)
    
labels=np.array(labels)
train_data_csv=np.array(train_data_csv)
train_data_pkl=np.array(train_data_pkl)

print("Shape of train_data_csv:", train_data_csv.shape)
print("Shape of train_data_pkl:", train_data_pkl.shape)
print("Shape of labels:", labels.shape)
print("dataset built")
labels1=labels.reshape(-1,1)
train_data_csv=train_data_csv.squeeze()
train_data_pkl=train_data_pkl.squeeze()
print("Shape of squeezed pkl data: ",train_data_pkl.shape)
print("Shape of squeezed csv data: ",train_data_csv.shape)

print("Shape of labels after reshaping:", labels1.shape)

rf_classifier_csv = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier_pkl = RandomForestClassifier(n_estimators=100, random_state=42)

rf_classifier_csv.fit(train_data_csv, labels1)
rf_classifier_pkl.fit(train_data_pkl, labels1)
print("training done")

test_data_pkl=[]
test_data_csv = []
test_df=pd.read_csv("11775-HW2/data/labels/test_for_students.csv",header=0)
for index,row in test_df.iterrows():
    filename=row[0]
    csv=os.path.join('11775-HW1/snf/',filename+".csv")
    pkl=os.path.join('11775-HW2/data/cnn3d/',filename+".pkl")
    if not os.path.exists(csv):  # Check if CSV file exists
        print(f"CSV file {csv} does not exist. Skipping...")
        continue
    with open(pkl,'rb') as file:
        pkl_data = pickle.load(file)
    csv_data = np.genfromtxt(csv, delimiter=',').reshape(-1, 1) 
    test_data_csv.append(csv_data)
    array2_np = np.array(pkl_data[1])
    meow_pkl = array2_np.flatten()
    meow_pkl = meow_pkl.reshape(-1,1)
    test_data_pkl.append(meow_pkl)

test_data_csv=np.array(test_data_csv)
test_data_pkl=np.array(test_data_pkl)
print("Shape of test data:", test_data_pkl.shape)

print("predicting")

test_data_csv = test_data_csv.squeeze()
test_data_pkl =test_data_pkl.squeeze()
print("Shape of squeezed test pkl data:", test_data_pkl.shape)
print("Shape of squeezed test csv data:", test_data_csv.shape)


# Get logits from both models
logits_csv = rf_classifier_csv.predict_proba(test_data_csv)
logits_pkl = rf_classifier_pkl.predict_proba(test_data_pkl)

# Combine logits (for example, by concatenating them)
combined_logits = np.concatenate((logits_csv, logits_pkl), axis=1)

# Make predictions based on the combined logits
pred = np.argmax(combined_logits, axis=1)
print("pred shape is ",pred.shape)
pred=np.array(pred)
result_df=pd.DataFrame({
    'Id':test_df['Id'],
    'category':pred
})

result_df.to_csv('high_fusion_pred.csv',index=False)