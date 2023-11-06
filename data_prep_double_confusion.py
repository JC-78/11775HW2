import pandas as pd
import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))

plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for RandomForest')
plt.savefig('confusion_matrix.png')  # Save the figure

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
train_logits_rf = rf_classifier.predict(data1)
train_logits_xgb = xgb_classifier.predict(data1)

# Calculate confusion matrix for RandomForest
confusion_rf = confusion_matrix(labels1, train_logits_rf)

# Calculate confusion matrix for XGBoost
confusion_xgb = confusion_matrix(labels1, train_logits_xgb)

# Print confusion matrices
print("Confusion Matrix for RandomForest:")
print(confusion_rf)

print("\nConfusion Matrix for XGBoost:")
print(confusion_xgb)

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_rf, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for RandomForest')
plt.savefig('confusion_matrix_rf.png')  # Save the figure
plt.show()

# Plot confusion matrix for XGBoost
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_xgb, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for XGBoost')
plt.savefig('confusion_matrix_xgb.png')  # Save the figure
plt.show()