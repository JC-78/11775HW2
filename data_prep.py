import pandas as pd
import pickle
import os
import numpy as np


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
    csv=os.path.join('11775-HW1/snf/',filename+".csv")
    pkl=os.path.join('data/cnn3d/',filename+".pkl")
    if not os.path.exists(csv):  # Check if CSV file exists
        print(f"CSV file {csv} does not exist. Skipping...")
        continue
    with open(pkl,'rb') as file:
        pkl_data=pickle.load(file)
    csv_data = np.genfromtxt(csv, delimiter=',').reshape(-1, 1) 
    array1_np = np.array(pkl_data[0])
    array2_np = np.array(pkl_data[1])
    # meow = array2_np.reshape(array2_np.shape[0]*array2_np.shape[2]*array2_np.shape[3],1)
    meow = array2_np.flatten()
    meow=meow.reshape(-1,1)
    combined_data = np.concatenate((csv_data, meow), axis=0)
    test_data.append(combined_data)

   

print("Shape of test data:", np.array(test_data).shape)
print("predicting")
pred = rf_classifier.predict(test_data)

pred=np.array(pred)
result_df=pd.DataFrame({
    'Id':test_df['Id'],
    'category':pred
})

result_df.to_csv('3d_pred.csv',index=False)