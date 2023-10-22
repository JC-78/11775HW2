import pandas as pd
import pickle
import os
import numpy as np
pkl='/Users/joonghochoi/Desktop/11775hw2/LTQyMjkzODk1NDU3MjkzOTUwMTA=.pkl'
with open(pkl,'rb') as file:
        pkl_data=pickle.load(file)
print(len(pkl_data))
print(pkl_data[0])
print(len(pkl_data[1]))

print("yesl")