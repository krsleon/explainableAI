import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import lime.lime_tabular

data = pd.read_csv('penguins.csv')
data = data.dropna() # removes rows with NaN values. 333 datapoints remain.

