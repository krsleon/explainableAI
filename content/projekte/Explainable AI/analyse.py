import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
#import lime.lime_tabular
import sys
from pathlib import Path
import plotly.graph_objects as go

ORDNER = Path(__file__).parent
if str(ORDNER) not in sys.path:
    sys.path.insert(0, str(ORDNER))

data = pd.read_csv(ORDNER/'penguins.csv')
data = data.dropna() # removes rows with NaN values. 333 datapoints remain.

