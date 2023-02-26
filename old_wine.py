import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

# Any results you write to the current directory are saved as output.

# df = pd.read_csv('wine.csv')
# print(type(df))
file = open('wine.csv', 'r')
print(file)