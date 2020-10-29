import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
%matplotlib inline


def draw_heatmap(data):
    sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
