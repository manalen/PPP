import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.model_selection import KFold
from matplotlib import style
style.use('fivethirtyeight')
%matplotlib inline
%pylab inline

def draw_heatmap(data):
    sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')

def convert_cast(df,categorical_cols):
   for col_name in categorical_cols:
       df[col_name]=df[col_name].astype("object")
   return df

def input_missing_values(df):
    for col in df.columns:
        if (df[col].dtype is float) or (df[col].dtype is int):
            df[col]=df[col].fillna(df[col].median())
        if (df[col].dtype == object):
            df[col]=df[col].fillna(df[col].mode()[0].split(" ")[0])
    return df

def parse_model(X, use_columns):
    if "Survived" not in X.columns :
        raise ValueError("target column survived should belong to df")
    target=X["Survived"]
    X=X[use_columns]
    return X, target

def plot_hist(feature, bins=20):
    x1 = np.array(dead[feature].dropna())
    x2 = np.array(survived[feature].dropna())
    plt.hist([x1, x2], label=["Victime", "Survivant"], bins=bins, color=['r', 'b'])
    plt.legend(loc="upper left")
    plt.title('Distribution relative de %s' %feature)
    plt.show()
    
def My_model (X,y,size,RdomState=42):
    #X,y
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=size,random_state=RdomState)
    model=LogisticRegression(random_state=RdomState)
    model.fit(X_train,y_train)
    #run the model
    y_pred=model.predict(X_test)
    y_prob=model.predict_proba(X_test)[:,1]
    score_train=model.score(X_train,y_train)
    score_test=model.score(X_test,y_test)
    
    return {"y_test":y_test,"prediction":y_pred,"proba":y_prob,"score_train":score_train,"score_test":score_test,"model":model}

