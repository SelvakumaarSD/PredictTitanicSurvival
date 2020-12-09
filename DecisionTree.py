#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np  #This module will help us to give numbers , or array
import pandas as pd #To read CSV files
import missingno as msno #To display dataset columns with null values
import re

#for visualization
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

#for ML model creation and prediction
from sklearn import preprocessing
# Import Decision Tree Classifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import confusion_matrix, accuracy_score #to measure the accuracy of predicted set and plot confusion matrix

import warnings
warnings.filterwarnings('ignore')


# In[2]:


train = pd.read_csv('dataset for ID3 decision tree/Titanic_training.csv')
msno.matrix(train)


# In[3]:


train.describe()


# In[4]:


train.head(5)


# In[5]:


test = pd.read_csv('dataset for ID3 decision tree/Titanic_test.csv')
msno.matrix(test)


# In[6]:


test.describe()


# In[7]:


test.head(5)


# In[8]:


combine_dataset = [train, test]

for source in combine_dataset:
    source['Cabin'] = source["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
    source['Title'] = source['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    source.loc[source['Title'].isin(['Capt', 'Rev', 'Major', 'Col', 'Sir', 'Jonkheer', 'Don', 'Dr']), ['Title']] = 'Mr'
    source.loc[source['Title'].isin(['Mlle', 'Ms']), ['Title']] = 'Miss'
    source.loc[source['Title'].isin(['Mme','Lady','Dona','Countess','the Countess']), ['Title']] = 'Mrs'
    
    source['Embarked'] = source['Embarked'].fillna('S')
    source['Fare'] = source['Fare'].fillna(train['Fare'].median())
    
    age_avg = source['Age'].mean()
    age_std = source['Age'].std()
    age_null_count = source['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    # Next line has been improved to avoid warning
    source.loc[np.isnan(source['Age']), 'Age'] = age_null_random_list
    source['Age'] = source['Age'].astype(int)


# In[9]:


drop_train_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived']
X = train.drop(drop_train_elements, axis = 1)
Y = train.Survived

X_onehot = pd.get_dummies(X, drop_first =False)
X_onehot_names = list(X_onehot)
print("Feature names after one hot encode: " , X_onehot_names)
print(X_onehot.head(3))

le = preprocessing.LabelEncoder()
y_coded = le.fit_transform(np.ravel(Y))


# In[10]:


colormap = plt.cm.inferno
mask = np.triu(np.ones_like(X_onehot.astype(float).corr(), dtype=bool))
plt.figure(figsize=(12,12))
plt.title('Correlation Matrix of Features in Titanic Training Dataset', y=1.05, size=15)
sns.heatmap(X_onehot.astype(float).corr(),mask=mask,linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True, cbar_kws={"shrink": .5})


# In[11]:


depth = range(1, len(X_onehot_names))
X_train, X_valid, y_train, y_valid = train_test_split(X_onehot,y_coded,test_size=0.2, train_size=0.5, random_state=30, stratify=y_coded)
acc_score = []

for d in depth:
    clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=d)
    clf = clf.fit(X_train, y_train)
    y_predict = clf.predict(X_valid)
    acc_score.append(metrics.accuracy_score(y_true=y_valid, y_pred=y_predict))


# In[12]:


df = pd.DataFrame({"Max Depth": depth, "Average Accuracy": acc_score})
df = df[["Max Depth", "Average Accuracy"]]
print(df.to_string(index=False))


# In[13]:


model = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)
model = model.fit(X_onehot,y_coded)
    
fig, ax = plt.subplots(figsize=(20, 7))
tree.plot_tree (model, feature_names=X_onehot_names, class_names=['S','D'], filled=True, rounded=True, proportion=True, fontsize=10)
plt.show()


# In[14]:


drop_test_features = ['PassengerId', 'Name', 'Ticket', 'Cabin']
X_test = test.drop(drop_test_features, axis = 1)
X_test_onehot = pd.get_dummies (X_test, drop_first =False)

test['Survived'] = model.predict(X_test_onehot)
print(test.head(3))


# In[15]:


result = pd.DataFrame({"PassengerId": test['PassengerId'], "Survived": test['Survived']})
result.to_csv('dataset for ID3 decision tree/Titanic_Dataset_Accuracy_Test/predictions.csv', index=False)

