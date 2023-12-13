#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import scikitplot as skplt

#dataset loading
df = pd.read_csv('/Users/farhatlamiabarsha/Downloads/archive/Fraud_Data.csv')
print ('Not Fraud % ',round(df['class'].value_counts()[0]/len(df)*100,2))
print ('Fraud %    ',round(df['class'].value_counts()[1]/len(df)*100,2))

fraud_counts = df['class'].value_counts()
real_transactions = fraud_counts[0]
fraud_transactions = fraud_counts[1]

print("Real Transactions:", real_transactions)
print("Fraud Transactions:", fraud_transactions)

total_count = len(df)
print("Total number of data:", total_count)

df.head()


# In[15]:


columns_to_drop = ['user_id','signup_time','device_id', 'source', 'browser', 'sex', 'age', 'ip_address']
df.drop(columns=columns_to_drop, axis=1, inplace=True)


# In[16]:


remaining_columns = df.columns
print(remaining_columns)


# In[17]:


df['purchase_time'] = pd.to_datetime(df['purchase_time'])
df['year'] = df['purchase_time'].dt.year
df['month'] = df['purchase_time'].dt.month
df['day'] = df['purchase_time'].dt.day
df['hour'] = df['purchase_time'].dt.hour
df['minute'] = df['purchase_time'].dt.minute
df['second'] = df['purchase_time'].dt.second
df = df.drop('purchase_time', axis=1)


# In[18]:


new_column_order = ['year', 'month', 'day', 'hour', 'minute', 'second', 'purchase_value', 'class']
new_df = df[new_column_order]
df = new_df
df.describe().round()


# In[19]:


feature_names = df.iloc[:, 0:7].columns
data_features = df[feature_names]
target = df.iloc[:1, 7:].columns
data_target = df[target]


# In[20]:


# dividing the dataframe into fraud and non fraud data
non_fraud=df[df['class']==0]
fraud=df[df['class']==1]


# In[21]:


non_fraud.shape, fraud.shape


# In[22]:


# select the 492 non-fraud entries from the dataframe 
non_fraud=non_fraud.sample(fraud.shape[0])
non_fraud.shape


# In[23]:


df = pd.concat([fraud, non_fraud], ignore_index=True)


# In[24]:


df


# In[33]:


df['class'].value_counts()


# In[46]:


# dividing the dataframe into dependent and independent varaible
X=df.drop(['class'], axis=1)
y=df['class']

# check the shape
X.shape, y.shape


# In[38]:


# divide the dataset into training and testing dataset
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=99)

# check the shape again
X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[39]:


lr = LogisticRegression()
lr.fit(X_train, y_train)


# In[40]:


def PrintStats(cmat, y_test, pred):
    tneg = cmat[0][0]
    fpos = cmat[0][1]
    fneg = cmat[1][0]
    tpos = cmat[1][1]
def RunModel(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train.values.ravel())
    pred = model.predict(X_test)
    matrix = confusion_matrix(y_test, pred)
    return matrix, pred


# In[48]:


# logistic regression model
y_prob = lr.predict_proba(X_test)[:, 1]
y_pred = (y_prob > 0.3).astype(int)  

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# In[52]:


conf_matrix = confusion_matrix(y_test, y_pred)

# Plotting Confusion Matrix
plt.figure(figsize=(4, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

Accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", Accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# In[50]:

# ROC curve values
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Plot ROC curve
plt.figure(figsize=(3, 3))
plt.plot(fpr, tpr, label='Logistic Regression')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend()
plt.show()






