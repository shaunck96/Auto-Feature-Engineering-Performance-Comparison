#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
#from autoviz.AutoViz_Class import AutoViz_Class


# In[2]:


#pip install PyImpetus


# In[3]:


#pip install featurewiz


# In[4]:


import math
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.metrics import log_loss, mean_squared_error
import scipy.stats as ss
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import contextlib
import joblib
from featurewiz import featurewiz


# In[5]:


from PyImpetus import PPIMBC
from PyImpetus import PPIMBR


# In[6]:


data = pd.read_csv(r'C:\Users\307164\Desktop\Additional Dataset\IonosphereData.csv')
data


# In[7]:


data.describe()


# Looks like mainly continuous features in the dataset

# In[8]:


data.nunique()


# In[9]:


pd.value_counts(data.a07)


# In[10]:


pd.value_counts(data.Target)


# Split into train and test ; Fit basic algorithms and see if imbalanced accuracy arises ; Check with algorithm fits best with the data due to imbalance

# In[11]:


data.corr()


# In[12]:


import seaborn as sns
corr = data.corr()
sns.heatmap(corr)


# In[13]:


import numpy as np

# Create correlation matrix
corr_matrix = data.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

# Drop features 
corr_features = data.drop(to_drop, axis=1)


# In[14]:


corr_matrix


# In[15]:


corr_features


# In[16]:


data['Target'].value_counts()


# In[17]:


#acacacacac
#data['Target']


# Checking feature correlation to get idea of useful features

# In[18]:


#data['Target'] = data['Target'].apply(lambda x: 1 if "g" else o if 'b' else 'NONE')
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
data["Target"] = lb_make.fit_transform(data["Target"])
data


# In[19]:


corr_matrix = data.corr()


# In[20]:


corr_matrix[corr_matrix['Target']>0.5]


# In[21]:


data.drop(columns=['a02'],inplace=True)


# Inference : a03, a05 are moderately correlated with target variable

# Lets check feature importance scores

# In[22]:


data


# In[23]:


data['a01'].value_counts()


# In[24]:


X = data.loc[:,data.columns!='Target']
y = data['Target']


# In[25]:


X


# In[26]:


y


# Feature Selection Techniques to be experimented with: 
#     (1) Autofeat
#     (2) FeatureSelector
#     (3) PyImpetus
#     (4) Eli5
#     (5) Feature-Engine
#     (6) Boruta/SHAP+Boruta
#     (7) FeatureViz
#     (8) analyticsvidhya.com/blog/2020/10/feature-selection-techniques-in-machine-learning/ - Check for automated technques
#     (9) 

# In[27]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42)


# Setting Baseline Models : 
#     (1) SVM Baseline
#     (2) LGB Baseline
#     (3) Cat Baseline
#     (4) Tabnet Baseline
#     (5) Tab Baseline
#     (6) AutoGluon
#     (7) Logistic Regression
#     (8) Decision Trees
#     (9) Random Forest
#     (10) Xgboost Classifier

# In[28]:


#Baseline Evaluation Metric
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


# Logistic Regression Baseline

# In[29]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train,y_train)
predictions = logreg.predict(X_test)
accuracy_score_lr_baseline = rmse(predictions,y_test)
accuracy_score_lr_baseline


# SVM Baseline

# In[30]:


from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

X_train_svm = X_train.to_numpy()
y_train_svm = y_train.to_numpy()
X_test_svm = X_test.to_numpy()
y_test_svm = y_test.to_numpy()

svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train_svm, y_train_svm)
predictions = svclassifier.predict(X_test_svm)
accuracy_score_svm_baseline = rmse(predictions,y_test_svm)
accuracy_score_svm_baseline


# Decision Tree Baseline

# In[31]:


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
predictions = dt.predict(X_test)
accuracy_score_dt_baseline = rmse(predictions,y_test)
accuracy_score_dt_baseline


# Random Forest Baseline

# In[32]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train,y_train)
predictions = rf.predict(X_test)
accuracy_score_rf_baseline = rmse(predictions,y_test)
accuracy_score_rf_baseline


# LGB Baseline

# In[33]:


import lightgbm as lgb

lgbclf = lgb.LGBMClassifier()
lgbclf.fit(X_train, y_train)
predictions = lgbclf.predict(X_test)
accuracy_score_lgb_baseline = rmse(predictions,y_test)
accuracy_score_lgb_baseline


# XGBoost Baseline

# In[34]:


import xgboost as xgb

xgbclf = xgb.XGBClassifier()
xgbclf.fit(X_train, y_train)
predictions = xgbclf.predict(X_test)
accuracy_score_xgb_baseline = rmse(predictions,y_test)
accuracy_score_xgb_baseline


# CATBoost Baseline

# #Try to Fix CATBOOST, PyTorch installations
# 
# # catboost as cat
# 
# #catclf = cat.XGBClassifier()
# #catclf.fit(X_train, y_train)
# #predictions = catclf.predict(X_test)
# #accuracy_score_cat_baseline = rmse(predictions,y_test)
# #accuracy_score_cat_baseline

# In[35]:


#Try to Fix CATBOOST, PyTorch installations

# catboost as cat

#catclf = cat.XGBClassifier()
#catclf.fit(X_train, y_train)
#predictions = catclf.predict(X_test)
#accuracy_score_cat_baseline = rmse(predictions,y_test)
#accuracy_score_cat_baseline


# In[36]:


get_ipython().system('pip install catboost')


# In[37]:


#pip install catboost --no-cache-dir


# Tabnet Baseline

# In[38]:


X_train_tabnet = X_train.to_numpy()
y_train_tabnet = y_train.to_numpy()
X_test_tabnet = X_test.to_numpy()
y_test_tabnet = y_test.to_numpy()


# In[39]:


import pytorch_tabnet
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

tabclf = TabNetClassifier(optimizer_fn=torch.optim.Adam,
                       optimizer_params=dict(lr=2e-2),
                       scheduler_params={"step_size":10, # how to use learning rate scheduler
                                         "gamma":0.9},
                       scheduler_fn=torch.optim.lr_scheduler.StepLR,
                       mask_type='entmax' # "sparsemax"
                      )

tabclf.fit(
    X_train_tabnet,y_train_tabnet,
    eval_set=[(X_train_tabnet, y_train_tabnet), (X_test_tabnet, y_test_tabnet)],
    eval_name=['train', 'valid'],
    eval_metric=['accuracy'],
    max_epochs=500 , patience=30,
    batch_size=128, virtual_batch_size=128,
    num_workers=0,
    weights=1,
    drop_last=False
) 


# In[40]:


y_pred = tabclf.predict(X_test_tabnet)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test_tabnet,y_pred))
print(classification_report(y_test,y_pred))


# OUTPUT of AutoML

# In[41]:


from flaml import AutoML
automl = AutoML()
automl.fit(X_train, y_train, task="classification")
y_pred = automl.predict(X_test)
accuracy_score_atuoml_baseline = rmse(y_pred,y_test)
accuracy_score_atuoml_baseline


# AutoGluon left to experiment

# Let's try some feature engineering techniques 

# AUTOFEAT

# In[42]:


#pip install autofeat


# AutoFeat is a python library that provides automated feature engineering and feature selection along with models such as AutoFeatRegressor and AutoFeatClassifier. These are built with many scientific calculations and need good computational power.
# 
# Properties
# Works similar to scikit learn models using functions such as fit(), fit_transform(), predict(), and score().
# Can handle categorical features with One hot encoding.
# Until now works only on Supervised Learning(Regression and Classification)
# Feature Selector class for selecting suitable features.
# Physical units of features can be passed and relatable features will be computed.
# Buckingham Pi theorem – used for computing dimensionless quantities.
# Only used for tabular data
# Advantages
# Simpler to understand
# Easy to use 
# Good for non-statisticians 
# Disadvantages
# Could miss out on some important features
# Features need to be scaled before using AutoFeat
# Cannot handle missing values 
# 
# Autofeat uses a wrapper method-based approach to perform multi-variate feature selection. The Lasso LARS linear regression model and the L1-regularized logistic regression model are used to choose features based on sparse weights. It uses a ‘noise filtering approach’ by training the model using the original features and the noise features and selecting only those features that have a model coefficient greater than the largest coefficient of the noise features. This approach works perfectly well when there is a sufficient number of features.
# But, if the dataset has a large number of correlated features and the total number of features is greater than the data samples, an initial set of features is obtained by training the L1-regularized model on all features and selecting features with the largest absolute coefficients. This set is then combined with the chunk of features obtained by equally splitting the remaining features from initial feature selection by an L1-regularized linear model. A model is fit on each such chunk to select additional features. All feature subsets are then combined and used to train another model based on which the final feature set is determined. Finally, the independent feature selection subsets are combined and highly correlated features are filtered out (keeping those features that were selected in the most runs). The remaining features are then again used to fit a model to select the ultimate feature set.

# In[43]:


from autofeat import AutoFeatRegressor

model = AutoFeatRegressor()
model


# In[44]:


X_train_autofeat = model.fit_transform(X_train.to_numpy(), y_train.to_numpy().flatten())
X_test_autofeat = model.transform(X_test.to_numpy())
#number of additional/reduced number of features


# In[45]:


X_train_autofeat.shape[1] - X_train.shape[1]


# Check Performance with AutoML

# In[46]:


from flaml import AutoML
automl = AutoML()
automl.fit(X_train_autofeat, y_train, task="classification")
y_pred = automl.predict(X_test_autofeat)
accuracy_score_atuoml_baseline = rmse(y_pred,y_test)
accuracy_score_atuoml_baseline


# In[ ]:





# (2) FEATURE SELECTOR ----> matplotlib version discrepency with python version (python version 2.7 needed for matplotlib 2.1.2
# 
# Steps involved in FeatureSelector: 
#     (1) Missing Values
#     (2) Single Unique Value
#     (3) Highly Correlated Features
#     (4) Zero Importance Features
#     (5) Low Importance Features
#     
#     https://github.com/WillKoehrsen/feature-selector/blob/master/Feature%20Selector%20Usage.ipynb

# In[47]:


#pip uninstall matplotlib


# In[49]:


#avsvsvsv


# In[55]:


pip install -U matplotlib


# In[56]:


#from platform import python_version

#print(python_version())


# In[57]:


pip install feature_selector


# In[ ]:


from feature_selector import FeatureSelector
fs = FeatureSelector(data = X_train, labels = y_train)


# The first feature selection method is straightforward: find any columns with a missing fraction greater than a specified threshold. For this example we will use a threhold of 0.6 which corresponds to finding features with more than 60% missing values

# In[ ]:


fs.identify_missing(missing_threshold=0.6)


# In[ ]:


missing_features = fs.ops['missing']
missing_features[:10]


# In[ ]:


fs.plot_missing()


# In[ ]:


fs.missing_stats.head(10)


# The next method is straightforward: find any features that have only a single unique value

# In[ ]:


fs.identify_single_unique()


# In[ ]:


single_unique = fs.ops['single_unique']
single_unique


# In[ ]:


fs.plot_unique()


# In[ ]:


fs.unique_stats.sample(5)


# This method finds pairs of collinear features based on the Pearson correlation coefficient. For each pair above the specified threshold (in terms of absolute value), it identifies one of the variables to be removed. We need to pass in a correlation_threshold
# 
# For each pair, the feature that will be removed is the one that comes last in terms of the column ordering in the dataframe. (This method does not one-hot encode the data beforehand unless one_hot=True. Therefore correlations are only calculated between numeric columns)

# In[ ]:


fs.identify_collinear(correlation_threshold=0.975)


# In[ ]:


correlated_features = fs.ops['collinear']
correlated_features[:5]


# In[ ]:


fs.plot_collinear()


# In[ ]:


fs.plot_collinear(plot_all=True)


# In[ ]:


fs.identify_collinear(correlation_threshold=0.98)
fs.plot_collinear()


# In[ ]:


fs.record_collinear.head()


# This method relies on a machine learning model to identify features to remove. It therefore requires a supervised learning problem with labels. The method works by finding feature importances using a gradient boosting machine implemented in the LightGBM library.
# 
# To reduce variance in the calculated feature importances, the model is trained a default 10 times. The model is also by default trained with early stopping using a validation set (15% of the training data) to identify the optimal number of estimators to train. The following parameters can be passed to the identify_zero_importance method:
# 
# task: either classification or regression. The metric and labels must match with the task
# eval_metric: the metric used for early stopping (for example auc for classification or l2 for regression). To see a list of available metrics, refer to the LightGBM docs
# n_iterations: number of training runs. The feature importances are averaged over the training runs (default = 10)
# early_stopping: whether to use early stopping when training the model (default = True). Early stopping stops training estimators (decision trees) when the performance on a validation set no longer decreases for a specified number of estimators (100 by default in this implementation). Early stopping is a form of regularization used to prevent overfitting to training data
# The data is first one-hot encoded for use in the model. This means that some of the zero importance features may be created from one-hot encoding. To view the one-hot encoded columns, we can access the one_hot_features of the FeatureSelector.
# 
# Note of caution: in contrast to the other methods, the feature imporances from a model are non-deterministic (have a little randomness). The results of running this method can change each time it is run.

# In[ ]:


fs.identify_zero_importance(task = 'classification', eval_metric = 'auc', 
                            n_iterations = 10, early_stopping = True)


# In[ ]:


one_hot_features = fs.one_hot_features
base_features = fs.base_features
print('There are %d original features' % len(base_features))
print('There are %d one-hot features' % len(one_hot_features))


# In[ ]:


fs.data_all.head(10)


# In[ ]:


zero_importance_features = fs.ops['zero_importance']
zero_importance_features[10:15]


# Plotting Feature Importances

# In[ ]:


fs.plot_feature_importances(threshold = 0.99, plot_n = 12)


# In[ ]:


fs.feature_importances.head(10)


# In[ ]:


one_hundred_features = list(fs.feature_importances.loc[:99, 'feature'])
len(one_hundred_features)


# In[ ]:


fs.identify_low_importance(cumulative_importance = 0.99)

low_importance_features = fs.ops['low_importance']
low_importance_features[:5]


# In[ ]:


low_importance_features = fs.ops['low_importance']
low_importance_features[:5]


# Removing Features

# In[ ]:


train_no_missing = fs.remove(methods = ['missing'])
train_no_missing_zero = fs.remove(methods = ['missing', 'zero_importance'])
all_to_remove = fs.check_removal()
all_to_remove[10:25]
train_removed = fs.remove(methods = 'all')


# In[ ]:


#To remove one hot features
train_removed_all = fs.remove(methods = 'all', keep_one_hot=False)


# In[ ]:


print('Original Number of Features', train.shape[1])
print('Final Number of Features: ', train_removed_all.shape[1])


# To Acheive all the steps above in simple code

# In[ ]:


fs = FeatureSelector(data = train, labels = train_labels)

fs.identify_all(selection_params = {'missing_threshold': 0.6, 'correlation_threshold': 0.98, 
                                    'task': 'classification', 'eval_metric': 'auc', 
                                     'cumulative_importance': 0.99})

train_removed_all_once = fs.remove(methods = 'all', keep_one_hot = True)

fs.feature_importances.head()


# In[ ]:





# PyImpetus Implementation

# In[58]:




model = PPIMBC(model, p_val_thresh, num_simul, simul_size, simul_type, sig_test_type, cv, verbose, random_state, n_jobs)


# In[ ]:


from PyImpetus import PPIMBC, PPIMBR

# Import the algorithm. PPIMBC is for classification and PPIMBR is for regression
from PyImeptus import PPIMBC, PPIMBR
# Initialize the PyImpetus object
model = PPIMBC(model=SVC(random_state=27, class_weight="balanced"), p_val_thresh=0.05, num_simul=30, simul_size=0.2, simul_type=0, sig_test_type="non-parametric", cv=5, random_state=27, n_jobs=-1, verbose=2)
# The fit_transform function is a wrapper for the fit and transform functions, individually.
# The fit function finds the MB for given data while transform function provides the pruned form of the dataset
df_train = model.fit_transform(X_train, y_train)
df_test = model.transform(df_test)
# Check out the MB
print(model.MB)
# Check out the feature importance scores for the selected feature subset
print(model.feat_imp_scores)
# Get a plot of the feature importance scores
model.feature_importance()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




