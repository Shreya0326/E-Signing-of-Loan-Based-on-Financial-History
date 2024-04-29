#!/usr/bin/env python
# coding: utf-8

# ## Importing the Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Readind the data

# In[2]:


data = pd.read_csv("financial_data.csv")


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


data.select_dtypes(include='object').columns


# In[8]:


data.select_dtypes(include=['int64','float64']).columns


# ## Dealing with missing data

# In[9]:


data.isnull().values.any()


# In[10]:


# check the numerical value
data.isnull().values.sum()


# In[11]:


## Dealing the null values using heatmap
plt.figure(figsize=(16,9))
sns.heatmap(data=data.isnull(),cmap = 'coolwarm')
plt.show()


# ## Encoding the categorical data

# In[12]:


data.head()


# In[13]:


# check columns with categorical


# In[14]:


data.select_dtypes(include='object').columns


# In[15]:


# Unique value in categorical value
data['pay_schedule'].unique()


# In[16]:


# checking the number
data['pay_schedule'].nunique()


# In[17]:


data.shape


# In[18]:


# Encode the categorical value
data = pd.get_dummies(data=data,drop_first=True)


# In[19]:


data.head()


# In[20]:


data.shape


# ## Countplot

# In[21]:


sns.countplot(data['e_signed'],label='Count')
plt.show()


# In[22]:


# checking the  e_signed values
(data.e_signed == 1).sum()


# In[23]:


# checking the not e_signed values
(data.e_signed == 0).sum()


# ## Correlation matrix and Heatmap

# In[24]:


data.head()


# In[25]:


data_2 = data.drop(columns = ['entry_id','e_signed'])


# In[26]:


data_2.shape


# In[27]:


data_2.corrwith(data.e_signed).plot.bar(
    figsize=(16,9),title = 'Correlation with E Signed',
    rot = 45, grid = True)


# In[28]:


# create correlation matrix
corr = data.corr()


# In[29]:


corr


# In[30]:


# correlaton heatmap
plt.figure(figsize = (16,9))
ax = sns.heatmap(corr,annot=True,linewidths=2)


# In[31]:


corr_2 = corr[corr>0.2]


# In[32]:


plt.figure(figsize=(16,9))
ax = sns.heatmap(corr_2,annot=True,linewidths = 2)


# ## Restructure the dataset

# In[33]:


data.head()


# In[34]:


# combine two columns 'months_employed' and 'years_employed'
data['employment_months'] = (data.months_employed + (data.years_employed*12))


# In[35]:


data.head()


# In[36]:


data = data.drop(columns=['months_employed', 'years_employed'])


# In[37]:


data.head()


# In[38]:


# combine two columns 'personal_account_m' and 'personal_account_y'
data['personal_accounts_months'] = (data.personal_account_m + (data.personal_account_y*12))


# In[39]:


data.head()


# In[40]:


# now dropping the column personal_account_m' and 'personal_account_y'
data = data.drop(columns=['personal_account_m','personal_account_y'])


# In[41]:


data.head()


# In[42]:


data.shape


# In[43]:


data.columns


# ## Spliting the data into test and train

# In[44]:


data.head()


# In[45]:


# Independent variable
x = data.drop(columns=['entry_id','e_signed'])


# In[46]:


# dependent variable
y = data['e_signed']


# In[47]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[48]:


x_train.shape


# In[49]:


x_test.shape


# In[50]:


y_train.shape


# In[51]:


y_test.shape


# ## Feature scaling

# In[52]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[53]:


x_train


# In[54]:


x_test


# ## Building the model

# ### Logestic Regression

# In[55]:


from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression(random_state=0)
classifier_lr.fit(x_train,y_train)


# In[56]:


y_pred = classifier_lr.predict(x_test)


# In[57]:


from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,precision_score,recall_score


# In[58]:


acc = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)


# In[59]:


results = pd.DataFrame([['Logestice Regression',acc,prec,rec,f1]],
                      columns = ['model','Accuracy','Precision','Recall','f1_score'])


# In[60]:


results


# ## SVM (linear)

# In[61]:


from sklearn.svm import SVC
classifier_svm = SVC(kernel='linear',random_state=0)
classifier_svm.fit(x_train,y_train)


# In[62]:


y_pred = classifier_svm.predict(x_test)


# In[63]:


acc = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)


# In[64]:


model_results = pd.DataFrame([['SVM_Linear',acc,prec,rec,f1]],
                      columns = ['model','Accuracy','Precision','Recall','f1_score'])


# In[65]:


results = results.append(model_results,ignore_index=True)
results


# ## SVM(RBF) , "Radial Basis Function"

# The RBF kernel is particularly useful for capturing complex non-linear relationships between features. It allows SVM to create non-linear decision boundaries in the input space, making it effective for a wide range of classification tasks, including those where the classes are not linearly separable.
# 
# When using SVM with an RBF kernel, it's essential to tune the γγ parameter appropriately to achieve the best performance and avoid overfitting or underfitting.

# In[66]:


from sklearn.svm import SVC
classifier_svm_2 = SVC(kernel='rbf',random_state = 0)
classifier_svm_2.fit(x_train,y_train)


# In[67]:


y_pred = classifier_svm_2.predict(x_test)


# In[68]:


acc = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)


# In[69]:


model_results = pd.DataFrame([['SVM_rbf',acc,prec,rec,f1]],
                            columns=['model','Accuracy','Precision','Recall','f1_score'])


# In[70]:


results = results.append(model_results,ignore_index=True)


# In[71]:


results


# ## Random_forest

# In[72]:


from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(random_state = 0,n_estimators=100,criterion='entropy') 
classifier_rf.fit(x_train,y_train)


# In[73]:


y_pred = classifier_rf.predict(x_test)


# In[74]:


acc = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)


# In[75]:


model_results = pd.DataFrame([['Random Foredt',acc,prec,rec,f1]],
                            columns=['model','Accuracy','Precision','Recall','f1_score'])


# In[76]:


results = results.append(model_results,ignore_index=True)
results


# ## K-fold Cross validation and confusion matrix

# In[ ]:


# k-Fold Cross Validation technique will create 10 training test folds
# that means we will get 10 accuracies and will compute the average of these 10 accuracies


# In[80]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier_rf,X=x_train,y=y_train,cv=10)
print('Accuracy is {:.2f} % '.format(accuracies.mean()*100))
print('Standard Deviation is {:.2f} %'.format(accuracies.std()*100))


# In[81]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)


# ##  XGBoost

# In[83]:


from xgboost import XGBClassifier
classifier_xgb = XGBClassifier(random_state=0)
classifier_xgb.fit(x_train,y_train)


# In[84]:


y_pred = classifier_xgb.predict(x_test)


# In[86]:


acc = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)


# In[91]:


model_results = pd.DataFrame([['XGBoost',acc,prec,rec,f1]],
                            columns=['model','Accuracy','Precision','Recall','f1_score'])


# In[92]:


results = results.append(model_results,ignore_index=True)


# In[93]:


results


# ## K_fold cross validation and confusion matrix

# In[97]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier_xgb,X=x_train,y=y_train,cv=10)
print('Accuracy id {:.2f} %'.format(accuracies.mean()*100))
print('Standard Deviation is {:.2f} %'.format(accuracies.std()*100))


# In[99]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)


# ## Applying Randomized Search to find the beat parameter

# In[103]:


from sklearn.model_selection import RandomizedSearchCV


# In[104]:


parameters = {
    'learning_rate' : [0.05,0.10,0.15,0.20,0.25,0.30],
    'max_depth' : [3,4,5,6,8,10,12,15],
    'min_child_weight' : [1,3,5,7],
    'gamma' : [0.0,0.1,0.2,0.3,0.4],
    'colsample_bytree': [0.3,0.4,0.5,0.7]
}


# In[107]:


randomized_search = RandomizedSearchCV(estimator=classifier_xgb,param_distributions=parameters,
                                      n_iter=5,n_jobs=-1,scoring='roc_auc',cv=5,verbose=3)


# In[108]:


randomized_search.fit(x_train,y_train)


# In[109]:


randomized_search.best_estimator_


# In[110]:


randomized_search.best_params_


# In[111]:


randomized_search.best_score_


# ## Final Model (XGBoost)

# In[113]:


from xgboost import XGBClassifier
classifier = XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=0.4, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              gamma=0.3, grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=0.05, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=10, max_leaves=None,
              min_child_weight=5, missing=float('nan'), monotone_constraints=None,
              multi_strategy=None, n_estimators=None, n_jobs=None,
              num_parallel_tree=None, random_state=0)


# In[119]:


classifier.fit(x_train,y_train)


# In[115]:


y_pred = classifier_xgb.predict(x_test)


# In[116]:


acc = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)


# In[117]:


final_results = pd.DataFrame([['XGBoost',acc,prec,rec,f1]],
                            columns=['model','Accuracy','Precision','Recall','f1_score'])


# In[118]:


final_results


# ## K_fold cross validation and confusion matrix

# In[120]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier,X=x_train,y=y_train,cv=10)
print('Accuracy id {:.2f} %'.format(accuracies.mean()*100))
print('Standard Deviation is {:.2f} %'.format(accuracies.std()*100))


# In[123]:


from sklearn.metrics import confusion_matrix,accuracy_score, classification_report
cm = confusion_matrix(y_test,y_pred)
print(cm)


# ## Predicting the single observation

# In[126]:


data.head()


# In[128]:


data.columns


# In[129]:


single_obs = [[45,1,3100,2,1,650,30000,0.537398,0.703517,0.365856,0.515977,0.580918,0.380918,11,0,0,0,36,30]]


# In[130]:


print(classifier.predict(sc.transform(single_obs)))


# In[ ]:




