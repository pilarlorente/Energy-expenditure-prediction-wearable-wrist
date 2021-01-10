#!/usr/bin/env python
# coding: utf-8

# In[14]:


##### Import packages
# Basic packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Modelling packages
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# To avoid warnings
import warnings
warnings.filterwarnings("ignore")


# In[15]:


df=pd.read_csv('energy_data.csv')


# In[16]:


df


# In[17]:


##### Defining MAPE(Mean Absolute Percentage Error)

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[18]:


##### Columns based on time, change the format

for col in ['date_Hr', 'startDate_energy', 'endDate_energy']:
    df[col] = pd.to_datetime(df[col])


# In[19]:


#Creating new columns of time
df["time_elapsed"] = (df["startDate_energy"] - df["date_Hr"]).astype('timedelta64[s]')
df["day"] = df.date_Hr.apply(lambda x: x.day)
df["month"] = df.date_Hr.apply(lambda x: x.month)
df["hour"] = df.date_Hr.apply(lambda x: x.hour)

df.drop(['date_Hr', 'startDate_energy', 'endDate_energy','totalTime_energy'], axis=1, inplace=True)
df.head(10)


# ## Modelling

# In[20]:


#target 
target= "value_energy"

#features
features=list(df.columns)
features.remove("id_")
features.remove("value_energy")

#Division
X = df[features].values
y = df[target].values

#Dividimos en dos conjuntos de datos para entrenar i testear los modelos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# ## Decision Tree

# In[25]:


model = DecisionTreeRegressor()

params = {'criterion':['mae'],
        'max_depth': [4,5,6,7],
        'max_features': [7,8,9,10], 
        'max_leaf_nodes': [30,40,50], 
        'min_impurity_decrease' : [0.0005,0.001,0.005], 
        'min_samples_split': [2,4]}

# GridSearch
grid_solver = GridSearchCV(estimator = model, 
                   param_grid = params,
                   scoring = 'neg_median_absolute_error',
                   cv = 10,
                   refit = 'neg_median_absolute_error',
                   verbose = 0)

model_result = grid_solver.fit(X_train,y_train)

reg = model_result.best_estimator_
reg.fit(X,y)


# In[26]:


print(model_result.best_params_)


# In[30]:



best_model=model_result.best_estimator_
#best model
final_model=best_model.fit(X,y)


# features importances
len(df[features].columns)
len(final_model.feature_importances_)
importances=pd.DataFrame([df[features].columns,final_model.feature_importances_], index=["feature","importance"]).T
importances.sort_values('importance', ascending=False)


# ## Random Forest
# 
# To do the CV of this model we will use a param_grid based on the results of the decision tree, because the random forest is born from the "trade-off" between bias and variance. The tree has a low bias but a high variance, so we will try to combine models with a low bias and that are not fully correlated to reduce the variance.

# In[ ]:


##### Activity Intensity
In addition to calculate the energy expenditure, for each time interval, the level of intensity of the activity carried out must be calculated. . The classification of the intensity level is based on the metabolic equivalents or METS (kcal/kg*h) of the activity being:light activity < 3 METS, moderate 3 - 6 METS and intense > 6 METS. . To estimate it, I consider a person of 75 kg. The model chosen is the Random Forest Regressor which has the lowest MAPE.

reg = RandomForestRegressor(criterion='mae', max_depth=8, max_features=12,
                      max_leaf_nodes=30, min_impurity_decrease=0.001,
                      n_estimators=15)
reg.fit(X,y)

yhat = reg.predict(X)

ids = df_acc_final['id_'].to_frame()
ids['yhat'] = yhat
ids['METs'] = ids["yhat"] / (75 * 62 / 3600)

conditions = [(ids["METs"] < 3 ),((3 < ids["METs"]) & (ids["METs"] < 6)),(ids["METs"] > 6)]
names = ['ligera', 'moderada', 'intensa']
ids['intensidad'] = np.select(conditions, names)

ids


# In[22]:


model = RandomForestRegressor()

params = {'bootstrap': [True],
        'criterion':['mae'],
        'max_depth': [8,10],
        'max_features': [10,12],
        'max_leaf_nodes': [10,20,30],
        'min_impurity_decrease' : [0.001,0.01],
        'min_samples_split': [2,4],
        'n_estimators': [10,15]}

# GridSearch
grid_solver = GridSearchCV(estimator = model, 
                   param_grid = params,
                   scoring = 'neg_median_absolute_error',
                   cv = 7,
                   refit = 'neg_median_absolute_error',
                   verbose = 0)

model_result = grid_solver.fit(X_train,y_train)

reg = model_result.best_estimator_
reg.fit(X,y)


# In[23]:


##### Mean Absolute Percentage Error

yhat = reg.predict(X_test)
print("Mean Absolute Percentage Error = %.2f" %mean_absolute_percentage_error(yhat,y_test),'%')


# In[24]:


##### Feature Importance

features_importance = reg.feature_importances_
features_array = np.array(features)
features_array_ordered = features_array[(features_importance).argsort()[::-1]]
features_array_ordered

plt.figure(figsize=(16,10))
sns.barplot(y = features_array, x = features_importance, orient='h', order=features_array_ordered[:50])

plt.show()


# ## Activity Intensity
#  
# In addition to calculate the energy expenditure, for each time interval, the level of intensity of the activity carried out must be calculated. The classification of the intensity level is based on the metabolic equivalents or METS (kcal/kg*h) of the activity being:
#  
# light activity < 3 METS, moderate 3 - 6 METS and intense > 6 METS. 
#  
# To estimate it, I consider a person of 75 kg. The model chosen is the Random Forest Regressor which has the lowest MAPE.

# In[33]:


df


# In[47]:


reg = RandomForestRegressor(criterion='mae', max_depth=8, max_features=12,
                      max_leaf_nodes=30, min_impurity_decrease=0.001,
                      n_estimators=15)
reg.fit(X,y)

yhat = reg.predict(X)

ids = df['id_'].to_frame()
ids['yhat'] = yhat
ids['METs'] = ids["yhat"] / (75 * 62 / 3600)

conditions = [(ids["METs"] < 3 ),((3 < ids["METs"]) & (ids["METs"] < 6)),(ids["METs"] > 6)]
names = ['light', 'moderate', 'intense']
ids['intensity'] = np.select(conditions, names)

ids


# The substantial improvement that can be seen when we introduce the non-linearity of the model invites us to deduce that the relationships between the variables and the target are not linear. More efforts should be made to collect all the information on physical activity. Additional information about individuals such as age, sex and weight would help to improve the ASM of the model in several points.
# 

# In[ ]:




