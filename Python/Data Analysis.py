#!/usr/bin/env python
# coding: utf-8

# In[42]:


##### Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from random import sample

# To avoid warnings
import warnings
warnings.filterwarnings("ignore")


# In[45]:


##### Import data

#X-axis accelerometry
dfx = pd.read_csv('accs_x.csv', names=['id_','time_ref','value_acc_x'], header=0)

#Y-axis accelerometry
dfy = pd.read_csv('accs_y.csv', names=['id_','time_ref','value_acc_y'], header=0)

#Z-axis accelerometry
dfz = pd.read_csv('accs_z.csv', names=['id_','time_ref','value_acc_z'], header=0)

#energy data associated with accelerometry and heart rate per time interval
df_energy = pd.read_csv('energy.csv')


# In[6]:


#Datasets info


# dfx
print(f' acc_x '.center(50,'#'))
print(dfx.info())
print(dfx.describe())
print(' ')

# dfy
print(f' acc_y '.center(50,'#'))
print(dfy.info())
print(dfy.describe())
print(' ')

# dfz
print(f' acc_z '.center(50,'#'))
print(dfz.info())
print(dfz.describe())
print(' ')

# df_energy
print(f' energy '.center(50,'#'))
print(df_energy.info())
print(df_energy.describe())
print(' ')


# ## Preprocessing data

# We create the variable 'jerk', which is the derivative of the acceleration. To do it, we approximate it using the variable as discrete and assuming that the intervals of time are constant. 
# The acceleration is a vector, therefore its derivative has vectorial character as well. Then, we calculate the partial derivatives respect to each axis.

# In[16]:


dfy


# In[46]:


dfx['jerk_x']=dfx['value_acc_x']-dfx.groupby('id_')['value_acc_x'].shift(-1)
dfy['jerk_y']=dfy['value_acc_y']-dfy.groupby('id_')['value_acc_y'].shift(-1)
dfz['jerk_z']=dfz['value_acc_z']-dfz.groupby('id_')['value_acc_z'].shift(-1)


# In[47]:


#Removing missing values created
dfx.dropna(inplace=True)
dfy.dropna(inplace=True)
dfz.dropna(inplace=True)


# In[48]:


##### Merge the three axes

dfx.time_ref=dfx.time_ref.apply(lambda x: x[x.rfind("_")+1:])
dfy.time_ref = dfy.time_ref.apply(lambda x: x[x.rfind("_")+1:])
dfz.time_ref = dfx.time_ref.apply(lambda x: x[x.rfind("_")+1:])

df_acc= pd.merge(pd.merge(dfx,dfy,how="inner",on=['id_', 'time_ref']),dfz,how="inner",on=['id_', 'time_ref'])


# Creatation of the acceleration module and jerk module variables, which are scalar magnitudes with crucial information about the activity and I assumed that are relevant to the model.

# In[49]:


df_acc["mod_acc"] = np.sqrt((df_acc.value_acc_x**2) + (df_acc.value_acc_y**2) + (df_acc.value_acc_z**2))
df_acc["mod_jerk"] = np.sqrt((df_acc.jerk_x**2) + (df_acc.jerk_y**2) + (df_acc.jerk_z**2))


# In[50]:


df_acc


# ## Data visualization
# 

# In[40]:


# Let's see some time series and distributions of the variables

sns.lineplot("time_ref","value_acc_x",data=df_acc)
sns.lineplot("time_ref","value_acc_y",data=df_acc)
sns.lineplot("time_ref","value_acc_z",data=df_acc)
sns.lineplot("time_ref","mod_acc",data=df_acc)
sns.lineplot("time_ref","mod_jerk",data=df_acc)


# In[ ]:


##### X,y,z accelerometry over time
#I choose 15 random ids

random_id = df_acc.loc[sample(range(0,len(df_acc.id_)),15)].id_.values
data = df_acc[df_acc.id_.isin(random_id)]


# In[62]:


for feat in ['value_acc_x', 'value_acc_y', 'value_acc_z']:
    plt.figure(figsize=(15, 10))
    sns.lineplot('time_ref', feat, data = data, hue = 'id_', legend=False, palette='pastel')
    plt.tick_params(bottom = False)
    plt.title(f'{feat} over time')
    plt.show()


# In[57]:


##### Acceleration module and jerk module variables over time
#I choose 15 random ids

random_id = df_acc.loc[sample(range(0,len(df_acc.id_)),15)].id_.values
data = df_acc[df_acc.id_.isin(random_id)]


# In[63]:


for feat in ['mod_acc', 'mod_jerk']:
    plt.figure(figsize=(15, 10))
    sns.lineplot('time_ref', feat, data = data, hue = 'id_', legend=False, palette='pastel')
    plt.tick_params(bottom = False)
    plt.title(f'{feat} over time')
    plt.show()


# ## Outliers
# I analyze the histograms of the key variables to locate possible failures in data collection. See that there are extreme values but a priori we cannot delete them since we cannot conclude with this data that they are errors in the measurement. We must be careful with them, because they can affect both the scoring and the model itself.

# In[64]:


plt.figure(figsize=(15, 10))
sns.kdeplot(df_acc.mod_acc)
plt.xlabel("mod_acc")
plt.ylabel("density")
plt.title('Data distribution of acceleration module')
plt.show()


# In[66]:


##### Relationship between acceleration module and jerk module
sns.jointplot(x = 'mod_acc', y = 'mod_jerk', data = df_acc, kind = 'reg',  height = 25)


# In[67]:


df_energy.columns


# In[70]:


# Let's see the distribution of energy data associated 
# with accelerometry and heart rate per time interval

for feat in ['value_energy','value_Hr']:

    plt.figure(figsize=(15, 10))
    sns.kdeplot(df_energy[feat])
    plt.xlabel(f"{feat}")
    plt.ylabel("density")
    plt.title(f'Data distribution of {feat}')
    plt.show()


# ## Features Engineering
# 
# I differentiate between two sets of variables: 
#    
# -First, I want to collect as much information as possible about the distribution of acceleration and jerk per invidual (mean, st, median...).
# 
# -Second, I'll look for working in the system of frequencies and moments by transformations. It's understood that, due to the nature of the problem, this transformation can help to extract more information from the physical activity. We take only one observation per id in order to train the model with the energy data.
# 

# In[72]:


# Percentile function for agregate
def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_


# In[73]:


df_acc_agg = df_acc.groupby("id_").agg({'mod_acc': [np.sum, percentile(25), percentile(75), np.mean, np.median],
                                        'mod_jerk':[np.sum, percentile(25), percentile(75), np.mean, np.median ]}).reset_index()
df_acc_agg.columns = ['_'.join(col).strip() for col in df_acc_agg.columns.values]   
df_acc_agg.rename(columns={"id__":"id_"},inplace=True)
df_acc_agg


# In[74]:


##### Merge of both datasets and save for modelling notebook

df_final = pd.merge(df_acc_agg, df_energy, how="inner", on="id_")
df_final.to_csv('energy_data.csv', index = False)
print('Done')


# In[ ]:




