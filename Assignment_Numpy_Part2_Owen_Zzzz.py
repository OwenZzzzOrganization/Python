#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
data = load_iris()
print(dir(data)) # 查看data所具有的属性或方法


# In[3]:


print(data.DESCR) # 查看数据集的简介


# In[4]:


data.feature_names


# In[44]:


<div class="burk">
# data.data</div><i class="fa fa-lightbulb-o "></i>


# In[6]:


data.target_names


# In[7]:


data.target


# In[ ]:





# In[ ]:


###### 导入鸢尾属植物数据集，保持文本不变。


# In[8]:


rawdata = pd.DataFrame(data.data,columns=data.feature_names,copy=True)
rawdata['class'] = data.target
rawdata.head()


# In[ ]:





# In[15]:


###### 求出鸢尾属植物萼片长度的平均值、中位数和标准差（第1列，sepallength）


# In[9]:


# rawdata['sepal length (cm)'].describe().index


# In[84]:


rawdata['sepal length (cm)'].describe()[['mean','50%','std']]


# In[ ]:





# In[16]:


###### 创建一种标准化形式的鸢尾属植物萼片长度，其值正好介于0和1之间，这样最小值为0，最大值为1（第1列，sepallength）。


# In[9]:


from sklearn.preprocessing import MinMaxScaler

MinMaxFunction = MinMaxScaler()
sepal_length_Scaled = MinMaxFunction.fit_transform(np.array(rawdata['sepal length (cm)']).reshape(-1,1))
# sepal_length_Scaled


# In[ ]:





# In[17]:


###### 找到鸢尾属植物萼片长度的第5和第95百分位数（第1列，sepallength）。


# In[86]:


np.percentile(np.array(rawdata['sepal length (cm)']),[5,95])
# np.percentile(rawdata,[5,95],axis = 0)


# In[ ]:





# In[18]:


###### 把iris_data数据集中的20个随机位置修改为np.nan值。


# In[13]:


# np.random.choice(rawdata.values.reshape(1,-1)[0],20)


# In[10]:


# random select index & column separately - drawback: can be repetitively
np.random.seed(1)
random_index_list = np.random.randint(0,150,20)
random_column_list = np.random.randint(0,4,20)
pd.DataFrame(list(zip(random_index_list,random_column_list))).head()


# In[261]:


# np.random.seed(1)
# test_list = np.random.choice(range(0,600),20,replace=False)
# test_list


# In[12]:


np.random.seed(1)
test_choiced = [[x%150,x//150] for x in np.random.choice(range(0,600),20,replace=False)]
test_choiced


# In[13]:


rawdata_nan = rawdata.copy()
for i in test_choiced:
    rawdata_nan.iloc[i[0],i[1]] = np.NAN

rawdata_nan.head()


# In[ ]:





# In[19]:


###### 在iris_data的sepallength中查找缺失值的个数和位置（第1列）。


# In[30]:


# np.isnan(rawdata_nan['sepal length (cm)'])


# In[14]:


print('count NaN values',len(np.where(np.isnan(rawdata_nan['sepal length (cm)']))[0]))
print('location NaN values',np.where(np.isnan(rawdata_nan['sepal length (cm)']))[0])


# In[ ]:





# In[ ]:





# In[20]:


###### 筛选具有 sepallength（第1列）< 5.0 并且 petallength（第3列）> 1.5 的 iris_data行。


# In[15]:


rawdata_nan[(rawdata_nan['sepal length (cm)'] < 5) & (rawdata_nan['petal length (cm)'] > 1.5)]


# In[21]:


###### 选择没有任何 nan 值的 iris_data行。


# In[16]:


# rawdata_nan[np.isnan(rawdata_nan.sum(axis = 1,skipna=False))==False]


# In[17]:


rawdata_nan[rawdata_nan.isna().sum(axis = 1) == 0].head()


# In[ ]:





# In[22]:


###### 计算 iris_data 中sepalLength（第1列）和petalLength（第3列）之间的相关系数。


# In[94]:


np.corrcoef(rawdata['sepal length (cm)'].values,rawdata['petal length (cm)'].values)


# In[18]:


# test_1 = np.array([1.4,1.4,np.nan,1.5,1.4])
# test_2 = np.array([4, 4.9,np.nan,4.6,5])

# np.corrcoef(test_1,test_2,rowvar=False)


# In[19]:


# rawdata.info()


# In[ ]:





# In[23]:


###### 找出iris_data是否有任何缺失值。


# In[20]:


print('nan value exist' if np.isnan(rawdata_nan).sum().sum() > 0 else 'no nan value')


# In[23]:


rawdata_nan


# In[ ]:


###### 在numpy数组中将所有出现的nan替换为0。


# In[29]:



column_names


# In[32]:


column_names = np.append(np.array(data.feature_names),'class')
rawdata_zero = pd.DataFrame(np.where(np.isnan(rawdata_nan),0,rawdata_nan),columns=column_names,copy=True)
rawdata_zero.head()


# In[ ]:





# In[ ]:


###### 找出鸢尾属植物物种中的唯一值和唯一值出现的数量。


# In[33]:


rawdata_str = rawdata_zero.astype(str)
rawdata_zero['flag_unique_species'] = rawdata_str['sepal length (cm)']+rawdata_str['sepal width (cm)']+rawdata_str['petal length (cm)']+rawdata_str['petal width (cm)']
rawdata_zero.head()


# In[117]:


# rawdata_zero.groupby('flag_unique_species')['flag_unique_species'].count()


# In[ ]:


###### 将 iris_data 的花瓣长度（第3列）以形成分类变量的形式显示。定义：Less than 3 --> ‘small’；3-5 --> ‘medium’；’>=5 --> ‘large’。


# In[35]:


bins = [-1,3,5,99]
petal_labels = ['small','medium','large']

rawdata_zero['petal_length_category'] = pd.cut(rawdata_zero['petal length (cm)'],bins = bins,labels=petal_labels)
rawdata_zero[['petal length (cm)','petal_length_category']].head()


# In[ ]:





# In[ ]:


###### 在 iris_data 中创建一个新列，其中 volume 是 (pi x petallength x sepallength ^ 2）/ 3。


# In[36]:


rawdata['petal_sepal_math'] = np.pi*rawdata['petal length (cm)']*pow(rawdata['sepal length (cm)'],2)/3
rawdata['petal_sepal_math'].head()


# In[ ]:





# In[ ]:


###### 随机抽鸢尾属植物的种类，使得Iris-setosa的数量是Iris-versicolor和Iris-virginica数量的两倍。


# In[37]:


rawdata['classes_weight'] = pd.Series(data.target).apply(lambda x: 2 if x==0 else 1)
rawdata['classes_weight'].head()


# In[39]:


rawdata.sample(frac = 0.05, weights=rawdata['classes_weight'],random_state=1)


# In[ ]:





# In[ ]:


###### 根据 sepallength 列对数据集进行排序。


# In[41]:


rawdata.sort_values(by='sepal length (cm)').head()


# In[ ]:





# In[ ]:


###### 在鸢尾属植物数据集中找到最常见的花瓣长度值（第3列）。


# In[42]:


print('most frequent petal length is {}(cm)'.format(rawdata.groupby(by='petal length (cm)')['class'].count().sort_values(ascending = False).index[0]))


# In[ ]:





# In[ ]:


###### 在鸢尾花数据集的 petalwidth（第4列）中查找第一次出现的值大于1.0的位置。


# In[43]:


print('row number where petalwidth first > 1.0 is row_{}'.format(rawdata[rawdata['petal width (cm)']>1].index[0]))

