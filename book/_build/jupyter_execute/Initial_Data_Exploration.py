#!/usr/bin/env python
# coding: utf-8

# # Initial Data Exploration  
# The purpose of our initial data exploration is to:
# <ol type = "a">
#     <li>Check the validity of the data and perform data cleaning methods if needed.</li>
#     <li>View the statistical details of the data and perform data visualization to improve our understanding of the data</li>
#     <li>Initiate new hypotheses on both the future clustering and evaluation method.</li>
#     <li>Validate assumptions of any clustering methods we intend to use & perform transformations if needed.</li>
#     <li>Measure clustering & central tendency.</li>
# </ol>
# 
# If you are viewing this as an HTML page, please use the content toolbar to the right for quick access to different sections.
# 
# ## Importing required libraries 
# Data processing

# In[1]:


import pandas as pd


# Data Visualization 

# In[2]:


import matplotlib.pyplot as plt


# For the purposes of this exploration, we load in 12 different csv files. 
# |    Type     | File Name |    Description      |
# | :------------ | -------------: | :------------ |
# |        Choices     |        choices_95.csv, choices_100.csv, choices_150.csv     | These CSV's contains all of the choices made by test-takers during the examined studies. **Note**, the 10 studies described in the Introduction section are grouped by the number of trails. The integer suffix of the file name indicates the number of trails performed. For example, the 1<sup>st</sup> row and 2<sup>nd</sup> column instance of the choices_95.csv file describes a participant's 2<sup>nd</sup> card choice in a 95 trail study.   |    
# |     Wins     |      wi_95.csv, wi_100.csv, wi_150.csv      |     These datasets describe the rewards received by participants in 95, 100 and 150 trail investigations, as indicated by the suffix. For example, the  3<sup>rd</sup> row and 5<sup>th</sup> column entry of the wi_100.csv file details the monetary gain received by a participant on their 5<sup>th</sup> choice in 100 trail study.     | 
# |        Losses    |        lo_95.csv, lo_100.csv, lo_150.csv     |   These files contain the loses received by participants in 95, 100 and 150 trail investigations, as indicated by the suffix. For example, the  2<sup>nd</sup> row and 8<sup>th</sup> column entry of the lo_150.csv file details the monetary penalty received by a participant on their 8<sup>th</sup> choice in 150 trail study.      |        
# |     Index     |      index_95.csv, index_100.csv, index_150.csv    | index_95.csv, index_100.csv, and index_150.csv map the first author of the study that reports the data to the corresponding subject. |      

# In[3]:


choices_95 = pd.read_csv('data/choice_95.csv')
choice_100 = pd.read_csv('data/choice_100.csv')
choice_150 = pd.read_csv('data/choice_150.csv')


# In[4]:


win_95 = pd.read_csv('data/wi_95.csv')
win_100 = pd.read_csv('data/wi_100.csv')
win_150 = pd.read_csv('data/wi_150.csv')


# In[5]:


loss_95 = pd.read_csv('data/lo_95.csv')
loss_100 = pd.read_csv('data/lo_100.csv')
loss_150 = pd.read_csv('data/lo_150.csv')


# In[6]:


index_95 = pd.read_csv('data/lo_95.csv')
index_100 = pd.read_csv('data/lo_100.csv')
index_150 = pd.read_csv('data/lo_150.csv')


# In[ ]:





# ## Data Cleaning  
# The purpose of our initial data exploration is to:
# <ol type = "a">
#     <li>Check the validity of the data and perform data cleaning methods if needed.</li>
#     <li>View the statistical details of the data and perform data visualization to improve our understanding of the data</li>
#     <li>Initiate new hypotheses on both the future clustering and evaluation method.</li>
#     <li>Validate assumptions of any clustering methods we intend to use & perform transformations if needed.</li>
#     <li>Measure clustering & central tendency.</li>
# </ol>
# 
# If you are viewing this as an HTML page, please use the content toolbar to the right for quick access to different sections.
# 
# ## Importing required libraries 
# ### Data processing

# In[7]:


# Fixing random state for reproducibility
np.random.seed(19680801)

N = 10
data = [np.logspace(0, 1, 100) + np.random.randn(100) + ii for ii in range(N)]
data = np.array(data).T
cmap = plt.cm.coolwarm
rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, N)))


from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
                Line2D([0], [0], color=cmap(.5), lw=4),
                Line2D([0], [0], color=cmap(1.), lw=4)]

fig, ax = plt.subplots(figsize=(10, 5))
lines = ax.plot(data)
ax.legend(custom_lines, ['Cold', 'Medium', 'Hot']);


# There is a lot more that you can do with outputs (such as including interactive outputs)
# with your book. For more information about this, see [the Jupyter Book documentation](https://jupyterbook.org)
