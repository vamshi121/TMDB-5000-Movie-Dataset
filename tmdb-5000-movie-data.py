#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Data Wrangling
# # General Dataset Properties

# 

# In[2]:


movies = pd.read_csv("/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv")
movies


# In[3]:


movies = pd.read_csv("/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv")
movies.columns


# In[4]:


credit = pd.read_csv("/kaggle/input/tmdb-movie-metadata/tmdb_5000_credits.csv")
credit


# In[5]:


credit = pd.read_csv("/kaggle/input/tmdb-movie-metadata/tmdb_5000_credits.csv")
credit.columns


# In[6]:


movies.info()


# In[7]:


movies.head(10)


# In[8]:


movies.describe()


# In[9]:


#filter the zero budget data
df_budget_zero = movies.query('budget == 0')
# choice the first three randomly
df_budget_zero.head(3)


# In[10]:


df_budget_0count =  movies.groupby('budget').count()['id']
df_budget_0count.head(2)


# In[11]:


movies['revenue'].drop_duplicates(inplace=True)
movies['revenue'].replace(r'\s+', 0, regex=True)
#count zero values in revenue data using groupby
df_revenue_0count =  movies.groupby('revenue').count()['id']
df_revenue_0count.head(2)


# In[12]:


#count zero values in runtime data using groupby
df_runtime_0count =  movies.groupby('runtime').count()['id']
df_runtime_0count.head(2)


# # Cleaning Decision Summary
# # Data Cleaning

# In[13]:


# After discussing the structure of the data and any problems that need to be
#   cleaned, perform those cleaning steps in the second part of this section.
# Drop extraneous columns
col = ['homepage', 'tagline', 'overview', 'status']
movies.drop(col, axis=1, inplace=True)


#  **Drop the duplicates.**

# In[14]:


#Drop the duplicates
movies.drop_duplicates(inplace=True)


# **Then, drop the null values in 'original_language', 'genres', 'original_title', 'revenue' columns.**

# In[15]:


#drop the null values in 'original_language', 'genres', 'original_title', 'revenue' columns
cal2 = ['original_language', 'genres', 'original_title', 'revenue']
movies.dropna(subset = cal2, how='any', inplace=True)


# In[16]:


# see if nulls are dropped.
movies.isnull().sum()


# In[17]:


#replace zero values with null values in the budget and revenue column.
movies['budget'] = movies['budget'].replace(0, np.NaN)
movies['revenue'] = movies['revenue'].replace(0, np.NaN)
movies['vote_average'] = movies['vote_average'].replace(0, np.NaN)
movies['vote_count'] = movies['vote_count'].replace(0, np.NaN)
# see if nulls are added in budget and revenue columns
movies.info()


# In[18]:


# directly filter the runtime data with nonzero value
movies.query('runtime != 0', inplace=True)
#check
movies.query('runtime == 0')


# # Cleaning Result Summary

# In[19]:


movies.info()


# In[20]:


movies.describe()


# # Exploratory Data Analysis
# 
# # Research Part 1: General Explore
# 
# # Question 1: Popularity Over Years

# In[21]:


movies.head(2)


# In[22]:


movies['release_date']=pd.to_datetime(movies['release_date'], format='%Y-%m-%d')
release_year = pd.DatetimeIndex(movies['release_date']).year
release_year
movies['release_year'] = release_year


# In[23]:


# compute the mean for popularity
p_mean = movies.groupby('release_year').mean()['popularity']
p_mean.tail()


# In[24]:


# compute the median for popularity
p_median = movies.groupby('release_year').median()['popularity']
p_median.tail()


# In[25]:


# build the index location for x-axis
index_mean = p_mean.index
index_median = p_median.index


# In[26]:


#set style
sns.set_style('whitegrid')
#set x, y axis data
#x1, y1 for mean data; x2, y2 for median data
x1, y1 = index_mean, p_mean
x2, y2 = index_median, p_median
#set size
plt.figure(figsize=(9, 4))
#plot line chart for mean and median
plt.plot(x1, y1, color = 'g', label = 'mean')
plt.plot(x2, y2, color = 'r', label = 'median')
#set title and labels
plt.title('Popularity Over Years')
plt.xlabel('Year')
plt.ylabel('Popularity');
#set legend
plt.legend(loc='upper left')


# # Research Part 2 : Find the Properties are Associated with Successful Movies
# # A. Function Prepare-- Build a level-devide function and a split string function.
# **A)The cut_into_quantile function- general use.**

# In[27]:


# quartile function
def cut_into_quantile(dfname ,column_name):
# find quartile, max and min values
    min_value = dfname[column_name].min()
    first_quantile = dfname[column_name].describe()[4]
    second_quantile = dfname[column_name].describe()[5]
    third_quantile = dfname[column_name].describe()[6]
    max_value = dfname[column_name].max()
# Bin edges that will be used to "cut" the data into groups
    bin_edges = [ min_value, first_quantile, second_quantile, third_quantile, max_value]
# Labels for the four budget level groups
    bin_names = [ 'Low', 'Medium', 'Moderately High', 'High'] 
# Creates budget_levels column
    name = '{}_levels'.format(column_name)
    dfname[name] = pd.cut(dfname[column_name], bin_edges, labels=bin_names, include_lowest = True)
    return dfname


# # B. Sample prepare-- Filter Top 100 and Worst 100 movies in each year as the research sample.
# **A) Select Top 100 popular movies in every year.**

# In[28]:


# split pipe characters and count their number of appeared times
#argument:dataframe_col is the target dataframe&column; num is the number of the top factor
def find_top(dataframe_col, num=3):
    # split the characters in the input column 
    #and make it to a list
    alist = dataframe_col.str.cat(sep='|').split('|')
    #transfer it to a dataframe
    new = pd.DataFrame({'top' :alist})
    #count their number of appeared times and
    #choose the top3
    top = new['top'].value_counts().head(num)
    return top


# **B) Select Top 100 high revenue movies in every year**.

# In[29]:


# Select Top 100 popular movies.
# fisrt sort it by release year ascending and popularity descending
df_top_p = movies.sort_values(['release_year','popularity'], ascending=[True, False])
#group by year and choose the top 100 high
df_top_p = df_top_p.groupby('release_year').head(100).reset_index(drop=True)
#check, it must start from 1960, and with high popularity to low
df_top_p.head(2)


# In[30]:


# Select Top 100 high revenue movies.
# fisrt sort it by release year ascending and revenue descending
df_top_r = movies.sort_values(['release_year','revenue'], ascending=[True, False])
#group by year and choose the top 100 high
df_top_r = df_top_r.groupby('release_year').head(100).reset_index(drop=True)
#check, it must start from 1960, and with high revenue to low
df_top_r.head(2)


# **C) Select Top 100 high score rating movies in every year.**

# In[31]:


# Select Top 100 high scorer ating movies.
# fisrt sort it by release year ascending and high scorer ating descending
df_top_s = movies.sort_values(['release_year','vote_average'], ascending=[True, False])
#group by year and choose the top 100 high
df_top_s = df_top_s.groupby('release_year').head(100).reset_index(drop=True)
#check, it must start from 1960, and with high scorer ating to low
df_top_s.head(2)


# **D) To compare to results, I also create three subdataset for the last 100 movies.**

# In[32]:


# the last 100 popular movies in every year
df_low_p = movies.sort_values(['release_year','popularity'], ascending=[True, True])
df_low_p = df_low_p.groupby('release_year').head(100).reset_index(drop=True)
print('\n{:s}'.format('\u0332'.join("THE LAST 100 POPULAR MOVIES IN EVERY YEAR:\n")))
print(df_low_p)
# the last 100 high revenue movies in every year
df_low_r = movies.sort_values(['release_year','revenue'], ascending=[True, True])
df_low_r = df_low_r.groupby('release_year').head(100).reset_index(drop=True)
print('\n{:s}'.format('\u0332'.join("THE LAST 100 HIGH REVENUE MOVIES IN EVERY YEAR:\n")))
print(df_low_r)
# the last 100 score rating movies in every year
df_low_s = movies.sort_values(['release_year','vote_average'], ascending=[True, True])
df_low_s = df_low_s.groupby('release_year').head(100).reset_index(drop=True)
print('\n{:s}'.format('\u0332'.join("THE LAST 100 SCORE RATING MOVIES IN EVERY YEAR:")))
print(df_low_s)


# **Question 1: What kinds of properties are associated with movies that have high popularity?**
# 
# **1.1 What's the budget level movie are associated with movies that have high popularity?**

# In[33]:


# quartile function
def cut_into_quantile(dfname ,column_name):
# find quartile, max and min values
    min_value = dfname[column_name].min()
    first_quantile = dfname[column_name].describe()[4]
    second_quantile = dfname[column_name].describe()[5]
    third_quantile = dfname[column_name].describe()[6]
    max_value = dfname[column_name].max()
# Bin edges that will be used to "cut" the data into groups
    bin_edges = [ min_value, first_quantile, second_quantile, third_quantile, max_value]
# Labels for the four budget level groups
    bin_names = [ 'Low', 'Medium', 'Moderately High', 'High'] 
# Creates budget_levels column
    name = '{}_levels'.format(column_name)
    dfname[name] = pd.cut(dfname[column_name], bin_edges, labels=bin_names, include_lowest = True)
    return dfname


# In[34]:


# Select Top 100 popular movies.
# fisrt sort it by release year ascending and popularity descending
df = movies.sort_values(['budget','popularity'], ascending=[True, False])
#group by year and choose the top 100 high
# use cut_into_quantile function to build a level column
df = cut_into_quantile(df,'budget')
df.head(1)


# In[35]:


# Find the mean and median popularity of each level with groupby
result_mean = df.groupby('budget_levels')['popularity'].mean()
result_mean 


# In[36]:


result_median = df.groupby('budget_levels')['popularity'].median()
result_median


# In[37]:


# the x locations for the groups
ind = np.arange(len(result_mean))  
# the width of the bars
width = 0.5       
ind


# In[38]:


# plot bars
#set style
sns.set_style('darkgrid')
bars = plt.bar(ind, result_mean, width, color='g', alpha=.7, label='mean')

# title and labels
plt.ylabel('popularity')
plt.xlabel('budget levels')
plt.title('Popularity with Budget Levels')
locations = ind  # xtick locations，345...
labels = result_median.index  
plt.xticks(locations, labels)
# legend
plt.legend() 


# **1.2 What's the runtime level are associated with movies that have high popularity on average?**

# In[39]:


df = cut_into_quantile(df,'runtime')
df.head(1)


# In[40]:


# Find the mean popularity of each level with groupby
result_mean = df.groupby('runtime_levels')['popularity'].mean()
result_mean


# In[41]:


# Find the median popularity of each level with groupby
result_median = df.groupby('runtime_levels')['popularity'].median()
result_median


# In[42]:


ind = np.arange(len(result_median))  # the x locations for the groups
width = 0.5       # the width of the bars


# In[43]:


# plot bars
bars = plt.bar(ind, result_median, width, color='#1ea2bc', alpha=.7, label='median')

# title and labels
plt.ylabel('popularity')
plt.xlabel('runtime levels')
plt.title('Popularity with Runtime Levels')
locations = ind  # xtick locations，345...
labels = result_median.index  
plt.xticks(locations, labels)
# legend
plt.legend() 


# **1.3 What's production countries, keywords, genres and production companies are associated with high popularity?**

# In[44]:


df_top_p.head(2)


# In[45]:


# find top three production_countries
a = find_top(df_top_p.production_countries)
# find top three keywords
c = find_top(df_top_p.keywords)
# find top three genres
d = find_top(df_top_p.genres)
# find top three production companies
e = find_top(df_top_p.production_companies)


# In[46]:


df_unpopular = pd.DataFrame({'unpopular_production_countries': a.index, 'unpopular_keywords': c.index, 'unpopular_genres': d.index, 'unpopular_producer': e.index})
df_unpopular


# # 2.1 What's the budget level are associated with movies that have high voting score?
# 
# **First, use the dataframe with budget level I have created in the previous question. Then find the mean and median of vote_average group by different budget level.**

# In[47]:


# Find the mean and median voting score of each level with groupby
result_mean = df.groupby('budget_levels')['vote_average'].mean()
result_mean


# In[48]:


# Find the mean and median voting score of each level with groupby
result_mean = df.groupby('budget_levels')['vote_average'].mean()
result_mean


# In[49]:


result_median = df.groupby('budget_levels')['vote_average'].median()
result_median


# **Let's visualize it.**

# In[50]:


# plot bars
#set style
sns.set_style('darkgrid')
ind = np.arange(len(result_mean))  # the x locations for the groups
width = 0.5       # the width of the bars

# plot bars
plt.subplots(figsize=(8, 6))
bars = plt.bar(ind, result_median, width, color='y', alpha=.7, label='mean')

# title and labels
plt.ylabel('rating')
plt.xlabel('budget levels')
plt.title('Rating with Budget Levels')
locations = ind  # xtick locations，345...
labels = result_median.index  
plt.xticks(locations, labels)
# legend
plt.legend( loc='upper left') 


# # 2.2 What's the runtime level are associated with movies that have high voting score?
# **First, use the dataframe with runtime level I have created in the previous question. Then find the mean and median of vote_average group by different runtime level.**

# In[51]:


# Find the mean popularity of each level with groupby
result_mean = df.groupby('runtime_levels')['vote_average'].mean()
result_mean


# In[52]:


result_median = df.groupby('runtime_levels')['vote_average'].median()
result_median


# **Let's visualize it.**

# In[53]:


sns.set_style('darkgrid')
ind = np.arange(len(result_mean))  # the x locations for the groups
width = 0.5       # the width of the bars

# plot bars
bars = plt.bar(ind, result_median, width, color='g', alpha=.7, label='mean')

# title and labels
plt.ylabel('rating')
plt.xlabel('runtime levels')
plt.title('Rating with Runtime Levels')
locations = ind  # xtick locations，345...
labels = result_median.index  
plt.xticks(locations, labels)
# legend
plt.legend() 


# **2.3 What's the original_language, keywords, genres are associated with voting score?**

# In[54]:


df_top_s.head(2)


# In[55]:


# find top three original_language
a = find_top(df_top_s.original_language)
# find top three keywords
b = find_top(df_top_s.keywords)
# find top three genres
c = find_top(df_top_s.genres)


# In[56]:


#create a summary dataframe.
df_high_score = pd.DataFrame({'high_score_original_language': a.index, 'high_score_keywords': b.index, 'high_score_genres': c.index})
df_high_score


# In[57]:


# call the dataset wiht the 100 low rating movies in each year
df_low_s.head(2)


# In[58]:


# find top three original_title among the among the 100 low rating movies
na = find_top(df_low_s.original_title)
# find top three keywords among the among the 100 low rating movies
nb = find_top(df_low_s.keywords)
# find top three genres among the among the 100 low rating movies
nc = find_top(df_low_s.genres)


# In[59]:


df_low_score = pd.DataFrame({'low_score_original_title': na.index, 'low_score_keywords': nb.index, 'low_score_genres': nc.index})
df_low_score


# In[60]:


# compare
df_high_score


# # Research Part 3 Top Keywords and Genres Trends by Generation
# 
# # Question 1: Number of movie released year by year

# In[61]:


movie_count = df.groupby('release_year').count()['id']
movie_count.head()


# **Then visualize the result.**

# In[62]:


#set style
sns.set_style('darkgrid')
#set x, y axis data
# x is movie release year
x = movie_count.index
# y is number of movie released
y = movie_count
#set size
plt.figure(figsize=(10, 5))
#plot line chart 
plt.plot(x, y, color = 'g', label = 'mean')
#set title and labels
plt.title('Number of Movie Released year by year')
plt.xlabel('Year')
plt.ylabel('Number of Movie Released');


# # Question 2: Genres Trends by Generation

# In[63]:


# sort the movie release year list.
dfyear= df.release_year.unique()
dfyear= np.sort(dfyear)
dfyear


# In[64]:


# year list of 1960s
y1960s =dfyear[:10]
# year list of 1970s
y1970s =dfyear[10:20]
# year list of 1980s
y1980s =dfyear[20:30]
# year list of 1990s
y1990s = dfyear[30:40]
# year list of afer 2000
y2000 = dfyear[40:]


# **Then for each generation dataframe, use the find_top to find out the most appeared keywords, then combine this result to a new datafram.**

# In[65]:


# year list of each generation
times = [y1960s, y1970s, y1980s, y1990s, y2000]
#generation name
names = ['1960s', '1970s', '1980s', '1990s', 'after2000']
#creat a empty dataframe,df_r3
df_r3 = pd.DataFrame()
index = 0
#for each generation, do the following procedure
for s in times:
    # first filter dataframe with the selected generation, and store it to dfn
    dfn = df[df.release_year.isin(s)] 
    #apply the find_top function with the selected frame, using the result create a dataframe, store it to dfn2 
    dfn2 = pd.DataFrame({'year' :names[index],'top': find_top(dfn.genres,1)})
     #append dfn2 to df_q2
    df_r3 = df_r3.append(dfn2)
    index +=1
df_r3


# **Now let's visualize the result.**

# In[66]:


# Setting the positions
generation = ['1960s', '1970s', '1980s', '1990s', 'after2000']
genres = df_r3.index
y_pos = np.arange(len(generation))
fig, ax = plt.subplots()
# Setting y1: the genre number
y1 = df_r3.top
# Setting y2 again to present the right-side y axis labels
y2 = df_r3.top
#plot the bar
ax.barh(y_pos,y1, color = '#007482')
#set the left side y axis ticks position
ax.set_yticks(y_pos)
#set the left side y axis tick label
ax.set_yticklabels(genres)
#set left side y axis label
ax.set_ylabel('genres')

#create another axis to present the right-side y axis labels
ax2 = ax.twinx()
#plot the bar
ax2.barh(y_pos,y2, color = '#27b466')
#set the right side y axis ticks position
ax2.set_yticks(y_pos)
#set the right side y axis tick label
ax2.set_yticklabels(generation)
#set right side y axis label
ax2.set_ylabel('generation')
#set title
ax.set_title('Genres Trends by Generation')


# In[67]:


# correlation map
f,ax = plt.subplots(figsize = (10,10))
sns.heatmap(movies.corr(), annot = True, linewidths=.5, fmt = '.1f', ax = ax)
plt.show()


# In[68]:


# Line plot
movies.revenue.plot(kind='line', color='r', label='revenue', linewidth=.7, alpha=.5, grid=True, linestyle='-' )
movies.budget.plot(color='g', label='budget', linewidth=.7, alpha=.8, grid=True, linestyle='-.')
plt.legend(loc='upper right')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Line Plot')
plt.show()


# In[69]:


# Scatter Plot
movies.plot(kind='scatter', x='vote_average', y='budget', alpha=.5, color='r')
plt.xlabel('vote_average')
plt.ylabel('budget')
plt.title('Scatter Plot')
plt.show()


# In[70]:


# Histogram
movies.budget.plot(kind='hist', bins = 20, figsize = (10,10))
plt.show()

