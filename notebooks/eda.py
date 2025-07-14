#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


pd.set_option('display.max_columns', None)

# 1. Load data
df = pd.read_csv("../data/student-mat.csv", sep=';')


# #### 2. Basic info

# In[5]:


print(df.info())
print(df.head())


# #### 3. Missing values

# In[7]:


print("\nMissing values:\n", df.isnull().sum())


# #### 4. Descriptive stats

# In[8]:


print("\nDescriptive statistics:\n", df.describe())


# #### 5. Distribution of target variable (G3)

# In[10]:


plt.figure(figsize=(6, 4))
sns.histplot(df['G3'], bins=21, kde=True)
plt.title("Distribution of Final Grade (G3)")
plt.xlabel("G3")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# #### 6. Categorical feature distributions

# In[11]:


categorical_features = ['school', 'sex', 'address', 'famsize', 'Pstatus',
                        'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup',
                        'famsup', 'paid', 'activities', 'nursery',
                        'higher', 'internet', 'romantic']

for col in categorical_features:
    plt.figure(figsize=(6, 3))
    sns.countplot(x=col, data=df)
    plt.xticks(rotation=45)
    plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()


# #### 7. Correlation heatmap (numeric features only)

# In[12]:


numeric_cols = df.select_dtypes(include='number')
plt.figure(figsize=(12, 10))
sns.heatmap(numeric_cols.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()


# #### 8. G1 and G2 vs G3

# In[14]:


plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='G1', y='G3')
plt.title("G1 vs G3")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='G2', y='G3')
plt.title("G2 vs G3")
plt.tight_layout()
plt.show()

