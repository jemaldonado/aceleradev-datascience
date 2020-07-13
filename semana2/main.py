#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[247]:


import pandas as pd
import numpy as np


# In[248]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[249]:


black_friday.describe()


# In[250]:


black_friday.shape


# In[251]:


black_friday.head()


# In[252]:


black_friday.count()


# In[253]:


## criei esse dataframe
##  para auxiliar nos calculos da questao 9
df_aux = pd.DataFrame()


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[254]:


def q1():
    return black_friday.shape
    pass


# In[255]:


q1()


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[256]:


def q2():
    return int(black_friday['Gender'].loc[ (black_friday['Gender'] == 'F') & ( black_friday['Age'] == '26-35') ].count())
    pass


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[257]:


len(black_friday['User_ID'].unique())


# In[258]:


def q3():
    return len(black_friday['User_ID'].unique())
    pass


# In[259]:


q3()


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[260]:


black_friday.dtypes.nunique()


# In[261]:


def q4():
    return black_friday.dtypes.nunique()
    pass


# In[262]:


q4()


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[263]:


def q5():
    return float(black_friday.isnull().sum().max()/black_friday.shape[0])
    pass


# In[264]:


q5()


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[265]:


#black_friday.count().idxmin()
max(black_friday.isnull().sum())


# In[266]:


def q6():
    return max(black_friday.isnull().sum())
    pass


# In[267]:


q6()


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[268]:


black_friday['Product_Category_3'].value_counts().idxmax()


# In[269]:


def q7():
    return black_friday['Product_Category_3'].value_counts().idxmax()
    pass


# In[270]:


q7()


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[271]:


#(valor-min) / (max-min)


# In[272]:


def q8():
    ret = (black_friday['Purchase'].mean() - black_friday['Purchase'].min()) / (black_friday['Purchase'].max() -black_friday['Purchase'].min())
    return float(ret) 
    pass


# In[273]:


q8()


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[274]:


def q9():
    mean = black_friday['Purchase'].mean()
    std = black_friday['Purchase'].std()
    #(x-x.mean/x.std)
    df_aux['Padronized-Purchase'] = (black_friday['Purchase'] - mean) / std
    ret = df_aux['Padronized-Purchase'] .loc[(df_aux['Padronized-Purchase'] >= -1) & (df_aux['Padronized-Purchase'] <= 1)].count()
    return int(ret)
    pass


# In[275]:


q9()


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[276]:


def q10():
    pc2 = black_friday['Product_Category_2'].isnull()
    pc3 = black_friday['Product_Category_3'].isnull()
    res = pc2.isin(pc3)
    
    return res.any().item()
q10()

