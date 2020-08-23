#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn.preprocessing import KBinsDiscretizer


# In[2]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

#from IPython.core.pylabtools import figsize


#figsize(12, 8)

#sns.set()


# In[3]:


countries = pd.read_csv("countries.csv")


# In[4]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[5]:


# Sua análise começa aqui.
countries.info()


# In[6]:


cols = ["Pop_density","Coastline_ratio","Net_migration","Infant_mortality","Literacy","Phones_per_1000","Arable"
       ,"Crops","Other","Birthrate","Deathrate","Agriculture","Industry","Service","Climate"]

countries[cols] = countries[cols].apply(lambda x: x.str.replace(',','.')).astype(float)

cols_with_space = ["Country","Region"]

countries[cols_with_space] = countries[cols_with_space].apply(lambda x: x.str.strip())


# In[7]:


countries.head()


# In[8]:


countries.info()


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[9]:


def q1():
        
    return sorted(countries['Region'].unique())
q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[10]:


def q2():
    dis = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    countries['Categoric_Pop_Density'] = dis.fit_transform(countries[['Pop_density']])
    percentil = countries['Categoric_Pop_Density'].quantile(0.9)
    qtd = countries.query('Categoric_Pop_Density > @percentil')
    return qtd.Country.nunique()
q2()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[11]:


from sklearn.preprocessing import (
    OneHotEncoder, Binarizer, KBinsDiscretizer,
    MinMaxScaler, StandardScaler, PolynomialFeatures
)


def q3():
    one_hot_encoder = OneHotEncoder(sparse=False, dtype=np.int)
    reg = one_hot_encoder.fit_transform(countries[["Region"]])
    
    cli = one_hot_encoder.fit_transform(countries[["Climate"]].astype(str))
    
    return reg.shape[1] + cli.shape[1]

q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[12]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[13]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

numeric_cols = new_column_names[2:]

ppl_median_std = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("standard", StandardScaler())
])



def q4():
    ppl_median_std.fit(countries[numeric_cols])
    col_arable = countries[numeric_cols].columns == 'Arable'
    country_t_transform = ppl_median_std.transform([test_country[2:]])
    return round(float(country_t_transform[0,col_arable]),3)

q4()


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[14]:


def q5():
    qtl1 = countries.Net_migration.quantile(0.25)
    qtl3 = countries.Net_migration.quantile(0.75)
    iqr  = qtl3 - qtl1

    outliers_inf = countries.Net_migration[countries.Net_migration < (qtl1 - 1.5 * iqr)]
    outliers_sup = countries.Net_migration[countries.Net_migration > (qtl3 + 1.5 * iqr)]
    return (int(outliers_inf.count()), int(outliers_sup.count()), False)

q5()


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[15]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)

def q6():
    count_vect      = CountVectorizer()
    newsgroup_count = count_vect.fit_transform(newsgroup.data)
    idx_phone       = count_vect.vocabulary_.get('phone')
    phone_total     = newsgroup_count[:,idx_phone].toarray().sum()
    return int(phone_total)

q6()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[16]:



def q7():
 # Retorne aqui o resultado da questão 4.
    count_tfidf     = TfidfVectorizer()
    newsgroup_tfidf = count_tfidf.fit_transform(newsgroup.data)
    idx_phone       = count_tfidf.vocabulary_.get('phone')
    phone_tfidf     = newsgroup_tfidf[:,idx_phone].toarray().sum()
    return float(phone_tfidf.round(3))


q7()


# In[ ]:




