
# coding: utf-8

# In[43]:

import numpy as np
import pandas as pd

titanic=pd.read_csv('./train.csv')


# First take a look at the data types and non-null entries

# In[44]:

print titanic.info()

print titanic.describe().T

print titanic.head(5)


# Some observations from looking at the above data include:
# 
#  - Age contains missing values for a relatively small set of the population. We will need to impute numbers for this somehow.
#  - Cabin contains many, many missing values. The first character also likely contains information on the deck location
#  - Embarked contains two missing values. We likely need to account for this, but likely through a simple missing dummy/factor
#  - There is quite a bit of information in the name field that could become useful. For example, while there is a sibling/spouse    field, it may be helpful to identify/separate whether it is a sibling or a spouse onboard using the "Miss/Mrs." feature as      well as knowing that female spouses have their husbands' name in the registry, so we can find a match.
#  - Variables to turn into factors include Pclass, (potentially, given likely discontinuous breaks) Age, and Sex

# Let's start with converting the Sex values into a boolean; just feels cleaner to start there, especially if we are going to be working with this variable when cleaning other variables

# In[45]:

titanic.Sex.replace(['male','female'],[True,False], inplace=True)


# Next, let's fill in the missing Age values. We have a pretty full dataset (about 7/8ths), so we can probably get away, in a first pass, with doing some simple stratification and assuming ages are missing-at-random within those strata. For now, let's stratify on gender and class.

# In[46]:

#print titanic.Age.mean()
    
titanic.Age= titanic.groupby(['Sex','Pclass'])[['Age']].transform(lambda x: x.fillna(x.mean()))
titanic.Fare= titanic.groupby(['Pclass'])[['Fare']].transform(lambda x: x.fillna(x.mean()))
#print titanic.info()


# Next, deal with converting Pclass into something we can work with, and also create dummies for deck location and port of embarkation (when found)

# In[47]:

titanic_class=pd.get_dummies(titanic.Pclass,prefix='Pclass',dummy_na=False)
titanic=pd.merge(titanic,titanic_class,on=titanic['PassengerId'])
titanic=pd.merge(titanic,pd.get_dummies(titanic.Embarked, prefix='Emb', dummy_na=True), on=titanic['PassengerId'])

titanic['Floor']=titanic['Cabin'].str.extract('^([A-Z])', expand=False)
#T only appears once, so let's just scrub that to NaN
titanic['Floor'].replace(to_replace='T',value=np.NaN ,inplace=True)
titanic['Floor'].replace(to_replace=['A','B','C','D','E','F','G'],value=['AB','AB','CD','CD','EFG','EFG','EFG'] ,inplace=True)

titanic=pd.merge(titanic,pd.get_dummies(titanic.Floor, prefix="Fl", dummy_na=True),on=titanic['PassengerId'])


# In[48]:

titanic['Age_cut']=pd.cut(titanic['Age'],[0,14.9,54.9,99], labels=['C','A','S'])
titanic=pd.merge(titanic,pd.get_dummies(titanic.Age_cut, prefix="Age_ct", dummy_na=False),on=titanic['PassengerId'])


# Finally, before going forward I'd really like to be able to separate spouses from siblings in that variable. One way to do this is that we see married women have their husbands' names located outside of parentheses within their own name. We create a new variable that just contains the words outside of the parentheses, and see if these match any other names in the dataset. 
# 
# Additionally, from just scanning the names in the dataset, it appears that titles always come between the comma after the last name and a period. This is a perfect opportunity to use regular expressions to extract that title and turn it into a feature we can consider in analysis.

# In[49]:

import re as re


# In[50]:

titanic['Title']=titanic['Name'].str.extract(', (.*)\.', expand=False)


# So there is some cleaning that could be done here. We'll do three things:
# 1.) Turn French ladies' titles into English ones
# 2.) Aggregate Military titles
# 3.) For all remaining titles with count less than five, create remainder bin

# In[51]:

titanic['Title'].replace(to_replace='Mrs\. .*',value='Mrs', inplace=True, regex=True)
titanic.loc[titanic.Title.isin(['Col','Major','Capt']),['Title']]='Mil'
titanic.loc[titanic.Title=='Mlle',['Title']]='Miss'
titanic.loc[titanic.Title=='Mme',['Title']]='Mrs'

print titanic.Title.value_counts()


# In[52]:

titanic['Title_ct']=titanic.groupby(['Title'])['Title'].transform('count')
titanic.loc[titanic.Title_ct<5,['Title']]='Other'

titanic.Title.value_counts()

titanic=pd.merge(titanic,pd.get_dummies(titanic.Title, prefix='Ti',dummy_na=False), on=titanic['PassengerId'])

print titanic.info()


# In[53]:

titanic['NameTest']=titanic.Name
titanic['NameTest'].replace(to_replace=" \(.*\)",value="",inplace=True, regex=True)
titanic['NameTest'].replace(to_replace=", M.*\.",value=", ",inplace=True, regex=True)


# In[54]:

name_list=pd.concat([titanic[['PassengerId','NameTest']]])
name_list['Sp_ct']=name_list.groupby('NameTest')['NameTest'].transform('count')-1
titanic=pd.merge(titanic,name_list[['PassengerId','Sp_ct']],on='PassengerId',how='left')


# In[55]:

titanic.to_csv('./titanic_clean_data.csv')


# Typically, the next step here would be to perform a univariate (and, for learners that do not naturally perform feature selection with interactions, potentially bivariate) analysis to see which features best predict the outcome. In some cases (Random Forests, Boosting trees) the learner naturally performs feature selection; in others (SVM, Logit, Naive Bayes), we rely on a univariate analysis pipelined into the algorithm to make these decisions.

# In[56]:

execfile('./Final_setup_Random_Forest.py')


# In[57]:

execfile('./Final_setup_GBoost.py')


# In[60]:

execfile('./Final_setup_SVM.py')


# In[61]:

execfile('./Final_setup_Logit.py')


# In[62]:

execfile('./Final_setup_NB.py')


# In[63]:

execfile('./Final_ensemble.py')


# In[ ]:



