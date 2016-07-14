
# coding: utf-8

# This script implements a Naive Bayes estimator on the Titanic dataset. I'll be honest, I've never really used this much in my own analysis, but it's pretty easy to implement, it's a different algorithm than the other learners, and if it's consistently wrong then that should get picked up by the ensembling test dataset.

# In[46]:

import numpy as np
import pandas as pd

titanic=pd.read_csv('./titanic_clean_data.csv')

cols_to_norm=['Age','Fare']
col_norms=['Age_z','Fare_z']

titanic[col_norms]=titanic[cols_to_norm].apply(lambda x: (x-x.mean())/x.std())

titanic['cabin_clean']=(pd.notnull(titanic.Cabin))

from sklearn import svm, datasets, feature_selection, cross_validation
from sklearn.cross_validation import train_test_split,  KFold, StratifiedKFold, cross_val_score
from sklearn.ensemble import  GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import GaussianNB

titanic_target=titanic.Survived.values
features=['Sex','SibSp','Parch','Pclass_1','Pclass_2','Pclass_3','Emb_C','Emb_Q','Emb_S',                         'Emb_nan','Age_ct_C','Age_ct_A','Age_ct_S', 'Sp_ct','Age_z','Fare_z',                        'Ti_Dr', 'Ti_Master', 'Ti_Mil', 'Ti_Miss', 'Ti_Mr', 'Ti_Mrs', 'Ti_Other', 'Ti_Rev',                         'Fl_AB', 'Fl_CD', 'Fl_EFG', 'Fl_nan']
titanic_features=titanic[features].values


titanic_features, ensemble_features, titanic_target, ensemble_target=     train_test_split(titanic_features,
                     titanic_target,
                     test_size=.1,
                     random_state=7132016)


# In[47]:


score=0

for x in range(10,29):
        var_filter=SelectKBest(f_classif)
        clf=GaussianNB()
        pipe_svm = Pipeline([('anova', var_filter), ('NB', clf)])
        pipe_svm.set_params(anova__k=x)
        score_test = cross_validation.cross_val_score(pipe_svm, titanic_features, titanic_target, n_jobs=1,                                                        cv=StratifiedKFold(titanic_target, n_folds=10, shuffle=True, random_state=7132016))
        if score_test.mean()>score:
            score=score_test.mean()
            k_out=x
 
   
    
model=pipe_svm.set_params(anova__k=k_out).fit(titanic_features, titanic_target)


# In[ ]:




# In[48]:

test_data=pd.read_csv('./test.csv')

test_data.Sex.replace(['male','female'],[True,False], inplace=True)
test_data.Age= test_data.groupby(['Sex','Pclass'])[['Age']].transform(lambda x: x.fillna(x.mean()))
test_data.Fare= titanic.groupby(['Pclass'])[['Fare']].transform(lambda x: x.fillna(x.mean()))
titanic_class=pd.get_dummies(test_data.Pclass,prefix='Pclass',dummy_na=False)
test_data=pd.merge(test_data,titanic_class,on=test_data['PassengerId'])
test_data=pd.merge(test_data,pd.get_dummies(test_data.Embarked, prefix='Emb', dummy_na=True), on=test_data['PassengerId'])
titanic['Floor']=titanic['Cabin'].str.extract('^([A-Z])', expand=False)
titanic['Floor'].replace(to_replace='T',value=np.NaN ,inplace=True)
titanic=pd.merge(titanic,pd.get_dummies(titanic.Floor, prefix="Fl", dummy_na=True),on=titanic['PassengerId'])
test_data['Age_cut']=pd.cut(test_data['Age'],[0,17.9,64.9,99], labels=['C','A','S'])
test_data=pd.merge(test_data,pd.get_dummies(test_data.Age_cut, prefix="Age_ct", dummy_na=False),on=test_data['PassengerId'])

test_data['Title']=test_data['Name'].str.extract(', (.*)\.', expand=False)
test_data['Title'].replace(to_replace='Mrs\. .*',value='Mrs', inplace=True, regex=True)
test_data.loc[test_data.Title.isin(['Col','Major','Capt']),['Title']]='Mil'
test_data.loc[test_data.Title=='Mlle',['Title']]='Miss'
test_data.loc[test_data.Title=='Mme',['Title']]='Mrs'
test_data['Title_ct']=test_data.groupby(['Title'])['Title'].transform('count')
test_data.loc[test_data.Title_ct<5,['Title']]='Other'
test_data=pd.merge(test_data,pd.get_dummies(test_data.Title, prefix='Ti',dummy_na=False), on=test_data['PassengerId'])

test_data['NameTest']=test_data.Name
test_data['NameTest'].replace(to_replace=" \(.*\)",value="",inplace=True, regex=True)
test_data['NameTest'].replace(to_replace=", M.*\.",value=", ",inplace=True, regex=True)


cols_to_norm=['Age','Fare']
col_norms=['Age_z','Fare_z']

test_data['Age_z']=(test_data.Age-titanic.Age.mean())/titanic.Age.std()
test_data['Fare_z']=(test_data.Fare-titanic.Fare.mean())/titanic.Fare.std()

#test_data[col_norms]=test_data[cols_to_norm].apply(lambda x: (x-titanic.x.mean())/titanic.x.std())

test_data['cabin_clean']=(pd.notnull(test_data.Cabin))


# In[49]:

name_list=pd.concat([titanic[['PassengerId','NameTest']],test_data[['PassengerId','NameTest']]])
name_list['Sp_ct']=name_list.groupby('NameTest')['NameTest'].transform('count')-1
test_data=pd.merge(test_data,name_list[['PassengerId','Sp_ct']],on='PassengerId',how='left')


# In[50]:

def add_cols(var_check,df):

    if var_check not in df.columns.values:
        df[var_check]=0

for x in features:
    add_cols(x, test_data)
    
features=['Sex','SibSp','Parch','Pclass_1','Pclass_2','Pclass_3','Emb_C','Emb_Q','Emb_S',                         'Emb_nan','Age_ct_C','Age_ct_A','Age_ct_S', 'Sp_ct','Age_z','Fare_z',                        'Ti_Dr', 'Ti_Master', 'Ti_Mil', 'Ti_Miss', 'Ti_Mr', 'Ti_Mrs', 'Ti_Other', 'Ti_Rev',                         'Fl_AB', 'Fl_CD', 'Fl_EFG', 'Fl_nan']
test_features=test_data[features].values


# In[51]:

predictions=model.predict(ensemble_features)
ensemble_nb=pd.DataFrame({'nb_pred':predictions})
ensemble_nb.to_csv('./ensemble_nb.csv', index=False)

predictions=model.predict(test_features)
test_data['Survived']=predictions
kaggle=test_data[['PassengerId','Survived']]
kaggle.to_csv('./kaggle_titanic_submission_nb.csv', index=False)

