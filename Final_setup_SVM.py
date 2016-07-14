
# coding: utf-8

# This script implements a Support Vector Machine on the Titanic dataset. Because an SVM doesn't naturally take into account interactions or feature selection, and we have a large number of potential polynomial interactions, we create some interactions that feel intuitively obvious (anecdotally, age, class, and gender, and family membership all interactively played a role in who survived), include these in our features, and then use a univariate analysis, along with cross-validation, to choose the variables that likely best predict our test data
#         

# In[82]:

import numpy as np
import pandas as pd

titanic=pd.read_csv('./titanic_clean_data.csv')

cols_to_norm=['Age','Fare']
col_norms=['Age_z','Fare_z']

titanic[col_norms]=titanic[cols_to_norm].apply(lambda x: (x-x.mean())/x.std())

#titanic['cabin_clean']=(pd.notnull(titanic.Cabin))

from sklearn.cross_validation import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.ensemble import  GradientBoostingClassifier, RandomForestClassifier
from sklearn import svm, datasets, feature_selection, cross_validation
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif



titanic['Gender']=titanic['Sex'].replace(to_replace=[True,False],value=['M','F'])
titanic['Parch_ind']=titanic.Parch>=1

titanic=pd.merge(titanic, pd.get_dummies(titanic['Gender'].str.cat(titanic['Pclass'].astype(str),sep='_')),                 on=titanic['PassengerId'])
titanic=pd.merge(titanic, pd.get_dummies(titanic['Gender'].str.cat(titanic['Parch_ind'].astype(str),sep='_')),                 on=titanic['PassengerId'])
titanic=pd.merge(titanic, pd.get_dummies(titanic['Gender'].str.cat(titanic['Age_cut'].astype(str),sep='_')),                 on=titanic['PassengerId'])


# In[83]:


titanic_target=titanic.Survived.values
features=['Sex','SibSp','Parch','Pclass_1','Pclass_2','Pclass_3','Emb_C','Emb_Q','Emb_S',                         'Emb_nan','Age_ct_C','Age_ct_A','Age_ct_S', 'Sp_ct','Age_z','Fare_z',                        'Ti_Dr', 'Ti_Master', 'Ti_Mil', 'Ti_Miss', 'Ti_Mr', 'Ti_Mrs', 'Ti_Other', 'Ti_Rev',                         'Fl_AB', 'Fl_CD', 'Fl_EFG', 'Fl_nan',                         'F_1', 'F_2', 'F_3', 'M_1', 'M_2', 'M_3', 'F_False', 'F_True', 'M_False', 'M_True',                         'F_A', 'F_C', 'M_A', 'M_C', 'M_S']            
titanic_features=titanic[features].values


titanic_features, ensemble_features, titanic_target, ensemble_target=     train_test_split(titanic_features,
                     titanic_target,
                     test_size=.1,
                     random_state=7132016)


# Our key parameters here are the penalty term, and the best k features from the univariate analysis

# In[84]:

score=0

for x in range(10,43):
    for y in np.linspace(.1,.5,5):
        var_filter=SelectKBest(f_classif)
        clf=svm.SVC(kernel='rbf')
        pipe_svm = Pipeline([('anova', var_filter), ('svc', clf)])
        pipe_svm.set_params(anova__k=x, svc__C=y)
        score_test = cross_validation.cross_val_score(pipe_svm, titanic_features, titanic_target, n_jobs=1,                                                        cv=StratifiedKFold(titanic_target, n_folds=10, shuffle=True, random_state=7132016))
        if score_test.mean()>score:
            score=score_test.mean()
            k_out=x
            C_out=y
            
print k_out
print C_out
print score


# 22
# 0.5
# 0.802584778872

# In[85]:

model=pipe_svm.set_params(anova__k=k_out, svc__C=C_out).fit(titanic_features, titanic_target)


# Prep the Kaggle test data, as well as the ensembling test data

# In[86]:

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
test_data['Gender']=test_data['Sex'].replace(to_replace=[True,False],value=['M','F'])
test_data['Parch_ind']=test_data.Parch>=1
#pd.get_dummies(str.cat(titanic[['Gender','Pclass']], sep='_'))
test_data=pd.merge(test_data, pd.get_dummies(test_data['Gender'].str.cat(test_data['Pclass'].astype(str),sep='_')), on=test_data['PassengerId'])
test_data=pd.merge(test_data, pd.get_dummies(test_data['Gender'].str.cat(test_data['Parch_ind'].astype(str),sep='_')), on=test_data['PassengerId'])
test_data=pd.merge(test_data, pd.get_dummies(test_data['Gender'].str.cat(test_data['Age_cut'].astype(str),sep='_')), on=test_data['PassengerId'])


# In[87]:

name_list=pd.concat([titanic[['PassengerId','NameTest']],test_data[['PassengerId','NameTest']]])
name_list['Sp_ct']=name_list.groupby('NameTest')['NameTest'].transform('count')-1
test_data=pd.merge(test_data,name_list[['PassengerId','Sp_ct']],on='PassengerId',how='left')


# In[88]:

def add_cols(var_check,df):

    if var_check not in df.columns.values:
        df[var_check]=0

for x in features:
    add_cols(x, test_data)
    
features=['Sex','SibSp','Parch','Pclass_1','Pclass_2','Pclass_3','Emb_C','Emb_Q','Emb_S',                         'Emb_nan','Age_ct_C','Age_ct_A','Age_ct_S', 'Sp_ct','Age_z','Fare_z',                        'Ti_Dr', 'Ti_Master', 'Ti_Mil', 'Ti_Miss', 'Ti_Mr', 'Ti_Mrs', 'Ti_Other', 'Ti_Rev',                         'Fl_AB', 'Fl_CD', 'Fl_EFG', 'Fl_nan',                         'F_1', 'F_2', 'F_3', 'M_1', 'M_2', 'M_3', 'F_False', 'F_True', 'M_False', 'M_True',                         'F_A', 'F_C', 'M_A', 'M_C', 'M_S']            
test_features=test_data[features].values


# In[89]:

predictions=model.predict(ensemble_features)
ensemble_svm=pd.DataFrame({'svm_pred':predictions})
ensemble_svm.to_csv('./ensemble_svm.csv', index=False)

predictions=model.predict(test_features)
test_data['Survived']=predictions
kaggle=test_data[['PassengerId','Survived']]
kaggle.to_csv('./kaggle_titanic_submission_svm.csv', index=False)

