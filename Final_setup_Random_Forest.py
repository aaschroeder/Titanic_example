
# coding: utf-8

# This script implements a Random Forest algorithm on the Titanic dataset. This is an algorithm that bootstraps both features and samples, and so will be auto-selecting features to evaluate. Since we're pretty agnostic about everything except prediction, let's just cycle through each of the option values to find some good ones (since we're not doing any backward evaluation after choosing an option, this probably isn't the optimal one, but it should be decent at least).

# In[7]:

import numpy as np
import pandas as pd

titanic=pd.read_csv('./titanic_clean_data.csv')

cols_to_norm=['Age','Fare']
col_norms=['Age_z','Fare_z']

titanic[col_norms]=titanic[cols_to_norm].apply(lambda x: (x-x.mean())/x.std())

titanic['cabin_clean']=(pd.notnull(titanic.Cabin))

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import  GradientBoostingClassifier, RandomForestClassifier

titanic_target=titanic.Survived.values
features=['Sex','SibSp','Parch','Pclass_1','Pclass_2','Pclass_3','Emb_C','Emb_Q','Emb_S',                         'Emb_nan','Age_ct_C','Age_ct_A','Age_ct_S', 'Sp_ct','Age_z','Fare_z',                        'Ti_Dr', 'Ti_Master', 'Ti_Mil', 'Ti_Miss', 'Ti_Mr', 'Ti_Mrs', 'Ti_Other', 'Ti_Rev',                         'Fl_AB', 'Fl_CD', 'Fl_EFG', 'Fl_nan']
titanic_features=titanic[features].values


titanic_features, ensemble_features, titanic_target, ensemble_target=     train_test_split(titanic_features,
                     titanic_target,
                     test_size=.1,
                     random_state=7132016)


# Now, what we're going to do is go stepwise through many of the features of the random forest classifier, to figure out which parameters will give us the best fit. RF naturally does cross-validation in the default, so we don't need to worry about that part. We'll go in the following order:
# 
#    * Number of Trees
#    * Loss Criterion
#    * max_features
#    * max_depth
#    * min_samples_split
#    * min_weight_fraction_leaf
#    * max_leaf_nodes

# 1.) Number of Trees

# In[8]:

score=0
scores=[]
for feature in range(50,1001,50):
    clf = RandomForestClassifier(n_estimators=feature, oob_score=True, random_state=7112016)
    clf.fit(titanic_features,titanic_target)
    score_test = clf.oob_score_
    scores.append(score_test)
    if score_test>score:
        n_out=feature
        score_diff=score_test-score
        score=score_test


print n_out        


# 2.) Loss Criterion

# In[9]:

crit_param = ['gini','entropy']
score=0
for feature in crit_param:
    clf = RandomForestClassifier(n_estimators=n_out, criterion=feature, oob_score=True, random_state=7112016)
    clf.fit(titanic_features,titanic_target)
    score_test = clf.oob_score_
    if score_test>score:
        crit_out=feature
        score_diff=score_test-score
        score=score_test

print crit_out        


# 3.) Number of features considered at each split

# In[10]:

feat_param = ['sqrt','log2',None]
score=0
for feature in feat_param:
    clf = RandomForestClassifier(n_estimators=n_out, criterion=crit_out, max_features=feature,                                 oob_score=True, random_state=7112016)
    clf.fit(titanic_features,titanic_target)
    score_test = clf.oob_score_
    if score_test>score:
        max_feat_out=feature
        score_diff=score_test-score
        score=score_test

print max_feat_out        


# 4.) Maximum depth of tree

# In[11]:

score=0
for feature in range(1,21):
    clf = RandomForestClassifier(n_estimators=n_out, criterion=crit_out, max_features=max_feat_out,                                  max_depth=feature, oob_score=True, random_state=7112016)
    clf.fit(titanic_features,titanic_target)
    score_test = clf.oob_score_
    if score_test>score:
        depth_out=feature
        score_diff=score_test-score
        score=score_test

print depth_out        


# 5.) The number of samples available in order to make another split

# In[12]:

score=0
scores=[]
for feature in range(1,21):
    clf = RandomForestClassifier(n_estimators=n_out, criterion=crit_out,                                 max_features=max_feat_out, max_depth=depth_out,                                 min_samples_split=feature, oob_score=True, random_state=7112016)
    clf.fit(titanic_features,titanic_target)
    score_test = clf.oob_score_
    scores.append(score_test)
    if score_test>=score:
        sample_out=feature
        score_diff=score_test-score
        score=score_test

print sample_out        


# 6.) Min required weighted fraction of samples in a leaf or node

# In[13]:

score=0

for feature in np.linspace(0.0,0.5,10):
    clf = RandomForestClassifier(n_estimators=n_out, criterion=crit_out,                                 max_features=max_feat_out, max_depth=depth_out,                                 min_samples_split=sample_out, min_weight_fraction_leaf=feature,                                  oob_score=True, random_state=7112016)
    clf.fit(titanic_features,titanic_target)
    score_test = clf.oob_score_
    scores.append(score_test)
    if score_test>score:
        frac_out=feature
        score_diff=score_test-score
        score=score_test

print frac_out


# 7.) Maximum possible number of nodes

# In[14]:

#max_leaf_nodes - Note here we don't reset score because in order to use this variable we'll need to change other stuff

node_out=None

for feature in range(2,11):
    clf = RandomForestClassifier(n_estimators=n_out, criterion=crit_out,                                 max_features=max_feat_out, max_depth=depth_out,                                 min_samples_split=sample_out, min_weight_fraction_leaf=frac_out,                                  max_leaf_nodes=feature, oob_score=True, random_state=7112016)
    clf.fit(titanic_features,titanic_target)
    score_test = clf.oob_score_
    scores.append(score_test)
    if score_test>score:
        node_out=feature
        score_diff=score_test-score
        score=score_test

print node_out

model=RandomForestClassifier(n_estimators=n_out, criterion=crit_out,                                 max_features=max_feat_out, max_depth=depth_out,                                 min_samples_split=sample_out, min_weight_fraction_leaf=frac_out,                                  max_leaf_nodes=node_out, random_state=7112016).fit(titanic_features, titanic_target)


# In[15]:

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

test_data['cabin_clean']=(pd.notnull(test_data.Cabin))


# In[16]:

name_list=pd.concat([titanic[['PassengerId','NameTest']],test_data[['PassengerId','NameTest']]])
name_list['Sp_ct']=name_list.groupby('NameTest')['NameTest'].transform('count')-1
test_data=pd.merge(test_data,name_list[['PassengerId','Sp_ct']],on='PassengerId',how='left')


# In[17]:

def add_cols(var_check,df):

    if var_check not in df.columns.values:
        df[var_check]=0

for x in features:
    add_cols(x, test_data)
    
features=['Sex','SibSp','Parch','Pclass_1','Pclass_2','Pclass_3','Emb_C','Emb_Q','Emb_S',                         'Emb_nan','Age_ct_C','Age_ct_A','Age_ct_S', 'Sp_ct','Age_z','Fare_z',                        'Ti_Dr', 'Ti_Master', 'Ti_Mil', 'Ti_Miss', 'Ti_Mr', 'Ti_Mrs', 'Ti_Other', 'Ti_Rev',                         'Fl_AB', 'Fl_CD', 'Fl_EFG', 'Fl_nan']
test_features=test_data[features].values


# In[18]:

predictions=model.predict(ensemble_features)
ensemble_rf=pd.DataFrame({'rf_pred':predictions})
ensemble_rf.to_csv('./ensemble_rf.csv', index=False)

predictions=model.predict(test_features)
test_data['Survived']=predictions
kaggle=test_data[['PassengerId','Survived']]
kaggle.to_csv('./kaggle_titanic_submission_rf.csv', index=False)

