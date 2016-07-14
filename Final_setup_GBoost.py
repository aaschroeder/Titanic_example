
# coding: utf-8

# This script implements a Gradient Boosting Machine on the Titanic dataset. This is a boosting algorithm that will be using trees, and will be auto-selecting features to evaluate. Since we're pretty agnostic about everything except prediction, let's just cycle through each of the option values to find some good ones (since we're not doing any backward evaluation after choosing an option, this probably isn't the optimal one, but it should be decent at least).
#         

# In[15]:

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



# Now, what we're going to do is go stepwise through many of the features of the random forest classifier, to figure out which parameters will give us the best fit. We'll go in the following order:
# 
#    * loss
#    * learning_rate
#    * n_estimators
#    * max_depth (depth of tree)
#    * min_samples_split (number of samples you need to be able to create a new branch/node)
#    * min_samples_leaf
#    * min_weight_fraction_leaf
#    * subsample
#    * max_features (number of features considered at each split)
#    * max_leaf_nodes (implicitly, if the best is not none, then we'll be ignoring the max depth parameter
#    * warm_start

# 1.) Loss Criteria

# In[16]:

feat_param=['deviance','exponential']
score=0
for feature in feat_param:
    clf = GradientBoostingClassifier(loss=feature, random_state=7112016)
    score_test= cross_val_score(clf,titanic_features,titanic_target,cv=10 )
    if score_test.mean()>score:
        loss_out=feature
        score_diff=score_test.mean()-score
        score=score_test.mean()

        
print loss_out        


# 2.) Learning Rate

# In[17]:

score=0
for feature in np.linspace(.05,.45,11):
    clf = GradientBoostingClassifier(loss=loss_out, learning_rate=feature, random_state=7112016)
    score_test= cross_val_score(clf,titanic_features,titanic_target,cv=10 )
    if score_test.mean()>score:
        rate_out=feature
        score_diff=score_test.mean()-score
        score=score_test.mean()

print rate_out


# 3.) Count of Estimators

# In[18]:

score=0
for feature in range(100,1001,100):
    clf = GradientBoostingClassifier(loss=loss_out, learning_rate=rate_out, n_estimators=feature, random_state=7112016)
    score_test= cross_val_score(clf,titanic_features,titanic_target,cv=10 )
    if score_test.mean()>score:
        feat_n_out=feature
        score_diff=score_test.mean()-score
        score=score_test.mean()

print feat_n_out


# 4.) Maximum depth of tree

# In[19]:

score=0
for feature in range(1,21):
    clf = GradientBoostingClassifier(loss=loss_out, learning_rate=rate_out, n_estimators=feat_n_out,                                     max_depth=feature, random_state=7112016)
    score_test= cross_val_score(clf,titanic_features,titanic_target,cv=10 )
    if score_test.mean()>score:
        depth_out=feature
        score_diff=score_test.mean()-score
        score=score_test.mean()

print depth_out        


# 5.) The number of samples available in order to make another split

# In[20]:

score=0
for feature in range(1,21):
    clf = GradientBoostingClassifier(loss=loss_out, learning_rate=rate_out, n_estimators=feat_n_out,                                     max_depth=depth_out, min_samples_split=feature, random_state=7112016)
    score_test= cross_val_score(clf,titanic_features,titanic_target,cv=10 )
    if score_test.mean()>score:
        sample_out=feature
        score_diff=score_test.mean()-score
        score=score_test.mean()

print sample_out


# 6.) The number of samples that must appear in a leaf

# In[21]:

score=0
for feature in range(1,21):
    clf = GradientBoostingClassifier(loss=loss_out, learning_rate=rate_out, n_estimators=feat_n_out,                                     max_depth=depth_out, min_samples_split=sample_out,                                     min_samples_leaf=feature, random_state=7112016)
    score_test= cross_val_score(clf,titanic_features,titanic_target,cv=10 )
    if score_test.mean()>score:
        sample_leaf_out=feature
        score_diff=score_test.mean()-score
        score=score_test.mean()

print sample_leaf_out        


# 7.) Min required weighted fraction of samples in a leaf or node

# In[22]:

score=0

for feature in np.linspace(0.0,0.5,10):
    clf = GradientBoostingClassifier(loss=loss_out, learning_rate=rate_out, n_estimators=feat_n_out,                                     max_depth=depth_out, min_samples_split=sample_out,                                     min_samples_leaf=sample_leaf_out, min_weight_fraction_leaf=feature, random_state=7112016)
    score_test= cross_val_score(clf,titanic_features,titanic_target,cv=10 )
    if score_test.mean()>score:
        frac_out=feature
        score_diff=score_test.mean()-score
        score=score_test.mean()

print frac_out




# 8.) Fraction of samples used in each base learner

# In[23]:

score=0

for feature in np.linspace(0.1,1,10):
    clf = GradientBoostingClassifier(loss=loss_out, learning_rate=rate_out, n_estimators=feat_n_out,                                     max_depth=depth_out, min_samples_split=sample_out,                                     min_samples_leaf=sample_leaf_out, min_weight_fraction_leaf=frac_out,                                     subsample=feature, random_state=7112016)
    score_test= cross_val_score(clf,titanic_features,titanic_target,cv=10 )
    if score_test.mean()>score:
        subsamp_out=feature
        score_diff=score_test.mean()-score
        score=score_test.mean()

print subsamp_out



# 9.) Maximum possible number of nodes

# In[24]:

node_out=None

for feature in range(2,11):
    clf = GradientBoostingClassifier(loss=loss_out, learning_rate=rate_out, n_estimators=feat_n_out,                                     max_depth=depth_out, min_samples_split=sample_out,                                     min_samples_leaf=sample_leaf_out, min_weight_fraction_leaf=frac_out,                                     subsample=subsamp_out, max_leaf_nodes=feature, random_state=7112016)
    score_test= cross_val_score(clf,titanic_features,titanic_target,cv=10 )
    if score_test.mean()>score:
        node_out=feature
        score_diff=score_test.mean()-score
        score=score_test.mean()

print node_out




# In[25]:

model=GradientBoostingClassifier(loss=loss_out, learning_rate=rate_out, n_estimators=feat_n_out,                                     max_depth=depth_out, min_samples_split=sample_out,                                     min_samples_leaf=sample_leaf_out, min_weight_fraction_leaf=frac_out,                                     subsample=subsamp_out, max_leaf_nodes=node_out,                                     random_state=7112016).fit(titanic_features, titanic_target)


# In[26]:

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


# In[27]:

name_list=pd.concat([titanic[['PassengerId','NameTest']],test_data[['PassengerId','NameTest']]])
name_list['Sp_ct']=name_list.groupby('NameTest')['NameTest'].transform('count')-1
test_data=pd.merge(test_data,name_list[['PassengerId','Sp_ct']],on='PassengerId',how='left')


# In[28]:

def add_cols(var_check,df):

    if var_check not in df.columns.values:
        df[var_check]=0

for x in features:
    add_cols(x, test_data)
    
features=['Sex','SibSp','Parch','Pclass_1','Pclass_2','Pclass_3','Emb_C','Emb_Q','Emb_S',                         'Emb_nan','Age_ct_C','Age_ct_A','Age_ct_S', 'Sp_ct','Age_z','Fare_z',                        'Ti_Dr', 'Ti_Master', 'Ti_Mil', 'Ti_Miss', 'Ti_Mr', 'Ti_Mrs', 'Ti_Other', 'Ti_Rev',                         'Fl_AB', 'Fl_CD', 'Fl_EFG', 'Fl_nan']
test_features=test_data[features].values


# In[29]:

predictions=model.predict(ensemble_features)
ensemble_gboost=pd.DataFrame({'gboost_pred':predictions})
ensemble_gboost.to_csv('./ensemble_gboost.csv', index=False)

predictions=model.predict(test_features)
test_data['Survived']=predictions
kaggle=test_data[['PassengerId','Survived']]
kaggle.to_csv('./kaggle_titanic_submission_gboost.csv', index=False)

