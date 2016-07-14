
# coding: utf-8

#  In this section, we're going to
# 
#    - Setup the titanic data for ensembling
#    - Call the five individual models
#    - Ensemble the models (We use another Gradient Booster
#    - Output for submission
#         

# In[189]:

import numpy as np
import pandas as pd

titanic=pd.read_csv('./titanic_clean_data.csv')


# OK, let's normalize our continous variables

# In[190]:

cols_to_norm=['Age','Fare']
col_norms=['Age_z','Fare_z']

titanic[col_norms]=titanic[cols_to_norm].apply(lambda x: (x-x.mean())/x.std())

titanic['cabin_clean']=(pd.notnull(titanic.Cabin))


# In[191]:

from sklearn import svm, datasets
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import StratifiedKFold, KFold, cross_val_score, train_test_split
from sklearn.ensemble import  GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# In[192]:

titanic_target=titanic.Survived.values
features=['Sex','SibSp','Parch','Pclass_1','Pclass_2','Pclass_3','Emb_C','Emb_Q','Emb_S',                         'Emb_nan','Age_ct_C','Age_ct_A','Age_ct_S', 'Sp_ct','Age_z','Fare_z',                        'Ti_Dr', 'Ti_Master', 'Ti_Mil', 'Ti_Miss', 'Ti_Mr', 'Ti_Mrs', 'Ti_Other', 'Ti_Rev',                         'Fl_AB', 'Fl_CD', 'Fl_EFG', 'Fl_nan']
titanic_features=titanic[features].values



# In[193]:

titanic_features, ensemble_features, titanic_target, ensemble_target=     train_test_split(titanic_features,
                     titanic_target,
                     test_size=.1,
                     random_state=7132016)


# In here will be the section where we import the relevant python code from each project
# 
# Now, we'll import the three csv datasets, merge them on PassengerID, then ensemble

# In[194]:

titanic_rf=pd.read_csv('./ensemble_rf.csv')
titanic_gboost=pd.read_csv('./ensemble_gboost.csv')
titanic_svm=pd.read_csv('./ensemble_svm.csv')
titanic_nb=pd.read_csv('./ensemble_nb.csv')
titanic_logit=pd.read_csv('./ensemble_logit.csv')


titanic_ensemble=pd.merge(titanic_rf, titanic_gboost, left_index=True, right_index=True)
titanic_ensemble=pd.merge(titanic_ensemble, titanic_svm, left_index=True, right_index=True)
titanic_ensemble=pd.merge(titanic_ensemble, titanic_nb, left_index=True, right_index=True)
titanic_ensemble=pd.merge(titanic_ensemble, titanic_logit, left_index=True, right_index=True)

print titanic_ensemble.head()


# Let's see what the correlations are across the learners, to find how much similarity in information they share

# In[195]:

print titanic_ensemble.corr()


# Now let's perform the ensembling, using the same method for Gradient Boosting we used earlier

# In[196]:

titanic_ensemble=titanic_ensemble.values

#clf=GradientBoostingClassifier().fit(titanic_ensemble,ensemble_target)

feat_param=['deviance','exponential']
score=0
for feature in feat_param:
    clf = GradientBoostingClassifier(loss=feature, random_state=7112016)
    score_test= cross_val_score(clf,titanic_ensemble,ensemble_target,cv=5 )
    if score_test.mean()>score:
        loss_out=feature
        score_diff=score_test.mean()-score
        score=score_test.mean()

score=0
for feature in np.linspace(.05,.45,11):
    clf = GradientBoostingClassifier(loss=loss_out, learning_rate=feature, random_state=7112016)
    score_test= cross_val_score(clf,titanic_ensemble,ensemble_target,cv=5 )
    if score_test.mean()>score:
        rate_out=feature
        score_diff=score_test.mean()-score
        score=score_test.mean()

score=0
for feature in range(100,1001,100):
    clf = GradientBoostingClassifier(loss=loss_out, learning_rate=rate_out, n_estimators=feature, random_state=7112016)
    score_test= cross_val_score(clf,titanic_ensemble,ensemble_target,cv=5)
    if score_test.mean()>score:
        feat_n_out=feature
        score_diff=score_test.mean()-score
        score=score_test.mean()
        
score=0

for feature in range(1,21):
    clf = GradientBoostingClassifier(loss=loss_out, learning_rate=rate_out, n_estimators=feat_n_out,                                     max_depth=feature, random_state=7112016)
    score_test= cross_val_score(clf,titanic_ensemble,ensemble_target,cv=5)
    if score_test.mean()>score:
        depth_out=feature
        score_diff=score_test.mean()-score
        score=score_test.mean()

score=0
for feature in range(1,21):
    clf = GradientBoostingClassifier(loss=loss_out, learning_rate=rate_out, n_estimators=feat_n_out,                                     max_depth=depth_out, min_samples_split=feature, random_state=7112016)
    score_test= cross_val_score(clf,titanic_ensemble,ensemble_target,cv=5)
    if score_test.mean()>score:
        sample_out=feature
        score_diff=score_test.mean()-score
        score=score_test.mean()

score=0
for feature in range(1,21):
    clf = GradientBoostingClassifier(loss=loss_out, learning_rate=rate_out, n_estimators=feat_n_out,                                     max_depth=depth_out, min_samples_split=sample_out,                                     min_samples_leaf=feature, random_state=7112016)
    score_test= cross_val_score(clf,titanic_ensemble,ensemble_target,cv=5)
    if score_test.mean()>score:
        sample_leaf_out=feature
        score_diff=score_test.mean()-score
        score=score_test.mean()

score=0

for feature in np.linspace(0.0,0.5,10):
    clf = GradientBoostingClassifier(loss=loss_out, learning_rate=rate_out, n_estimators=feat_n_out,                                     max_depth=depth_out, min_samples_split=sample_out,                                     min_samples_leaf=sample_leaf_out, min_weight_fraction_leaf=feature, random_state=7112016)
    score_test= cross_val_score(clf,titanic_ensemble,ensemble_target,cv=5 )
    if score_test.mean()>score:
        frac_out=feature
        score_diff=score_test.mean()-score
        score=score_test.mean()

score=0

for feature in np.linspace(0.1,1,10):
    clf = GradientBoostingClassifier(loss=loss_out, learning_rate=rate_out, n_estimators=feat_n_out,                                     max_depth=depth_out, min_samples_split=sample_out,                                     min_samples_leaf=sample_leaf_out, min_weight_fraction_leaf=frac_out,                                     subsample=feature, random_state=7112016)
    score_test= cross_val_score(clf,titanic_ensemble,ensemble_target,cv=5)
    if score_test.mean()>score:
        subsamp_out=feature
        score_diff=score_test.mean()-score
        score=score_test.mean()

node_out=None

for feature in range(2,11):
    clf = GradientBoostingClassifier(loss=loss_out, learning_rate=rate_out, n_estimators=feat_n_out,                                     max_depth=depth_out, min_samples_split=sample_out,                                     min_samples_leaf=sample_leaf_out, min_weight_fraction_leaf=frac_out,                                     subsample=subsamp_out, max_leaf_nodes=feature, random_state=7112016)
    score_test= cross_val_score(clf,titanic_ensemble,ensemble_target,cv=5)
    if score_test.mean()>score:
        node_out=feature
        score_diff=score_test.mean()-score
        score=score_test.mean()

        
clf=GradientBoostingClassifier(loss=loss_out, learning_rate=rate_out, n_estimators=feat_n_out,                                     max_depth=depth_out, min_samples_split=sample_out,                                     min_samples_leaf=sample_leaf_out, min_weight_fraction_leaf=frac_out,                                     subsample=subsamp_out, max_leaf_nodes=node_out,                                     random_state=7112016).fit(titanic_ensemble,ensemble_target)


# In[197]:

titanic_rf=pd.read_csv('./kaggle_titanic_submission_rf.csv')
titanic_gboost=pd.read_csv('./kaggle_titanic_submission_gboost.csv')
titanic_svm=pd.read_csv('./kaggle_titanic_submission_svm.csv')
titanic_nb=pd.read_csv('./kaggle_titanic_submission_nb.csv')
titanic_logit=pd.read_csv('./kaggle_titanic_submission_logit.csv')


titanic_ensemble=pd.merge(titanic_rf, titanic_gboost, on='PassengerId')
titanic_ensemble.rename(columns={'Survived_x':'rf_pred','Survived_y':'gboost_pred'}, inplace=True)

titanic_ensemble=pd.merge(titanic_ensemble, titanic_svm, on='PassengerId')
titanic_ensemble.rename(columns={'Survived':'svm_pred'}, inplace=True)

titanic_ensemble=pd.merge(titanic_ensemble, titanic_nb, on='PassengerId')
titanic_ensemble.rename(columns={'Survived':'nb_pred'}, inplace=True)

titanic_ensemble=pd.merge(titanic_ensemble, titanic_logit, on='PassengerId')
titanic_ensemble.rename(columns={'Survived':'log_pred'}, inplace=True)

titanic_ensemble=titanic_ensemble[['rf_pred','gboost_pred','svm_pred', 'nb_pred','log_pred']]
titanic_ensemble=titanic_ensemble.values

predictions=clf.predict(titanic_ensemble)

kaggle=pd.DataFrame({'PassengerId':titanic_rf['PassengerId']})
kaggle['Survived']=predictions
kaggle.to_csv('./kaggle_titanic_submission_ensemble.csv', index=False)



# In[ ]:



