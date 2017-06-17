# -*- coding: utf-8 -*-
"""
Created on Wed May 17 12:32:06 2017

@author: caesarliu
"""

import matplotlib.pyplot as plt
import pandas as pd
features_to_remove=[ 'new_cac_uec', 
                    'new_eht_uec'
 ,'new_auxht_uec'
 ,'new_wht_uec'
 ,'new_whts_uec'
 ,'new_rac_uec'
 ,'new_swp_uec'
 ,'new_olt_uec'
 ,'new_ctv_uec'
 ,'new_off_uec'
 ,'new_pcs_uec'
 ,'new_pmp_uec'
 ,'new_wpm_uec'
 ,'new_ref1_uec'
 ,'new_ref2_uec'
 ,'new_fz_uec'
 ,'new_ehp_uec'
 ,'new_vent1_uec'
 ,'new_vent2_uec'
 ,'new_msc_uec'
 ,'new_rng_uec'
 ,'new_mwv_uec'
 ,'new_dwh_uec'
 ,'new_cws_uec'
 ,'new_edy_uec'
 ,'new_spa_uec'
 ,'new_sph_uec'
 ,'new_ght_uec'
 ,'new_gauxht_uec'
 ,'new_gwh_uec'
 ,'new_gswh_uec'
 ,'new_grng_uec'
 ,'new_gdry_uec'
 ,'new_gpht_uec'
 ,'new_gspa_uec'
 ,'new_gmiss_uec'
 ,'new_all_uec'
 ,'new_gall_uec'
 ,'elemn12'
 ,'thmmn12'
 ,'fzusage1'
 ,'rfusage1'
 ,'rfusage2'
 ,'thmcda'
 ,'thmmncda'
 ,'msthmcda'
 ,'ele02sum'
 ,'mselesum'
 ,'thm02sum'
 ,'msthmsum'
 ,'elecda'
 ,'elemncda'
 ,'mselecda'
 ,'hdd65'
 ,'cdd65'
 ,'ght_uec'
 ,'gwh_uec'
 ,'gswh_uec'
 ,'gdry_uec'
 ,'grng_uec'
 ,'gpht_uec'
 ,'gspa_uec'
 ,'ghh_uec'
 ,'eht_uec'
 ,'ehp_uec'
 ,'vnt1_uec'
 ,'vnt2_uec'
 ,'cac_uec'
 ,'rac_uec'
 ,'swp_uec'
 ,'wht_uec'
 ,'whts_uec'
 ,'edy_uec'
 ,'cws_uec'
 ,'dwh_uec'
 ,'ref1_uec'
 ,'ref2_uec'
 ,'fz_uec'
 ,'pmp_uec'
 ,'spa_uec'
 ,'olt_uec'
 ,'rng_uec'
 ,'ctv_uec'
 ,'sph_uec'
 ,'mwv_uec'
 ,'off_uec'
 ,'pcs_uec'
 ,'wbh_uec'
 ,'wpm_uec'
 ,'msc_uec'
 ,'hh_uec']

raw_data=pd.read_csv('project.csv',low_memory=False)
raw_data=raw_data.drop(features_to_remove,axis=1)# Remove features(comment this line out if do not want to remove above features)
raw_data=raw_data.sample(frac=1)# random shuffle the data

## Data Preprocessing
features=raw_data.columns.values.tolist()

def NA_Columns_reducer(features,dataframe,threshold):
    drop=[]
    for i in range(len(features)):
        count_NA=sum(dataframe[features[i]].isnull())
        if count_NA>threshold:
            drop.append(features[i])
    data=dataframe.drop(drop, axis=1)
    return data
  
Reduced_data=NA_Columns_reducer(features,raw_data, 10408*0.05)# Drop columns which has more than 5% observations be NA

imputed_ann=Reduced_data['annkwh']
imputed_annth=Reduced_data['anntherm']  

target_kwh=Reduced_data['kwh_frommonthly']
target_ng=Reduced_data['ng_frommonthly']

# Create Histograms to compare actual and imputed use
target_kwh.hist(bins=30,ec='black')
plt.savefig('hist_kwh.jpg')

target_ng.hist(bins=30,ec='black',color='red')
plt.savefig('hist_ng.jpg')


imputed_ann.hist(bins=30,ec='black')
plt.xlim(0,30000)
plt.savefig('hist_akwh.jpg')

imputed_annth.hist(bins=30,ec='black',color='red')
plt.savefig('hist_annt.jpg')


train_data=Reduced_data.drop(['annkwh','anntherm','kwh_frommonthly','ng_frommonthly'],axis=1)

# Seperate Numerical features and categorical features
cols=train_data.columns
num_cols=train_data._get_numeric_data().columns
train_num=train_data._get_numeric_data()
train_cat=train_data.drop(num_cols,axis=1)
train_cat_use=train_cat['county']

# Impute missing observations for continuous features
from sklearn.preprocessing import Imputer
imputer=Imputer(strategy='median')
imputer.fit(train_num)
Train_num=imputer.transform(train_num)


# Features scaling for numerical features(optional in case of random forest)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(Train_num)
Scaled_Train=scaler.transform(Train_num)


# One-hot encodes Categorical features
from sklearn.preprocessing import LabelBinarizer
encoder=LabelBinarizer()
encoder.fit(train_cat_use)
Train_cat=encoder.transform(train_cat_use)

# Obatin the final cleaned up data set
import numpy as np
Data=np.c_[Scaled_Train,Train_cat]


## Training and fine tuning
'''
kwh_frommonthly
'''
# Ttrain test split
from sklearn.model_selection import train_test_split
# Train a baseline Randomforest to predict kwh_frommonthly
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
def display_scores(scores):
    print('Scores:', scores)
    print('Mean:', scores.mean())
    print('Standard deviation:', scores.std())
    
forest_reg_kwh=RandomForestRegressor(n_jobs=-1)

forest_kwh_scores=cross_val_score(forest_reg_kwh,Data,target_kwh,scoring='neg_mean_squared_error',cv=5)

kwh_scores= -(forest_kwh_scores)

display_scores(kwh_scores)

# Tuneing the baseline random forest for kwh_frommonthly with randomized search
from sklearn.model_selection import RandomizedSearchCV
forest_reg_kwh=RandomForestRegressor(n_estimators=100,n_jobs=-1)
param_distributions={'max_features':[100,120,150,170,200,250,300,350],'max_depth':[24,25,26,27,28,29,30],
                     'min_samples_leaf':[5,6,7,8,9,10]}

random_search=RandomizedSearchCV(forest_reg_kwh,param_distributions,n_iter=100,cv=5)

random_search.fit(Data,target_kwh)
# Train and test the tuned random forest

forest_reg_kwh=RandomForestRegressor(max_depth=29,max_features=350,n_estimators=300,min_samples_leaf=5,n_jobs=-1)
forest_kwh_scores=cross_val_score(forest_reg_kwh,Data,target_kwh,scoring='neg_mean_squared_error',cv=5)

kwh_scores= -(forest_kwh_scores)

display_scores(kwh_scores)
'''
Scores: [ 4288602.27463903  4683398.40878997  4342491.56198323  4267412.92395219
  4172777.25505146]
Mean: 4350936.48488
Standard deviation: 175045.535643
'''
# Gradient boosted decision trees for kwh_frommonthly

from sklearn.ensemble import GradientBoostingRegressor

grad_reg_kwh=GradientBoostingRegressor(learning_rate=0.1)

grad_kwh_scores=cross_val_score(grad_reg_kwh,Data,target_kwh,scoring='neg_mean_squared_error',cv=5)

kwh_scores= -(grad_kwh_scores)

display_scores(kwh_scores)



# Same randomized search can be employed to fine tune the parameters of gradient boosted trees
grad_reg_kwh=GradientBoostingRegressor(learning_rate=0.1)
param_distributions={'max_features':[100,120,150,170,200,250,300,350],'max_depth':[5,10,11,12,13],
                     'min_samples_leaf':[5,6,7,8,9,10],'n_estimators':[30,50,70,75,80,90,100,120,150]}

random_search=RandomizedSearchCV(grad_reg_kwh,param_distributions,n_iter=100,cv=5)

random_search.fit(Data,target_kwh)

# Train and test the tuned gradientboosted decision trees
grad_reg_kwh=GradientBoostingRegressor(max_depth=10,max_features=100,learning_rate=0.1,n_estimators=75,min_samples_leaf=5)

grad_kwh_scores=cross_val_score(grad_reg_kwh,Data,target_kwh,scoring='neg_mean_squared_error',cv=5)

kwh_scores= -(grad_kwh_scores)

display_scores(kwh_scores)
'''
Scores: [ 3856538.60639622  4506151.98933914  4049051.58135624  3921770.28812903
  4067138.76757834]
Mean: 4080130.24656
Standard deviation: 227024.353435
Note: socre used is MSE 

'''

from sklearn.ensemble import BaggingRegressor

bag_reg_kwh=BaggingRegressor(grad_reg_kwh,n_estimators=10)
bag_kwh_scores=cross_val_score(bag_reg_kwh,Data,target_kwh,scoring='neg_mean_squared_error',cv=5)

kwh_scores= -(bag_kwh_scores)

display_scores(kwh_scores)

'''
Scores: [ 3731000.77429245  4353247.75625897  3930043.35468009  3887136.49341325
  3901624.58215534]
Mean: 3960610.59216
Standard deviation: 208181.055652
Note: socre used is MSE 
'''
# Fit the final bagged GBTs and make predictions on 20 percent randomly selected data points in the dataset

X_train,X_val,y_train,y_val=train_test_split(Data,target_kwh,test_size=0.2)

grad_reg_kwh=GradientBoostingRegressor(max_depth=10,max_features=100,learning_rate=0.1,n_estimators=75,min_samples_leaf=5)

bag_reg_kwh=BaggingRegressor(grad_reg_kwh,n_estimators=10)

bag_reg_kwh.fit(X_train,y_train)

kwh_predic=bag_reg_kwh.predict(X_val)

plt.figure()
plt.hist(y_val,bins=30,ec='black')
plt.savefig('kwh_target.jpg')

plt.figure()
plt.hist(kwh_predic,bins=30,ec='black')
plt.savefig('kwh_predict.jpg')


'''
ng_grommonthly
'''

# Train a base randomforest model
forest_reg_ng=RandomForestRegressor(n_jobs=-1)

forest_ng_scores=cross_val_score(forest_reg_ng,Data,target_ng,scoring='neg_mean_squared_error',cv=5)

ng_scores= -(forest_ng_scores)

display_scores(ng_scores)

# Randomized search to fine tune
forest_reg_ng=RandomForestRegressor(n_estimators=100,n_jobs=-1)
param_distributions={'max_features':[100,120,150,170,200,250,300,350],'max_depth':[24,25,26,27,28,29,30],
                     'min_samples_leaf':[5,6,7,8,9,10]}

random_search=RandomizedSearchCV(forest_reg_ng,param_distributions,n_iter=35,cv=5)

random_search.fit(Data,target_ng)
# Train and test the tuned random forest

forest_reg_ng=RandomForestRegressor(max_depth=28,max_features=300,n_estimators=300,min_samples_leaf=5,n_jobs=-1)
forest_ng_scores=cross_val_score(forest_reg_ng,Data,target_ng,scoring='neg_mean_squared_error',cv=5)

ng_scores= -(forest_ng_scores)

display_scores(ng_scores)


'''
Scores: [ 47191.58557426  73852.11896503  41568.02359935  44987.25424046
  59896.7284854 ]
Mean: 53499.1421729
Standard deviation: 11914.3767925
'''

# Gradient boosted decision trees for ng_frommonthly


grad_reg_ng=GradientBoostingRegressor(learning_rate=0.1)

grad_ng_scores=cross_val_score(grad_reg_ng,Data,target_ng,scoring='neg_mean_squared_error',cv=5)

ng_scores= -(grad_ng_scores)

display_scores(ng_scores)



# Same randomized search can be employed to fine tune the parameters of gradient boosted trees
grad_reg_ng=GradientBoostingRegressor(learning_rate=0.1)
param_distributions={'max_features':[100,120,150,170,200,250,300,350],'max_depth':[5,10,11,12,13],
                     'min_samples_leaf':[5,6,7,8,9,10],'n_estimators':[30,50,70,75,80,90,100,120,150]}

random_search=RandomizedSearchCV(grad_reg_ng,param_distributions,n_iter=35,cv=5)

random_search.fit(Data,target_ng)

# Train and test the tuned gradientboosted decision trees
grad_reg_ng=GradientBoostingRegressor(max_depth=5,max_features=170,learning_rate=0.1,n_estimators=150,min_samples_leaf=8)

grad_ng_scores=cross_val_score(grad_reg_ng,Data,target_ng,scoring='neg_mean_squared_error',cv=5)

ng_scores= -(grad_ng_scores)

display_scores(ng_scores)
'''
Scores: [ 45847.79850864  70093.65229616  38725.36005197  41742.38528512
  57173.86419945]
Mean: 50716.6120683
Standard deviation: 11534.2794299
Note: socre used is MSE 

'''
bag_reg_ng=BaggingRegressor(grad_reg_ng,n_estimators=10)
bag_ng_scores=cross_val_score(bag_reg_ng,Data,target_ng,scoring='neg_mean_squared_error',cv=5)

ng_scores= -(bag_ng_scores)

display_scores(ng_scores)

'''
Scores: [ 44495.70240416  70983.88316534  37909.66152658  40507.98598413
  55697.6781978 ]
Mean: 49918.9822556
Standard deviation: 12159.9833608
'''
# Fit the final bagged GBTs and make predictions on 20 percent random selected data points in the dataset

X_train,X_val,y_train,y_val=train_test_split(Data,target_ng,test_size=0.2)

grad_reg_ng=GradientBoostingRegressor(max_depth=5,max_features=170,learning_rate=0.1,n_estimators=150,min_samples_leaf=8)

bag_reg_ng=BaggingRegressor(grad_reg_ng,n_estimators=10)

bag_reg_ng.fit(X_train,y_train)

ng_predic=bag_reg_ng.predict(X_val)


plt.figure()
plt.hist(y_val,bins=30,ec='black',color='red')
plt.savefig('ng_target.jpg')

plt.figure()
plt.hist(ng_predic,bins=30,ec='black',color='red')
plt.savefig('ng_predict.jpg')




















