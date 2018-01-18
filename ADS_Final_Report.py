#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, f1_score, recall_score,mean_squared_error 
from sklearn.model_selection import train_test_split
import xgboost as xgb
import operator

def score(true,pred):
  return (precision_score(true,pred),
          recall_score(true,pred),
          f1_score(true,pred))

def print_score(s):
  print("""
Precision:  {:0.3}
Recall:     {:0.3}
F-Score:    {:0.3}
""".format(*s))

# Loading data
df_train_data = pd.read_csv('train_users_2.csv')
# Getting a label at 80% test data
piv_train = round(df_train_data.shape[0]*0.8)

# Printing the Value
print('piv_train -', piv_train)

# Splitting the Training and Test Data
df_train = df_train_data.iloc[:piv_train,:]
df_test  = df_train_data.iloc[piv_train:,:]

# Printing the shape of train and test data
print("df_train.shape -->" ,df_train.shape)
print("df_test.shape -->" ,df_test.shape)

# Storing the labels of train and test data in labels_train and labels_test
# These Values are stored since country destination will be dropped from
# df_train, and labels_test will be used as true values
labels_train = df_train['country_destination'].values
labels_test = df_test['country_destination'].values

# Dropping the country destination from train and test data
df_train = df_train.drop(['country_destination'], axis=1)
df_test  = df_test.drop(['country_destination'], axis=1)

# Print the shapes

print("df_train.shape after drop of country destination" ,df_train.shape)
print("df_test.shape after drop of country destination" ,df_test.shape)

#
df_test = df_test.reset_index(drop=True)

## Check First three Rows
# df_test.iloc[:3]
       
## Getting the id's from df_test       
id_test = df_test['id']

# Check first three rows of id
# id_test.iloc[:3]

## Creating a DataFrame With train and test data
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
print("df_all.shape -->" ,df_all.shape)


# Handling Age

df_all.loc[(df_all['age'] > 1900), 'age'] = 2017 - df_all['age']

#Identify age which is outlier
df_all.loc[(df_all['age'] <= 14) | (df_all['age'] >= 100), 'age'] = np.NAN


##Removing id and date_first_booking to make it ready for sparse matrix
df_all = df_all.drop(['id', 'date_first_booking'], axis=1)

## Filling nan with -1
df_all = df_all.fillna(-1)


# Creating the date_account_created by splitting it in to year, month and date
dac = np.vstack(
    df_all.date_account_created.astype(str).apply(
        lambda x: list(map(int, x.split('-')))
        ).values
    )
df_all['dac_year'] = dac[:,0]
df_all['dac_month'] = dac[:,1]
df_all['dac_day'] = dac[:,2]
df_all = df_all.drop(['date_account_created'], axis=1)

# creating the timestamp_first_active
tfa = np.vstack(
    df_all.timestamp_first_active.astype(str).apply(
        lambda x: list(map(int, [x[:4], x[4:6], x[6:8],
                                 x[8:10], x[10:12],
                                 x[12:14]]))
        ).values
    )
df_all['tfa_year'] = tfa[:,0]
df_all['tfa_month'] = tfa[:,1]
df_all['tfa_day'] = tfa[:,2]
df_all = df_all.drop(['timestamp_first_active'], axis=1)

# Check values, first three rows
# dfa_all.iloc[:3]

# Use the get dummies to create the sparse matrix
# This code will create binary 0 or 1 baased on category
ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 
             'affiliate_channel', 'affiliate_provider', 
             'first_affiliate_tracked', 'signup_app', 
             'first_device_type', 'first_browser']
for f in ohe_feats:
    df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
    df_all = df_all.drop([f], axis=1)
    df_all = pd.concat((df_all, df_all_dummy), axis=1)

print("df_all.shape -->" ,df_all.shape)

#We need to split the sparse metrix
df_train_tdm = df_all.iloc[:piv_train,:]
df_test_tdm  = df_all.iloc[piv_train:,:]

print("df_train_tdm.shape -->" ,df_train_tdm.shape)
print("df_test_tdm.shape -->" ,df_test_tdm.shape)

# Using the labelencoder to transform the country destinations of 
# training set
le = LabelEncoder() 
y = le.fit_transform(labels_train)   

# Print the Value
print("y -->" ,y.shape)

# Classifier
params = {'eta': 0.2,
          'max_depth': 6,
          'subsample': 0.5,
          'colsample_bytree': 0.5,
          'objective': 'multi:softprob',
          'num_class': 12}
num_boost_round = 1

dtrain = xgb.DMatrix(df_train_tdm, y)

# dtrain.num_row()
# dtrain.num_col()

clf1 = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_boost_round)

# Get feature scores and store in DataFrame
importance = clf1.get_fscore()
importance_df = pd.DataFrame(
    sorted(importance.items(), key=operator.itemgetter(1)), 
    columns=['feature','fscore']
    )
# Plot feature importance of top 20
importance_df.iloc[-20:,:].plot(x='feature',y='fscore',kind='barh')

print("df_test.shape -->" ,df_test.shape)

# Only select features w/ a feature score (can also specify min fscore)
# Retrain model with reduced feature set
df_all = df_all[importance_df.feature.values]
print("df_all.shape -->" ,df_all.shape)

#We need to split the sparse metrix
df_train_tdm = df_all.iloc[:piv_train,:]
df_test_tdm  = df_all.iloc[piv_train:,:]

print("df_train_tdm.shape -->" ,df_train_tdm.shape)
print("df_test_tdm.shape -->" ,df_test_tdm.shape)

# Create a matrix to pass to the train along with the output Values
dtrain = xgb.DMatrix(df_train_tdm, y)
print("dtrain.rows -->" ,dtrain.num_row())
print("dtrain.cols -->" ,dtrain.num_col())

clf2 = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_boost_round)

# Added this code for predict

clf2.save_model("xgboost_train")
bst = xgb.Booster(params)
bst.load_model("xgboost_train")

y_pred = bst.predict(xgb.DMatrix(df_test_tdm)).reshape(df_test.shape[0],12)

# Converts the Model Predicted values to list of destination countries
def get_predicted_vals(pred):
    ids = []  #list of ids
    cts = []  #list of countries
    for i in range(len(df_test)):    
        idx = id_test[i]        
        ids += [idx] * 1
        #print (idx,ids,i)
        cts += le.inverse_transform(np.argsort(pred[i])[::-1])[:1].tolist()
    return cts
    #print (cts)

## So cts are the predicted values for ids
# Check the values as cts[0]
# cts[1]

# Lets do a loop and ocnvert the true value into binary format, if it matches 1
# else 0
def convert_2_binary(true_val, predicted):
    label_true = []
    label_predicted = []
    for i in range(len(cts)):
        if true_val[i] != predicted[i]:
            label_true.append(1)
            label_predicted.append(0)
        else:
            label_true.append(1) 
            label_predicted.append(1)
    return label_true, label_predicted

print("\n\n XGB Performance")
cts = get_predicted_vals(y_pred)
print('XGB Score',f1_score(labels_test, cts, average ='micro'))

true_vals, pred_val = convert_2_binary(labels_test,cts)
s = score(true_vals,pred_val)
print_score(s)
          
#Naive Bayes
print("\n\n Naive Bayes Performance")
clfNB = GaussianNB()
clfNB.fit(df_train_tdm, y)
y_nb_pred = clfNB.predict(df_test_tdm)
cts = get_predicted_vals(y_nb_pred)
print('Naive Bayes Score -> ',f1_score(labels_test, cts, average='micro'))

true_vals, pred_val = convert_2_binary(labels_test,cts)
s = score(true_vals,pred_val)
print_score(s)

#Random Forest 
print("\n\n Random Forest Performance")
forest = RandomForestClassifier(n_estimators=100)
forest.fit(df_train_tdm, y)
y_rf_pred = forest.predict_proba(df_test_tdm) 
cts = get_predicted_vals(y_rf_pred)
forest_score = f1_score(labels_test, cts, average='micro') 
print ("Random Forest Score: ", forest_score)

true_vals, pred_val = convert_2_binary(labels_test,cts)
s = score(true_vals,pred_val)
print_score(s)

#Logistic Regression
print("\n\n Logistic Regression Performance")
clf_lr = LogisticRegression(solver='lbfgs',multi_class='multinomial').fit(df_train_tdm,y)
y_lr_pred = clf_lr.predict_proba(df_test_tdm)
cts = get_predicted_vals(y_lr_pred)
score_lr = f1_score(labels_test, cts, average = 'micro') 
print ("Logistics Regression Score: ", score_lr)

true_vals, pred_val = convert_2_binary(labels_test,cts)
s = score(true_vals,pred_val)
print_score(s)

#Neural Network
print("\n\nNeural Network Performance")
from sklearn.neural_network import MLPClassifier
nn = MLPClassifier()
nn.fit(df_train_tdm, y)
y_nn_pred = nn.predict(df_test_tdm)
cts = get_predicted_vals(y_nn_pred)
score_nn = f1_score(labels_test, cts, average = 'micro') 
print ("Nueral Network Score: ", score_nn)


true_vals, pred_val = convert_2_binary(labels_test,cts)
s = score(true_vals,pred_val)
print_score(s)




