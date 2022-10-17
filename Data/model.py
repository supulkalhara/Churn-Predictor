from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

import joblib
import pandas as pd
import numpy as np

def iqr(df , features):
  for f in features:
    mean = df[f].mean()
    Q1 = df[f].quantile(0.25)
    Q3 = df[f].quantile(0.75)
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    df[f] = np.where(df[f] <lower_range,lower_range ,df[f])
    df[f] = np.where(df[f] >upper_range, upper_range,df[f])
  return df

# RandomForestClassifier
def random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    model_rf = RandomForestClassifier(n_estimators=1000 , 
                                      oob_score = True, 
                                      n_jobs = -1,
                                      random_state =50, 
                                      max_features = "auto",
                                      max_leaf_nodes = 30)
    model_rf.fit(X_train, y_train)
    preds = model_rf.predict(X_test)
    return model_rf, metrics.accuracy_score(y_test, preds)
  
# LogisticRegression
def logistic_regression(X, y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
  model = LogisticRegression()
  model.fit(X_train, y_train)
  prediction_test = model.predict(X_test)
  return model, metrics.accuracy_score(y_test, prediction_test)

#ADA Boost
def ada_boost(X, y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)
  model = AdaBoostClassifier()
  model.fit(X_train,y_train)
  preds = model.predict(X_test)
  return model, metrics.accuracy_score(y_test, preds)

chatterbox = pd.read_csv('Data\Train_Dataset.csv')

"""## **Data Cleaning**"""
customer_id_temp = chatterbox['customer_id']
chatterbox = chatterbox.drop(columns=['customer_id', 'Unnamed: 20'])
obj_features = ['location_code', 'intertiol_plan' , 'voice_mail_plan' , 'Churn']
num_features = list(set(chatterbox.columns) - set(obj_features))

cat_col1 = ['location_code' , 'intertiol_plan' , 'voice_mail_plan']

correlated_feature_couple = [['total_eve_charge', 'total_eve_min'],
                   ['total_night_charge', 'total_night_minutes'],
                   ['total_intl_charge', 'total_intl_minutes'],
                   ['total_day_charge', 'total_day_min']]

non_corr_numer_col = ['account_length',
             'number_vm_messages',
             'total_day_calls',
             'total_eve_calls',
             'total_night_calls',
             'total_intl_calls',
             'customer_service_calls']

onehot_col = ['location_code_452' , 'location_code_445' ,'location_code_547']

chatterbox = chatterbox.drop_duplicates()

for f in num_features:
  chatterbox[f] = np.where(chatterbox[f] < 0, np.NaN , chatterbox[f])

chatterbox = chatterbox.dropna()

rem_features = ['total_day_min', 'total_day_calls', 'total_day_charge', 
                'total_eve_min', 'total_eve_calls', 'total_eve_charge', 
                'total_night_minutes', 'total_night_calls', 'total_night_charge', 
                'total_intl_minutes', 'total_intl_calls', 'total_intl_charge']

chatterbox = iqr(chatterbox, rem_features)

chatterbox.voice_mail_plan = chatterbox.voice_mail_plan.map(dict(yes=1, no=0))
chatterbox.intertiol_plan = chatterbox.intertiol_plan.map(dict(yes=1, no=0))
chatterbox.Churn = chatterbox.Churn.map(dict(Yes=1, No=0))

chatterbox['location_code'].replace(to_replace=445.0, value=0, inplace=True)
chatterbox['location_code'].replace(to_replace=452.0,  value=1, inplace=True)
chatterbox['location_code'].replace(to_replace=547.0,  value=2, inplace=True)

chatterbox['total_mins'] = chatterbox['total_day_min'] + chatterbox['total_eve_min'] + chatterbox['total_night_minutes']
chatterbox['total_calls'] = chatterbox['total_day_calls'] + chatterbox['total_eve_calls'] + chatterbox['total_night_calls']
chatterbox['total_charge'] = chatterbox['total_day_charge'] + chatterbox['total_eve_charge'] + chatterbox['total_night_charge']
chatterbox['avg_min_per_call'] = chatterbox['total_mins']  / chatterbox['total_calls']

num_features = list(set(chatterbox.columns) - set(obj_features))
pre_processed_dataset_train = chatterbox

y = chatterbox['Churn']
X = chatterbox.drop(columns=['Churn'])

rf_model, rf_accuracy = random_forest(X, y)
lr_model, lr_accuracy = logistic_regression(X, y)
ada_model, ada_accuracy = ada_boost(X, y)

filename = 'model_rf.sav'
joblib.dump(rf_model, filename)

filename = 'model_lr.sav'
joblib.dump(lr_model, filename)

filename = 'model_ada.sav'
joblib.dump(ada_model, filename)

X.to_csv('model_data.csv' , index = False)
chatterbox.to_csv('chatterbox.csv' , index = False)