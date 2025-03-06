import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder
import pickle

# load dataset
X_train = pd.read_csv('train_data.csv').drop(['Heart Attack Likelihood','Unnamed: 0'],axis=1)
y_train = pd.read_csv('train_data.csv')['Heart Attack Likelihood']


# # Over sampling using Random Over Sampler 
from imblearn.over_sampling import RandomOverSampler 
sampler = RandomOverSampler(random_state=11)
X_train,y_train = sampler.fit_resample(X_train,y_train)


# Feature seperation 
# converting '172.5/180.2' into 172.5 and 180.2 in different columns

def feature_seperation(X_train):
    def systolic(i):
        return float(i.split('/')[0].strip())

    def diastolic(i):
        return float(i.split('/')[1].strip())

    X_train['systolic mmHg'] = X_train['Blood Pressure (systolic/diastolic mmHg)'].apply(systolic)
    X_train['diastolic mmHg'] = X_train['Blood Pressure (systolic/diastolic mmHg)'].apply(diastolic)

    # Dropping original column after feature seperation
    X_train = X_train.drop('Blood Pressure (systolic/diastolic mmHg)',axis=1)

    return X_train 

X_train = feature_seperation(X_train)

with open('feature_seperation.pkl','wb') as file:
    pickle.dump(feature_seperation,file)

# categorical and numerical data
cat_X = X_train.select_dtypes(include=['object','category'])
num_X = X_train.select_dtypes(include=['int','float'])

# Create encoder library
enocder_lib = {}
for i in cat_X:
    le = LabelEncoder()
    X_train[i] = le.fit_transform(X_train[i])
    enocder_lib[i] = le 
    

# pickle encoder library 
import pickle
with open('encoder_lib.pkl','wb') as file:
    pickle.dump(enocder_lib,file)

# Scaling 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns)

# Pickle Scaler
with open('Scaler.pkl','wb') as file:
    pickle.dump(scaler,file)


# Make target numerical
# y_replace_dict = {'Yes':1,'No':0}
# y_train = y_train.replace(y_replace_dict)

# Fit model
from sklearn.tree import DecisionTreeClassifier 



dtc = DecisionTreeClassifier(random_state=11,max_depth=10,min_samples_split=10,class_weight='balanced') #  
dtc.fit(X_train,y_train)

# Pickle trained model object
with open('dtc.pkl','wb') as file:
    pickle.dump(dtc,file)



with open('column_names.pkl','wb') as file:
    pickle.dump(X_train.columns,file)


with open('feature_names.pkl','wb') as file:
    pickle.dump(X_train.columns,file)

