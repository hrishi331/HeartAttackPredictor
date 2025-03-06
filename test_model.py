import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report 

import pickle

# Import test data 
X_test = pd.read_csv("test_data.csv").drop(['Heart Attack Likelihood','Unnamed: 0'],axis=1)
y_test = pd.read_csv("test_data.csv")['Heart Attack Likelihood']

# Import feature_seperation moudle
from train_model import feature_seperation
X_test = feature_seperation(X_test)


# Import label encoder
with open('encoder_lib.pkl','rb') as file:
    encoder_lib = pickle.load(file)


# Create categorical column segreation
cat_df = X_test.select_dtypes(include=['object','category'])


# Encoding of test data
for i in encoder_lib:
    encoder = encoder_lib[i]
    X_test[i] = encoder.transform(X_test[i])

# Scaling of test data 
with open('Scaler.pkl','rb') as file:
    Scaler = pickle.load(file)

X_test = pd.DataFrame(Scaler.transform(X_test),columns=X_test.columns)

# Encoding target
# rep_target = {'Yes':1,'No':0} 
# y_test = y_test.replace(rep_target)

# Loading model 
with open('dtc.pkl','rb') as file:
    dtc = pickle.load(file)

y_predicted = dtc.predict(X_test)

# Creating classification report
from sklearn.metrics import classification_report



def model_report(y_test,y_predicted):
    from sklearn.metrics import classification_report
    cr = classification_report(y_test,y_predicted,output_dict=True)
    return pd.DataFrame(cr).transpose()



classification_report = model_report(y_test,y_predicted)
with open('classification_report.pkl','wb') as file:
    pickle.dump(classification_report,file)

print(classification_report)


