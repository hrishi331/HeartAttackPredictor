import pandas as pd 
import numpy as np 
import streamlit as st 
import pickle
# import train_model
# import test_model
import time

# Header of app
st.header('Heart Attack Predictor in Indian Youth')

df = pd.read_csv('train_data.csv').drop(['Heart Attack Likelihood','Unnamed: 0'],axis=1)

# Feature seperation
# from train_model import feature_seperation 
# df = feature_seperation(df)

with open('feature_seperation.pkl','rb') as file:
    feature_seperation = pickle.load(file)

df = feature_seperation(df)

# Feature names 
feature_names = df.columns

# Column names segregation in nominal and numerical
cat_col_names = df.select_dtypes(include=['object','category']).columns
num_col_names = df.select_dtypes(include=['int','float']).columns



# User inputs
user_inputs = []

n=1
for i in df:
    if i in cat_col_names:
        if n!=5:
            option_list = df[i].unique()
            user_inputs.append(st.radio(f"{n}.{i}",option_list))
        else:
            option_list = df[i].unique()
            user_inputs.append(st.radio(f"{n}.{i} (Socio Economic Status)",option_list))
    
    else:
        user_inputs.append(st.number_input(f"{n}.{i}"))

    n=n+1


# User inputs converted to array
user_inputs = np.array(user_inputs).reshape(1,len(user_inputs))

# User input converted to dataframe
df = pd.DataFrame(user_inputs,columns=feature_names)


# User input encoded
with open('encoder_lib.pkl','rb') as file:
    encoder_lib = pickle.load(file)


for i in encoder_lib:
    df[i] = encoder_lib[i].transform(df[i])


# Scaling
with open('Scaler.pkl','rb') as file:
    Scaler = pickle.load(file)

df = pd.DataFrame(Scaler.transform(df),columns=df.columns)

with open('dtc.pkl','rb') as file:
    dtc = pickle.load(file)

prediction = dtc.predict(df)
predict_probability = dtc.predict_proba(df)

# Load classification report
with open("classification_report.pkl",'rb') as file:
    classification_report = pickle.load(file)

if st.button('SUBMIT'):
    # Initiate progress bar
    progress_bar = st.progress(0)

    # Update the progress bar over time
    for i in range(100):
        time.sleep(0.05)  # Simulate a task by sleeping for a short time
        progress_bar.progress(i + 1)  # Update progress bar

    # Display a completion message
    st.success("Task Completed!")

    st.subheader('RESULTS')

    if prediction=='No':
        st.write(f"User is not likely to have **heart attack** with probability of **{round(predict_probability[0][0]*100,1)} %** ")    

    else:
        st.write(f"User is likely to have **heart attack** with probability of **{round(predict_probability[0][1]*100,1)} %**!!")

    st.subheader('Classification Report')
    st.write('.......for reference of Data Scientist')
    st.write(classification_report)

