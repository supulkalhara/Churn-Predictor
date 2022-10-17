from re import X
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pymongo
from .home import *

def employees():
    st.title('Find Your Employeee!')
    model_rf = get_model('Data/model_rf.sav')
    model_lr = get_model('Data/model_lr.sav')
    model_ada = get_model('Data/model_ada.sav')

    # Initialize connection.
    # Uses st.experimental_singleton to only run once.
    @st.experimental_singleton
    def init_connection():
        return pymongo.MongoClient(**st.secrets["mongo"])

    client = init_connection()
    db = client.Churn

    items = db.Customers.find()
    array = list(items)

    user_arr = []
    for i in array:
        user_arr.append(i['customer_id'])
    
    user_id = st.selectbox ("Select user: ",user_arr)
    selected_user = ''

    for i in array:
        if i['customer_id'] == user_id:
            selected_user = i
            break

    keys = []
    for i in selected_user:
        keys.append(i)
    
    values = []
    for i in range(len(keys)):
        values.append(selected_user[keys[i]])

    del keys[0]
    del values[0]

    st.write("Click here and find out if this Customer will churn in the future!")

    col1, col2, col3 , col4, col5 = st.columns(5)

    with col1:
        pass
    with col2:
        pass
    with col4:
        pass
    with col5:
        pass
    with col3 :
        center_button = st.button('Find')

    if (center_button):
        unwanted = [0, 19, 20, 21, 22, 23]
 
        for ele in sorted(unwanted, reverse = True):
            del values[ele]

        total_calls = values[7] + values[10] + values[13] + values[16]
        total_charge = values[8] + values[11] + values[14] + values[17]
        total_mins = values[6] + values[9] + values[12] + values[15]

        if total_calls != 0: 
            avg_min = total_mins / total_calls
        else:
            avg_min = 0
        
        if values[1] == '445':
            values[1] = 0
        elif values[1] == '452':
            values[1] = 1
        else:
            values[1] = 2
        
        if values[2] == 'yes':
            values[2] = 1
        else:
            values[2] = 0
            
        if values[3] == "yes":
            values[3] = 1
        else:
            values[3] = 0

        values.append(total_calls)
        values.append(total_charge)
        values.append(total_mins)
        values.append(avg_min)

        df_columns = ['account_length', 
            'location_code', 
            'intertiol_plan',
            'voice_mail_plan',
            'number_vm_messages',
            'total_day_min',
            'total_day_calls',
            'total_day_charge',
            'total_eve_min',
            'total_eve_calls',
            'total_eve_charge',
            'total_night_minutes',
            'total_night_calls',
            'total_night_charge',
            'total_intl_minutes',
            'total_intl_calls',
            'total_intl_charge',
            'customer_service_calls',
            'total_mins',
            'total_calls',
            'total_charge',
            'avg_min_per_call'
            ]

        df = pd.DataFrame (values).T
        df.columns = df_columns   # type: ignore

        col1, col2, col3, col4, col5 = st.columns(5)

                # Random Forest
        col1.subheader("Random Forest")
        rf_prediction = model_rf.predict(df)
        if rf_prediction == 0:
            churn = "NO CHURN"
        else:
            churn = "CHURN"
        col1.write("user will " + str(churn))
            
        # Logistic Regression
        col3.subheader("Logistic Regression")
        lr_prediction = model_lr.predict(df)
        if lr_prediction == 0:
            churn2 = "NO CHURN"
        else:
            churn2 = "CHURN"
        col3.write("user will " + str(churn2))
            
            
        # ADA Boost
        col5.subheader("ADA Boost")
        ada_prediction = model_ada.predict(df)
        if ada_prediction == 0:
            churn3 = "NO CHURN"
        else:
            churn3 = "CHURN"
        col5.write("user will " + str(churn3))
        
        st.write("")
        st.write("")
        st.write("")

        col1, col2, col3 , col4, col5 = st.columns(5)

        with col1:
            pass
        with col2:
            pass
        with col4:
            pass
        with col5:
            pass
        with col3 :
            # Final Prediction
            if churn == churn2 == churn3:
                col3.metric("The User will:", 
                        churn)
            elif churn == churn2 or churn == churn3 :
                col3.metric("The User will:", 
                        churn)
            elif churn3 == churn2:
                col3.metric("The User will:", 
                        churn2)
            else:
                col3.write("3 models have different predictions!")

    else:
        df = pd.DataFrame({'X':np.array(keys), 'Y':np.array(values)})
        st.table(df)


