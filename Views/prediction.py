# from distutils.log import error
from matplotlib.ft2font import HORIZONTAL
import streamlit as st
import pandas as pd
from .home import *

def churnPredict():
    model_rf = get_model('Data/model_rf.sav')
    model_lr = get_model('Data/model_lr.sav')
    model_ada = get_model('Data/model_ada.sav')

    with st.form("my_form"):
            st.write("Please fill out all the fields for the customer")
        
            col1, col2, col3, col4 = st.columns(4)
            acc_length = col1.number_input("Account length: ")
            location_code = col2.selectbox(
                            'Location Code: ',
                            ('445','452','547' 
                            )
                        )  
            
            intertiol_plan = col3.selectbox(
                            'International plan: ',
                            ('yes', 'no')
                        )   
            
            voice_mail_plan = col4.selectbox(
                            'Voice Mail Plan: ',
                            ('yes', 'no')
                        )   
            
            number_vm_messages = col1.number_input("VM message count: ")
            total_day_min = col2.number_input("Day minutes: ")
            total_day_calls = col3.number_input("Day calls: ")
            total_day_charge = col4.number_input("Day charge: ")
            total_eve_min = col1.number_input("Evening minutes: ")
            total_eve_calls = col2.number_input("Evening calls: ")
            total_eve_charge = col3.number_input("Evening charge: ")
            total_night_minutes = col4.number_input("Night minutes: ")
            total_night_calls = col1.number_input("Night calls: ")
            total_night_charge = col2.number_input("Night charge: ")
            total_intl_minutes = col3.number_input("International minutes: ")
            total_intl_calls = col4.number_input("International calls: ")
            total_intl_charge = col1.number_input("International charge: ")
            customer_service_calls = col2.number_input("Customer Service Calls: ")
            total_calls = total_day_calls + total_eve_calls + total_night_calls + total_intl_calls
            total_charge = total_day_charge + total_eve_charge + total_night_charge + total_intl_charge
            total_mins = total_day_min + total_eve_min + total_night_minutes + total_intl_minutes
            
            if total_calls != 0: 
                avg_min_per_call = total_mins/total_calls
            else:
                avg_min_per_call = 0
            
            if location_code == '445':
                location_code = 0
            elif location_code == '452':
                location_code = 1
            else:
                location_code = 2
            
            if intertiol_plan == 'yes':
                intertiol_plan = 1
            else:
                intertiol_plan = 0
                
            if voice_mail_plan == "yes":
                voice_mail_plan = 1
            else:
                voice_mail_plan = 0
            
        # Every form must have a submit button.
            try:
                submitted = st.form_submit_button("Submit")
                if submitted:
                    new_row = [acc_length, 
                            location_code, 
                            intertiol_plan,
                            voice_mail_plan,
                            number_vm_messages,
                            total_day_min,
                            total_day_calls,
                            total_day_charge,
                            total_eve_min,
                            total_eve_calls,
                            total_eve_charge,
                            total_night_minutes,
                            total_night_calls,
                            total_night_charge,
                            total_intl_minutes,
                            total_intl_calls,
                            total_intl_charge,
                            customer_service_calls,
                            total_mins,
                            total_calls,
                            total_charge,
                            avg_min_per_call
                            ]
                    df_columns = ['acc_length', 
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
                    
                    col1, col2, col3 = st.columns(3)
                    
                    df = pd.DataFrame (new_row).T
                    df.columns = df_columns   # type: ignore
                    
                    
                    # Random Forest
                    col1.subheader("Random Forest")
                    rf_prediction = model_rf.predict(df)
                    if rf_prediction == 0:
                        churn = "NO CHURN"
                    else:
                        churn = "CHURN"
                    col1.write("model will " + str(churn))
                       
                       
                    # Logistic Regression
                    col2.subheader("Logistic Regression")
                    lr_prediction = model_lr.predict(df)
                    if lr_prediction == 0:
                        churn2 = "NO CHURN"
                    else:
                        churn2 = "CHURN"
                    col2.write("model will " + str(churn2))
                        
                        
                    # ADA Boost
                    col3.subheader("ADA Boost")
                    ada_prediction = model_ada.predict(df)
                    if ada_prediction == 0:
                        churn3 = "NO CHURN"
                    else:
                        churn3 = "CHURN"
                    col3.write("model will " + str(churn3))
                    
                    st.write("")
                    st.write("")
                    st.write("")
                    
                    # Final Prediction
                    if churn == churn2 == churn3:
                        st.metric("Predicted Churn for the entered details:", 
                                churn)
                    elif churn == churn2 or churn == churn3 :
                        st.metric("Predicted Churn for the entered details:", 
                                churn)
                    elif churn3 == churn2:
                        st.metric("Predicted Churn for the entered details:", 
                                churn2)
                    else:
                        st.write("3 models have different predictions!")

                        # st.markdown("<p style='text-align: center; color: Black;'>User will</p>",unsafe_allow_html=True)    
                        # st.markdown("<h2 style='text-align: center; color: Green;'>Not Churn!</h2>",unsafe_allow_html=True)    

                        # st.markdown("<p style='text-align: center; color: Black;'>User will</p>",unsafe_allow_html=True)    
                        # st.markdown("<h2 style='text-align: center; color: Red;'>Churn!</h2>",unsafe_allow_html=True)  

            except TypeError as err:
                st.write('err', err)