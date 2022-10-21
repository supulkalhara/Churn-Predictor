# from distutils.log import error
from matplotlib.ft2font import HORIZONTAL
import streamlit as st
import pandas as pd
from .home import *

def churnPredict():
    model_rf = get_model('Data/model_rf.sav')
    model_cat = get_model('Data/model_cat.sav')
    model_lgbm = get_model('Data/model_lgbm.sav')

    with st.form("my_form"):
            st.write("Please fill out all the fields for the customer")
        
            col1, col2, col3, col4 = st.columns(4)
            
            acc_length = col1.number_input("Account length: ")
            location_code = col2.selectbox(
                            'Location Code: ',
                            ('408','415','510' 
                            )
                        )  
            
            intertiol_plan = col3.selectbox(
                            'International Plan: ',
                            ('yes', 'no')
                        )   
            State = col4.number_input("State: ")
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
            service_quality = col3.number_input("Service Quality: ")
            priority_customer_level = col4.number_input("Priority Customer Level: ")
            total_data_usage = col1.number_input("Total Data Usage: ")
            network_coverage = col2.number_input("Network Coverage: ")
            total_messages_usage = col3.number_input("Total Message Usage: ")

            total_calls = total_day_calls + total_eve_calls + total_night_calls + total_intl_calls
            total_mins = total_day_min + total_eve_min + total_night_minutes + total_intl_minutes
            total_charge = total_day_charge + total_eve_charge + total_night_charge + total_intl_charge
            
            if total_calls != 0: 
                avg_min_per_call = total_mins/total_calls
            else:
                avg_min_per_call = 0
            
            if location_code == '408':
                location_code = 0
            elif location_code == '415':
                location_code = 1
            else:
                location_code = 2  

            if intertiol_plan == "yes":
                intertiol_plan = 1
            else:
                intertiol_plan = 0
            
        # Every form must have a submit button.
            try:
                submitted = st.form_submit_button("Submit")
                if submitted:
                    new_row = [acc_length, 
                            intertiol_plan,
                            number_vm_messages,
                            total_day_min,
                            total_day_calls,
                            total_eve_min,
                            total_eve_calls,
                            total_night_minutes,
                            total_night_calls,
                            total_intl_minutes,
                            total_intl_calls,
                            customer_service_calls,
                            service_quality,
                            priority_customer_level,
                            total_data_usage,
                            network_coverage,
                            total_messages_usage,
                            total_mins,
                            total_calls,
                            total_charge,
                            avg_min_per_call
                            ]
                    df_columns = ['account_length', 'intertiol_plan', 'number_vm_messages',
       'total_day_min', 'total_day_calls', 'total_eve_min', 'total_eve_calls',
       'total_night_minutes', 'total_night_calls', 'total_intl_minutes',
       'total_intl_calls', 'customer_service_calls', 'service quality',
       'priority customer level', 'total data usage', 'network coverage',
       'total messages usage', 'total_mins', 'total_calls', 'total_charge',
       'avg_min_per_call']
                    
                    col1, col2, col3 = st.columns(3)
                    
                    df = pd.DataFrame (new_row).T
                    df.columns = df_columns   # type: ignore

                    # Random Forest
                    col1.subheader("Random Forest")
                    rf_prediction = model_rf.predict(df)
                    print("random forest", rf_prediction)

                    if rf_prediction == 0:
                        churn = "NO CHURN"
                    else:
                        churn = "CHURN"
                    col1.write("User will " + str(churn))
                       
                       
                    # CatBoost Classifier
                    col2.subheader("CatBoost Classifier")
                    cat_prediction = model_cat.predict(df)
                    print("Catboost", cat_prediction)

                    if cat_prediction == 0:
                        churn2 = "NO CHURN"
                    else:
                        churn2 = "CHURN"
                    col2.write("User will " + str(churn2))
                        
                        
                    # LGBM Classifier
                    col3.subheader("LGBM Classifier")
                    lgbm_prediction = model_lgbm.predict(df)
                    print("LGBM", lgbm_prediction)

                    if lgbm_prediction == 0:
                        churn3 = "NO CHURN"
                    else:
                        churn3 = "CHURN"
                    col3.write("User will " + str(churn3))
                    
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