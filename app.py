import streamlit as st
st.set_page_config(page_title="Churn Dashboard",
                   page_icon=":bar_chart:", layout="wide")

from Views import employee, home,about,prediction
from streamlit_option_menu import option_menu

selected = option_menu(None, ["Home", "About Data", 'Predict', 'Employees'], 
    icons=['house', "bar-chart-line-fill", 'chevron-double-right', 'chevron-double-right'], 
    menu_icon="cast", default_index=0, orientation="horizontal",
    styles={
            "container": {"padding": "0!important"},
            "icon": {"color": "orange"}, 
            "nav-link": {"margin":"0px", "--hover-color": "#eee"},
        })

if selected == 'About Data':
    about.aboutView()
        
elif selected == 'Predict':
    prediction.churnPredict()

elif selected == 'Employees':
    employee.employees()
   
else:
    home.homeView()