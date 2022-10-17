import streamlit as st
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
from .home import *
import lightgbm as lgb


def show_correlations(dataframe, show_chart = True):
    corr = dataframe.corr()
    if show_chart == True:
        fig = px.imshow(corr,
                        text_auto=True,
                        width=600,
                        height=600)
        return fig

def feature_importance(X, rf_model):
    importances = rf_model.feature_importances_
    fig = px.bar(x=X.columns, y=importances)
    return fig
    
def aboutView():
    df = get_data_from_csv_final()
    df_data = get_data_from_csv_original()
    X = get_data_from_csv_model()
    model_rf = get_model('Data/model_rf.sav')
    model_lgbm = get_model('Data/model_lgbm.sav')

    obj_features = ['State', 'intertiol_plan', 'Churn']
    num_features = list(set(df.columns) - set(obj_features))

    with st.expander("See summary of the dataset"):
            st.write(df_data.describe())
            
#--------------------------------------
    st.write("")
    st.subheader("Feature Analysis")
    st.write("Select the features that you need to see relation")
    
    option3 = st.selectbox(
        'Graph type',
        ('Bar plot', 
         'Histogram', 
         'Line Chart', 
         'Area plot', 
         'Scatter plot'
         ) # 
    )   

    col1, col2 = st.columns(2)
    
    if (option3 == 'Bar plot'):

        option4 = col1.selectbox(
        'Feature 1:',
        (num_features))  
        
        type = col1.radio(
            "Do you want to plot 2 fetures?",
            ('No', 'Yes'))
        if type == 'No':
            st.bar_chart(df[option4])
        elif type == 'Yes':
            option5 = col2.selectbox(
            'Feature 2:',
            (num_features))
            
            if (option4 != option5):
    
                chart_data = [df[option4], df[option5]]
                st.bar_chart(chart_data)
            else:
                original_title = '<p style="font-family:Courier; color:red; font-size: 15px;">You cannot select the same feature</p>'
                st.markdown(original_title, unsafe_allow_html=True)

    elif (option3 == 'Histogram'):
        option4 = col1.selectbox(
        'Feature 1:',
        (num_features))  
        
        type = col1.radio(
            "Do you want to plot 2 fetures?",
            ('No', 'Yes'))
        if type == 'No':
            fig = ff.create_distplot([df[option4]], [option4])
            st.plotly_chart(fig, use_container_width=True)
            
        elif type == 'Yes':
            option5 = col2.selectbox(
            'Feature 2:',
            (num_features))
            
            if (option4 != option5):
                x1 = df[option4]
                x2 = df[option5]
                
                group_labels = [option4, option5]
                
                hist_data = [x1, x2]
                fig = ff.create_distplot(hist_data, group_labels, bin_size=[.1, .25, .5])
                st.plotly_chart(fig, use_container_width=True)
            else:
                original_title = '<p style="font-family:Courier; color:red; font-size: 15px;">You cannot select the same feature</p>'
                st.markdown(original_title, unsafe_allow_html=True)
        
    elif (option3 == 'Line Chart') :
        option4 = col1.selectbox(
        'Feature 1:',
        (num_features))  

        option5 = col2.selectbox(
        'Feature 2:',
        (num_features))
        
        if (option4 != option5):
            x1 = df[option4]
            x2 = df[option5]
            group_labels = [option4, option5]
            
            chart_data = [x1, x2]
            fig = ff.create_distplot(chart_data, group_labels)
            st.plotly_chart(fig)
        else:
            original_title = '<p style="font-family:Courier; color:red; font-size: 15px;">You cannot select the same feature</p>'
            st.markdown(original_title, unsafe_allow_html=True)
        
    elif (option3 == 'Area plot'):
        st.area_chart(df)
        
    elif (option3 == 'Scatter plot'):
        option4 = col1.selectbox(
        'Feature 1:',
        (num_features))  
        
        option5 = col2.selectbox(
        'Feature 2:',
        (num_features))
        
        if (option4 != option5):
            x1 = df[option4]
            x2 = df[option5]
            group_labels = [option4, option5]
            fig = px.scatter(df,
                             x=option4,
                             y=option5,
            )
            st.plotly_chart(fig)
        else:
            original_title = '<p style="font-family:Courier; color:red; font-size: 15px;">You cannot select the same feature</p>'
            st.markdown(original_title, unsafe_allow_html=True)
                
    st.subheader("Feature Importance")   
    st.write("See feature importance using light GBM:")
    fig = feature_importance(X, model_lgbm)
    st.plotly_chart(fig, use_container_width=True)


    #---------------------------------------------
            
    st.subheader("Correlations")
    st.write("Select the features that you need to see correlation")
    col1, col2 = st.columns(2)
    
    col = list(df.columns)
    correlation_df = show_correlations(df[col],show_chart=True)
    
    option = col1.selectbox(
        'Feature 1',
        (df.columns))   

    option2 = col2.selectbox(
        'Feature 2',
        (df.columns))
    
    result = df[option].corr(df[option2])

    if (result < 0.8):
        st.metric("Correlation", "{:.2f}".format(result), "- low")
    else:
        st.metric("Correlation", "{:.2f}".format(result), "+ high")
        
    st.plotly_chart(correlation_df, use_container_width=True)
        
