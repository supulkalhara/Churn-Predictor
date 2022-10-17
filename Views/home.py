import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.error import URLError
import joblib
import plotly.express as px
import plotly.graph_objects as go
import requests
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie

# @st.cache
def get_data_from_csv():
    df = pd.read_csv('Data/churn_dataset.csv')
    return df

def get_model(file_path):
    loaded_model = joblib.load(file_path)
    return loaded_model

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def get_filtered_df(args , df):
    col = [
             'total_calls',
             'total_minutes',
             'total_charges',
             ]
    df_filtered = df.copy()
    df_filtered = df_filtered.drop(columns=col , errors='ignore')
    df_filtered['total_calls'] = 0
    df_filtered['total_minutes'] = 0
    df_filtered['total_charges'] = 0
    if("day" in args):
        df_filtered['total_calls'] += df['total_day_min']
        df_filtered['total_minutes'] += df['total_day_calls']
        df_filtered['total_charges'] += df['total_day_charge']
    if("evening" in args):
        df_filtered['total_calls'] += df['total_eve_min']
        df_filtered['total_minutes'] += df['total_eve_calls']
        df_filtered['total_charges'] += df['total_eve_charge']
    if("night" in args):
        df_filtered['total_calls'] += df['total_night_minutes']
        df_filtered['total_minutes'] += df['total_night_calls']
        df_filtered['total_charges'] += df['total_night_charge']
    if("international" in args):
        df_filtered['total_calls'] += df['total_intl_minutes']
        df_filtered['total_minutes'] += df['total_intl_calls']
        df_filtered['total_charges'] += df['total_intl_charge']

    return df_filtered

# @st.cache
def get_data_from_csv_model():
    df = pd.read_csv('Data/model_data.csv')
    return df

def get_data_from_csv_final():
    df = pd.read_csv('Data/chatterbox.csv')
    return df

def get_data_from_csv_original():
    df = pd.read_csv('Data/churn_dataset.csv')
    return df

def feature_importance(X, model):
    importances = model.feature_importances_
    # st.write(X.columns)
    fig = px.bar(x=X.columns, y=importances)
    return fig

def get_Table(p):
    df1 = p.describe().reset_index()
    header = df1.columns
    fig = go.Figure(data = go.Table(header = dict(values = list(header) , fill_color = "#002080" , align = 'center') ,
                                    cells = dict(values = df1 , fill_color = "#b3b3b3" , align = 'left')))
    #fig.update_layout(margin = dict(l=5 , r=5 , b=10 , t=10) , paper_bgcolor = "#000000")
    return fig

def show_correlations(dataframe, show_chart = True):
    corr = dataframe.corr()
    if show_chart == True:
        fig = px.imshow(corr,
                        text_auto=True,
                        width=900,
                        height=800)
        return fig

def homeView():
    df = get_data_from_csv()
    chatterbox = get_data_from_csv_final()

    df['voice_mail_plan_en'] = df.voice_mail_plan.map(dict(yes=1, no=0))
    df['intertiol_plan_en'] = df.intertiol_plan.map(dict(yes=1, no=0))
    df['total_plans'] = df['voice_mail_plan_en'] + df['intertiol_plan_en']

    fig_churn = px.pie(df, values=df.index, names='Churn')
    fig_churn.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    )

    fig_plans = px.pie(df, values=df.index, names='total_plans')
    fig_plans.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    )

    fig_cus_serve = px.ecdf(df, x="customer_service_calls" , y=df.index , color="Churn" ,  labels={"customer_service_calls": "customer service calls" , "index": "Customer Count"})
    fig_cus_serve.update_layout(
    xaxis=dict(tickmode="linear"),
    plot_bgcolor="rgba(0,0,0,0)",
    yaxis=(dict(showgrid=False)),)

    fig_serve_quality = px.ecdf(df, x="service quality" , y=df.index , color="Churn" ,  labels={"service quality": "service quality" , "index": "Customer Count"})
    fig_cus_serve.update_layout(
    xaxis=dict(tickmode="linear"),
    plot_bgcolor="rgba(0,0,0,0)",
    yaxis=(dict(showgrid=False)),)

    row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((.1, 2.3, .1, 1.3, .1))
    with row0_1:
        st.title('Churn Predictor Dashboard')
    with row0_2:
        st.text("")
        st.subheader('By [Haritha Perera](https://www.linkedin.com/in/)')

    st.header('Model Accuracies:')
    col1, col2, col3 = st.columns(3)
    col1.metric("RandomForest Classifier", "{:.2f}%".format(0.95*100), "high" )
    col2.metric("CatBoost Classifier",  "{:.2f}%".format(0.99*100), "high")
    col3.metric("LGBM Classifier",  "{:.2f}%".format(0.99*100), "high")

    st.text("")
    st.header('Current Churnings:')
    col1, col2, col3, col4 = st.columns(4)

    no_churns = chatterbox.loc[chatterbox['Churn'] == 0]['account_length'].count()
    churns = chatterbox.loc[chatterbox['Churn'] == 1]['account_length'].count()
    col1, col2, col3 = st.columns(3)
    
    col1.metric('Total: ', no_churns+churns)
    col2.metric('Available: ', no_churns)
    col3.metric('Churns: ', churns)
    st.markdown("""---""")

    with st.container():

        left_column, middle_column, right_column = st.columns(3)

        left_column.markdown("<h3 style='text-align: center; color: black;'>Churn Rate</h3>",
        unsafe_allow_html=True)
        left_column.plotly_chart(fig_churn, use_container_width=True)

        middle_column.markdown("<h3 style='text-align: center; color: black;'>Customer Services Calls</h3>",
        unsafe_allow_html=True)
        middle_column.plotly_chart(fig_cus_serve, use_container_width=True)

        right_column.markdown("<h3 style='text-align: center; color: black;'>Customer Service Quality</h3>",
        unsafe_allow_html=True)
        right_column.plotly_chart(fig_serve_quality, use_container_width=True)
    
    st.markdown("---")
    st.header('Variance:')
    st.markdown('Please select all the filters that are needed:')
    mainSection = st.container()

    with mainSection:
        col1, col2 = st.columns(2)
        churn = col1.multiselect(
            'Churn Rate',
            df["Churn"].unique(),
            default=df["Churn"].unique()
        )

        location = col2.multiselect(
            'Location',
            df["location_code"].unique(),
            default=df["location_code"].unique()
        )

        type = st.multiselect(
            'Type',
            ['day' , "evening" , "night" , "international"],
            default=['day' , "evening" , "night" , "international"]
        )

        df_selection = df.query(
            "Churn == @churn & location_code ==@location"
        )

        df_selection = get_filtered_df(type , df_selection)

        fig_call_1 = px.ecdf(df_selection, x="total_calls", color="Churn" , labels={"total_calls": "Total Calls"})
        fig_call_1.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False)))

        fig_min_1 = px.ecdf(df_selection, x="total_minutes", color="Churn" , labels={"total_minutes": "Total Minutes"})
        fig_min_1.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False)))

        fig_charge_1 = px.ecdf(df_selection, x="total_charges", color="Churn" , labels={"total_charges": "Total Charges"})
        fig_charge_1.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False)))

        fig_call_2 = px.scatter(df_selection, x="total_calls", y="customer_id", color="location_code",log_x= True, labels={"total_calls": "Total Calls" , "customer_id": "Customer Count"})
        fig_call_2.update_layout(
        xaxis=dict(tickmode="linear"),
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=(dict(showgrid=False)),)

        fig_min_2 = px.scatter(df_selection, x="total_minutes", y="customer_id",  color="location_code",log_x=True , labels={"total_minutes": "Total Minutes" , "customer_id": "Customer Count"})
        fig_min_2.update_layout(
        xaxis=dict(tickmode="linear"),
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=(dict(showgrid=False)),)

        fig_charge_2 = px.scatter(df_selection, x="total_charges", y="customer_id",  color="location_code", log_x=True , labels={"total_charges": "Total Charges" , "customer_id": "Customer Count"})
        fig_charge_2.update_layout(
        xaxis=dict(tickmode="linear"),
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=(dict(showgrid=False)),)

        fig_int_serve = px.bar(df_selection, x="intertiol_plan" , y=df_selection.index , color="Churn" ,  labels={"intertiol_plan": "International Services Plan" , "index": "Customer Count"})
        fig_int_serve.update_layout(
        xaxis=dict(tickmode="linear"),
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=(dict(showgrid=False)),)

        fig_voice_serve = px.bar(df_selection, x="voice_mail_plan" , y=df_selection.index , color="Churn" , labels={"voice_mail_plan": "Voice Mails Services Plan" , "index": "Customer Count"})
        fig_voice_serve.update_layout(
        xaxis=dict(tickmode="linear"),
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=(dict(showgrid=False)),)

        with st.container():
            st.markdown("<h3 style='text-align: center;'>Total Calls</h3>", unsafe_allow_html=True)
            left_column, right_column = st.columns(2)
            left_column.plotly_chart(fig_call_1, use_container_width=True)
            right_column.plotly_chart(fig_call_2, use_container_width=True)
        
        st.markdown("""---""")

        with st.container():
            st.markdown("<h3 style='text-align: center;'>Total Minutes</h3>", unsafe_allow_html=True)
            left_column, right_column = st.columns(2)
            left_column.plotly_chart(fig_min_1, use_container_width=True)
            right_column.plotly_chart(fig_min_2, use_container_width=True)
        
        st.markdown("""---""")

        with st.container():
            st.markdown("<h3 style='text-align: center;'>Total Charges</h3>", unsafe_allow_html=True)
            left_column, right_column = st.columns(2)
            left_column.plotly_chart(fig_charge_1, use_container_width=True)
            right_column.plotly_chart(fig_charge_2, use_container_width=True)
        
        st.markdown("""---""")

        with st.container():
            st.markdown("<h3 style='text-align: center;'>Services & Plans</h3><br/>",
            unsafe_allow_html=True)

            left_column, right_column = st.columns(2)

            left_column.markdown("<h5 style='text-align: center;'>International Services</h5>", unsafe_allow_html=True)
            left_column.plotly_chart(fig_int_serve, use_container_width=True)

            right_column.markdown("<h5 style='text-align: center;'>Voice Services</h5>", unsafe_allow_html=True)
            right_column.plotly_chart(fig_voice_serve, use_container_width=True)

        st.markdown("""---""")
    

        col1, col2 = st.columns(2)

        
        with st.expander("variance of churn with features:"):
            option = col1.selectbox(
            'Features:',
            (['priority customer level', 'service quality', 'total data usage', 'network coverage', 'total messages usage'])) 
            
            plt.figure()
            
            if option == 'priority customer level':
                count_0 = chatterbox.loc[(chatterbox['Churn'] == 1) & (chatterbox[option] == 1)]['account_length'].count()
                count_1 = chatterbox.loc[(chatterbox['Churn'] == 1) & (chatterbox[option] == 2)]['account_length'].count()
                count_2 = chatterbox.loc[(chatterbox['Churn'] == 1) & (chatterbox[option] == 3)]['account_length'].count()

                x = ['1','2','3']
                y = [count_0, count_1, count_2]
                
            # elif option == 'service quality':
            ## TODO!

            else:
                count_0 = chatterbox.loc[(chatterbox['Churn'] == 1) & (chatterbox[option] == 0)]['account_length'].count()
                count_1 = chatterbox.loc[(chatterbox['Churn'] == 1) & (chatterbox[option] == 1)]['account_length'].count()
                
                x = ['No','Yes']
                y = [count_0, count_1]
                
            data = pd.DataFrame({
                'index': x,
                'churn': y,
            }).set_index('index')
            
            col1.bar_chart(data)
        



