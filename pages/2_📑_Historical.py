
# Import necessary libraries
import streamlit as st
import pandas as pd
import os
import voice_recognition_module as vr
import plotly.express as px
import numpy as np 
import plotly.graph_objects as go
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import sqlite3
import pyodbc
from datetime import datetime, date, timedelta
from functions import * 
from variables import *

if "authentication_status" not  in  st.session_state:
    st.session_state["authentication_status"] = ""
st.set_page_config(page_icon=':bar_chart:' )
if not st.session_state["authentication_status"]:
    st.warning("You must log-in to see the content of this sensitive page! Reload page to login.")
    st.stop()  # App won't run anything after this line


def plot_bubble_chart(transcription, speaker):
    # Tokenize the transcription
    words = word_tokenize(transcription.lower())

    # Remove punctuation and stopwords
    filtered_words = [word for word in words if word.isalpha() and word not in stopwords.words('english')]

    # Count the frequency of each word
    word_counts = Counter(filtered_words)

    # Convert to DataFrame for plotting
    df = pd.DataFrame(word_counts.items(), columns=['Word', 'Frequency'])

    # Plot using Plotly
    fig = px.scatter(df, x='Word', y='Frequency', size='Frequency', size_max=100, template='plotly_white',
                     title="Word Frequencies in Transcription Speaker " + speaker)

    # Display the chart
    st.plotly_chart(fig)


def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

st.title('Historical data')
st.write("Filter by client or date of the analyzed call recordings and then click on the desired record to display the analytics.")

# Using the container's class for styling
local_css("style.css")

#Database connection 

#conn = sqlite3.connect('OPTcallsAnalytics.db')
Driver="Driver={ODBC Driver 18 for SQL Server};Server=tcp:opt-call-analyzer-server.database.windows.net,1433;Database=OPTCallsAnalytics;Uid="+str(SQLUSER)+";Pwd={"+str(SQLPASS)+"};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=250;"

conn = pyodbc.connect(Driver)
c = conn.cursor()
if 'viewPage' not in st.session_state:
    st.session_state.viewPage = "first"
##Filters section 

if st.session_state.viewPage == "first":
    col1, col2,col3,col4,col5 = st.columns([5,5,5,5,5])
    # Fetch client names from the Customers table
    c.execute("SELECT DISTINCT name FROM customersCalls")
    client_names = [item[0] for item in c.fetchall()]
    c.execute("SELECT DISTINCT EmployeeName FROM customersCalls")
    employees = [item[0] for item in c.fetchall()]
    c.execute("SELECT DISTINCT callType FROM customersCalls")
    callTypes = [item[0] for item in c.fetchall()]

    # Dropdown to select a client
    selected_client = selectbox_with_default('Client', client_names, col=col1)
    selected_employee = selectbox_with_default('Employee', employees, col=col2)
    selected_call_type = selectbox_with_default('Type', callTypes, col=col3)

    #
    ## Dropdown to select a date
    if selected_client ==DEFAULT and selected_employee ==DEFAULT:
        c.execute("SELECT Date FROM customersCalls")
    else:
        query_str = 'name="' + selected_client+'"' if selected_client!=DEFAULT else ''
        query_str += (" and " if query_str and selected_employee!=DEFAULT else "") + 'EmployeeName="' + selected_employee+'"' if selected_employee!=DEFAULT else ''
        query_str +=( " and " if query_str  and selected_call_type!=DEFAULT else "") + 'callType="' + selected_call_type+'"' if selected_call_type!=DEFAULT else ''
        c.execute(f"SELECT Date FROM customersCalls where {query_str}")

    dates = [item[0] for item in c.fetchall()]
    #selected_date = selectbox_with_default('Date', dates, col=col2)


    selected_date_1 = col4.date_input("From:",(min(dates)))
    selected_date_2 = col5.date_input("To:", max(dates)+timedelta(days=1))


    ################
    ##table with all records 
    if selected_client ==DEFAULT and selected_employee ==DEFAULT:
        customersCalls = f"SELECT name, recordingID, EmployeeName,callType, date from customersCalls where date between '{selected_date_1}' and '{selected_date_2}'"
    else: 
        query_str = 'name="' + selected_client+'"' if selected_client!=DEFAULT else ''
        query_str += (" and " if query_str and selected_employee!=DEFAULT else "" )+ 'EmployeeName="' + selected_employee+'"' if selected_employee!=DEFAULT else ''
        query_str += (" and " if query_str and selected_call_type!=DEFAULT else "" )+ 'callType="' + selected_call_type+'"' if selected_call_type!=DEFAULT else ''
        customersCalls = f"SELECT name, EmployeeName,callType, recordingID, date from customersCalls where date between '{selected_date_1}' and '{selected_date_2}'{' and ' if query_str else ''} {query_str}"
    df = pd.read_sql_query(customersCalls, conn)

    df = df.rename(columns={"name":"Client", "date":"Date"})

    st.text("Records on file")

    selection = dataframe_with_selections(df)
    if 'historical_selections' not in st.session_state:
        st.session_state.historical_selections = {}




    if st.button("View Analytics"):
        if len(selection['selected_rows_indices']) == 0:
            st.warning("🔥 No selected items")
        else:
            st.session_state.historical_selections = selection
            st.session_state.viewPage = "second"
            st.experimental_rerun()
   

#Tabs section
elif st.session_state.viewPage == "second":
    if st.button("Return to selection"):
        st.session_state.viewPage = "first"
        st.session_state.historical_selections = {}
        st.experimental_rerun()
    tabs = (tuple(st.session_state.historical_selections['selected_rows']['recordingID']))
    tabs_list = st.tabs(tabs)

    for i, item in enumerate(tabs):
        with tabs_list[i]:
            c.execute(f"SELECT cleanTranscription FROM customersCalls WHERE recordingID = '{str(item)}'")
            transcription = (c.fetchone()[0])
            c.execute(f"SELECT callType FROM customersCalls WHERE recordingID = '{str(item)}'")
            call_type = (c.fetchone()[0])
            c.execute(f"SELECT EmployeeName FROM customersCalls WHERE recordingID = '{str(item)}'")
            EmployeeName = (c.fetchone()[0])
            c.execute(f"SELECT name FROM customersCalls WHERE recordingID = '{str(item)}'")
            client = (c.fetchone()[0])
            st.markdown(
                    f"""
                    <div>
                        <b>Client: {client} - Call Type: {call_type} - Employee: {EmployeeName}<br></b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with st.expander("Clear Transcription", expanded=False):
                st.markdown(
                    f"""
                    <div class="summary-container">
                        {transcription}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            c.execute(f"SELECT summary FROM customersCalls WHERE recordingID = '{str(item)}'")
            summary = (c.fetchone()[0])
            with st.expander("Summary", expanded=False):
                st.markdown(
                    f"""
                    <div class="summary-container">
                        {summary}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            ##Sentiment Analysis Data
            ################################
            
            sentiment = pd.read_sql_query(f"select * from callsRecords where recordingID = {str(item)}", conn)
            sentiment = sentiment.astype({'sentiment_score_pos':'float',
                                          'sentiment_score_neu':'float',
                                          'sentiment_score_neg':'float',
                                          'sentiment_score_overall':'float',
                                          'toxicity':'float',
                                          'severe_toxicity':'float',
                                          'obscene':'float',
                                          'threat':'float',
                                          'insult':'float',
                                          'identity_attack':'float',
                                          
                                          })
            with st.expander("Sentiment Analysis", expanded=False):
                
                sentiment_scores = sentiment.groupby('speaker')['sentiment_score_overall']
                sentiment_speakers = list(sentiment_scores.groups.keys())

                fig_A = gauge_sentiment_plot(sentiment_scores.mean()[sentiment_speakers[0]],speaker=sentiment_speakers[0])
                fig_B = gauge_sentiment_plot(sentiment_scores.mean()[sentiment_speakers[1]],speaker=sentiment_speakers[1])
                

                st.pyplot(fig_A)
                st.pyplot(fig_B)

                negative_sentences = sentiment[sentiment['sentiment_score_overall']< -0.45]
                if negative_sentences.empty== False:
                    st.subheader("Highly negative sentences:")
                for index,row in negative_sentences.iterrows():
                        st.markdown(f"""<div><b>&#x2022;Speaker: {row['speaker']} - Sentiment Score: {row['sentiment_score_overall']:.2f}:</b><br>  {row['text']}</div>""",
                        unsafe_allow_html=True)



             ##########Toxicity###########
            #st.dataframe(sentiment)
            toxic_scores = sentiment[(sentiment['toxicity'] > 0.4) | ( sentiment['insult'] > 0.4) |  (sentiment['obscene'] > 0.4) |  (sentiment['threat'] > 0.4)]

            if not toxic_scores.empty:
                with st.expander("Toxicity Analysis", expanded=False):

                    for index,row in toxic_scores.iterrows():
                        st.write("Speaker "+ row['speaker'] +": "+ row['text'])
                        fig1 = linear_gauge("Toxicity", row['toxicity']*100)
                        st.plotly_chart(fig1,use_container_width=True, theme=None)
                        fig2 = linear_gauge("Insult", row['insult']*100)
                        st.plotly_chart(fig2,use_container_width=True, theme=None)
                        fig3 = linear_gauge("Threat", row['threat']*100)
                        st.plotly_chart(fig3,use_container_width=True, theme=None)

            # Bubble plots of word frequency
            speaker_counts = Counter(sentiment['speaker'])
            speakerA_text = sentiment[sentiment['speaker'] == list(speaker_counts.keys())[0]]['text']
            speakerA = ' '.join(speakerA_text)

            speakerB_text = sentiment[sentiment['speaker'] == list(speaker_counts.keys())[1]]['text']
            speakerB = ' '.join(speakerB_text)


            with st.expander("Word Frequecy", expanded=False):
                plot_bubble_chart(speakerA, list(speaker_counts.keys())[0])
                plot_bubble_chart(speakerB,list(speaker_counts.keys())[1])

    

conn.close()