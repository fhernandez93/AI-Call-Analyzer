
# Import necessary libraries
import streamlit as st
import pandas as pd
import os
import voice_recognition_module as vr
from tempfile import NamedTemporaryFile
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np 
import plotly.graph_objects as go
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import sqlite3
from datetime import datetime
import time
from functions import * 


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

conn = sqlite3.connect('OPTcallsAnalytics.db')
c = conn.cursor()
if 'viewPage' not in st.session_state:
    st.session_state.viewPage = "first"
##Filters section 

if st.session_state.viewPage == "first":
    col1, col2,col3 = st.columns(3)
    # Fetch client names from the Customers table
    c.execute("SELECT name FROM Customers")
    client_names = [item[0] for item in c.fetchall()]
    client_names = client_names

    # Dropdown to select a client
    selected_client = selectbox_with_default('Client', client_names, col=col1)

    #
    ## Dropdown to select a date
    if selected_client ==DEFAULT:
        c.execute("SELECT Date FROM customersCalls")
    else:
        c.execute(f"SELECT Date FROM customersCalls where name = '{selected_client}'")
    dates = [item[0] for item in c.fetchall()]
    selected_date = selectbox_with_default('Date', dates, col=col2)


    ################
    ##table with all records 
    if selected_client == DEFAULT and selected_date == DEFAULT:
        customersCalls = f"SELECT name, recordingID, date from customersCalls"
    elif selected_client!= DEFAULT and selected_date!= DEFAULT: 
        customersCalls = f"SELECT name, recordingID, date from customersCalls where name = '{selected_client}' and date = '{selected_date}'"
    elif selected_client!= DEFAULT:
        customersCalls = f"SELECT name, recordingID, date from customersCalls where name = '{selected_client}'"
    elif selected_date!= DEFAULT: 
        customersCalls = f"SELECT name, recordingID, date from customersCalls where date = '{selected_date}'"



    df = pd.read_sql_query(customersCalls, conn)

    df = df.rename(columns={"name":"Client", "date":"Date"})

    st.text("Records on file")

    #st.table(df.assign(hack='').set_index('hack'))
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

            
            sentiment = pd.read_sql_query(f"select * from callsRecords where recordingID = {str(item)}", conn)
            sentiment = sentiment.astype({'sentiment_score_pos':'float',
                                          'sentiment_score_neu':'float',
                                          'sentiment_score_neg':'float',
                                          
                                          })

            with st.expander("Sentiment Analysis", expanded=False):
                st.write("""Using a model trained on Twitter data, we analyze each sentence in a conversation and assign a sentiment score ranging from 0 to 1. 
                         For example, a highly negative sentence might receive scores like: neg:0.8, pos:0.2, and neu:0. 
                         The bars in our visual representation indicate the average sentiment score for each speaker in the conversation. Meanwhile, the error bars show the range or variation in sentiment scores for those speakers. 
                         """)
#
                    # Calculate the meansx
                mean_pos = sentiment.groupby('speaker')['sentiment_score_pos'].mean()
                mean_neu = sentiment.groupby('speaker')['sentiment_score_neu'].mean()
                mean_neg = sentiment.groupby('speaker')['sentiment_score_neg'].mean()

                # Calculate standard errors
                std_error_pos = sentiment.groupby('speaker')['sentiment_score_pos'].std() / np.sqrt(sentiment.groupby('speaker').size())
                std_error_neu = sentiment.groupby('speaker')['sentiment_score_neu'].std() / np.sqrt(sentiment.groupby('speaker').size())
                std_error_neg = sentiment.groupby('speaker')['sentiment_score_neg'].std() / np.sqrt(sentiment.groupby('speaker').size())

                # Positive Sentiment Plot with Error Bars
                fig_pos = go.Figure()
                fig_pos.add_trace(go.Bar(x=mean_pos.index, y=mean_pos, 
                                         error_y=dict(type='data', array=std_error_pos, visible=True),
                                         marker_color=px.colors.qualitative.Vivid))
                fig_pos.update_layout(title="Positive Sentiment", template="plotly_white")
                st.plotly_chart(fig_pos)

                # Neutral Sentiment Plot with Error Bars
                fig_neu = go.Figure()
                fig_neu.add_trace(go.Bar(x=mean_neu.index, y=mean_neu, 
                                         error_y=dict(type='data', array=std_error_neu, visible=True),
                                         marker_color=px.colors.qualitative.Vivid))
                fig_neu.update_layout(title="Neutral Sentiment", template="plotly_white")
                st.plotly_chart(fig_neu)

                # Negative Sentiment Plot with Error Bars
                fig_neg = go.Figure()
                fig_neg.add_trace(go.Bar(x=mean_neg.index, y=mean_neg, 
                                         error_y=dict(type='data', array=std_error_neg, visible=True),
                                         marker_color=px.colors.qualitative.Vivid))
                fig_neg.update_layout(title="Negative Sentiment", template="plotly_white")
                st.plotly_chart(fig_neg)

            # Bubble plots of word frequency
            speaker_counts = Counter(st.session_state.sentiment['speaker'])
            speakerA_text = st.session_state.sentiment[st.session_state.sentiment['speaker'] == list(speaker_counts.keys())[0]]['text']
            speakerA = ' '.join(speakerA_text)

            speakerB_text = st.session_state.sentiment[st.session_state.sentiment['speaker'] == list(speaker_counts.keys())[1]]['text']
            speakerB = ' '.join(speakerB_text)


            with st.expander("Word Frequecy", expanded=False):
                plot_bubble_chart(speakerA, list(speaker_counts.keys())[0])
                plot_bubble_chart(speakerB,list(speaker_counts.keys())[1])

    

conn.close()