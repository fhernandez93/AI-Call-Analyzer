# Streamlit app for a healthcare tech company

# Import necessary libraries
import streamlit as st
import pandas as pd
import os
import voice_recognition_module as vr
import whisper
from tempfile import NamedTemporaryFile
import json
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

nltk.download('stopwords')
nltk.download('punkt')
extension = ''

#Database connection 
conn = sqlite3.connect('OPTcallsAnalytics.db')
c = conn.cursor()


####Functions########
def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

def check_file_exists(name):
    """Function to check if a file exists in the database by name."""
    c.execute("SELECT COUNT(*) FROM customersCalls WHERE fileName=?", (name,))
    count = c.fetchone()[0]
    return count > 0

# Using the container's class for styling
local_css("style.css")

def generateID():
    return round(time.time() * 1000)

class JSONObject:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, JSONObject(value))
            else:
                setattr(self, key, value)

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

testing = False
##########################################


# Set the app title
st.title('Calls Analytics Dashboard')

st.write("The summary and transcription showed here are a clean version from the full transcription. We use AI to identify undesired speakers and sounds to show only relevant information of the uploaded call recording file.")

# Create a sidebar for user inputs and instructions
st.sidebar.header('Upload your Data')

# Add a file uploader to the sidebar
uploaded_file = st.sidebar.file_uploader("Choose an audio file", type=['wav','mp3'])


# If a file is uploaded, process it
if uploaded_file:
    #check if file has not been already processed
    if check_file_exists(uploaded_file.name):
        st.success('This file is on database, check historical section', icon="✅")

    else:
        extension = os.path.splitext(uploaded_file.name)[1]
        with open("./temp/temp_file"+extension, "wb") as f:
                f.write(uploaded_file.read())

        if 'sentiment' not in st.session_state:
            st.session_state.sentiment = pd.DataFrame()
        if 'transcription' not in st.session_state:
            st.session_state.transcription = ""
        if 'summary' not in st.session_state:
            st.session_state.summary = ""


        if st.session_state.summary == "":
            if testing: 
                st.session_state.sentiment = pd.read_csv('sentiment.csv')
                with open("transcript.txt","r") as f: 
                    st.session_state.transcription  = f.read()
                with open("transcript_summary.txt","r") as f: 
                    st.session_state.summary = f.read()

            else: 
                #####Relevant functions for transcription######

                transcriptionClass = vr.voiceTranscription("./temp/temp_file"+extension)

                # Once done, remove the temporary file if you wish
                ###########
                st.session_state.transcription =( transcriptionClass.cleaned_string()).replace('\n','<br>')
                st.session_state.summary = (transcriptionClass.bart_summarize() + '<br><br>' + transcriptionClass.summarize_from_text(st.session_state.transcription,0.1)).replace('\n','<br>')
                st.session_state.sentiment = pd.json_normalize(transcriptionClass.getFullSentimentSpeakersArray())

        ##Save to DB functionality   

        if 'save_mode' not in st.session_state:
            st.session_state.save_mode = False

        col1, col2 = st.sidebar.columns(2)

        if col1.button("Save"):
            st.session_state.save_mode = True

        if col2.button("Cancel"): 
            st.session_state.save_mode = False

        if st.session_state.save_mode:
            
            st.sidebar.write("Select the client and press Done")

            # Fetch client names from the Customers table
            c.execute("SELECT name FROM Customers")
            client_names = [item[0] for item in c.fetchall()]

            # Dropdown to select a client
            selected_client = st.sidebar.selectbox('Client', client_names)
            
            if st.sidebar.button("Done"):
                # Store selected client and current time in the customers_time table
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                df =  st.session_state.sentiment
                df = df.rename(columns={'sentiment_score.neg':'sentiment_score_neg','sentiment_score.neu':'sentiment_score_neu','sentiment_score.pos':'sentiment_score_pos'})
                c.execute("SELECT clientID FROM Customers WHERE name=?", (selected_client,))
                client_id = int(c.fetchone()[0])
                recordingID = int(generateID())
                df['recordingID'] = recordingID
                df['clientID'] = client_id
                df['date'] = current_time
                df.to_sql('callsRecords', conn, if_exists='append', index=False)
                c.execute("INSERT OR IGNORE INTO customersCalls (clientID,name,date,recordingID,fileName, cleanTranscription, Summary) VALUES (?, ?, ?, ?, ?, ?, ?)", 
                          (client_id,selected_client,current_time,recordingID,uploaded_file.name, st.session_state.transcription, st.session_state.summary))
                conn.commit()

                st.session_state.save_mode = False
                st.experimental_rerun()
                    



        ######################################## 

        ##Accordions with all relevant information
        with st.expander("Clear Transcription", expanded=False):
            st.markdown(
                f"""
                <div class="summary-container">
                    {st.session_state.transcription}
                </div>
                """,
                unsafe_allow_html=True
            )

        with st.expander("Summary", expanded=False):
            st.markdown(
                f"""
                <div class="summary-container">
                   {st.session_state.summary}
                </div>
                """,
                unsafe_allow_html=True
            )

        ##Sentiment Analysis Data
        ################################

        with st.expander("Sentiment Analysis", expanded=False):
            st.write("""Using a model trained on Twitter data, we analyze each sentence in a conversation and assign a sentiment score ranging from 0 to 1. 
                     For example, a highly negative sentence might receive scores like: neg:0.8, pos:0.2, and neu:0. 
                     The bars in our visual representation indicate the average sentiment score for each speaker in the conversation. Meanwhile, the error bars show the range or variation in sentiment scores for those speakers. 
                     """)

            # Calculate the meansx
            mean_pos = st.session_state.sentiment.groupby('speaker')['sentiment_score.pos'].mean()
            mean_neu = st.session_state.sentiment.groupby('speaker')['sentiment_score.neu'].mean()
            mean_neg = st.session_state.sentiment.groupby('speaker')['sentiment_score.neg'].mean()

            # Calculate standard errors
            std_error_pos = st.session_state.sentiment.groupby('speaker')['sentiment_score.pos'].std() / np.sqrt(st.session_state.sentiment.groupby('speaker').size())
            std_error_neu = st.session_state.sentiment.groupby('speaker')['sentiment_score.neu'].std() / np.sqrt(st.session_state.sentiment.groupby('speaker').size())
            std_error_neg = st.session_state.sentiment.groupby('speaker')['sentiment_score.neg'].std() / np.sqrt(st.session_state.sentiment.groupby('speaker').size())

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



        os.remove("./temp/temp_file"+extension)
        conn.close()

        # Add any other analytics you think would be relevant for your data.
    




