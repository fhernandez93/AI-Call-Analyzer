# Streamlit app for a healthcare tech company

# Import necessary libraries
import streamlit as st
import pandas as pd
import os
import voice_recognition_module as vr
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np 
import plotly.graph_objects as go
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import pyodbc
from datetime import datetime
import time
from authetication.add_user import *
from authetication.login import *
from functions import *
from sqlalchemy import create_engine

nltk.download('punkt')
nltk.download('stopwords')
#try:
#    nltk.data.find('tokenizers/punkt')
#except:
#    nltk.download('punkt')
#
#try:
#    nltk.data.find('tokenizers/stopwords')
#except:
#    nltk.download('stopwords')

extension = ''


if 'tempKey' not in st.session_state:
        st.session_state.tempKey = ""

if 'access' not in st.session_state:
        st.session_state.access = False

if 'login' not in st.session_state:
        st.session_state.login = True

if 'secureLogin' not in st.session_state:
        st.session_state.secureLogin = False

if 'qrSecurity' not in st.session_state:
        st.session_state.qrSecurity = False

if 'success_signup' not in st.session_state:
        st.session_state.success_signup = False

if not st.session_state.access:
    st.set_page_config(page_title="Home",initial_sidebar_state="collapsed",page_icon=':bar_chart:' )
    st.markdown(
            """
        <style>
            [data-testid="collapsedControl"] {
                display: none
            }
        </style>
        """,
            unsafe_allow_html=True,
        )
    if st.session_state.login:
        if st.session_state.secureLogin:
             secure_login()
        else:
            login_screen()
    else:
        if st.session_state.qrSecurity:
            if st.session_state.success_signup:
                successQR()
            else:
                addQR()
        else:
            runAddUser()
else:
    
    st.set_page_config(page_title="Call Analyzer",page_icon=':bar_chart:' )

    
    #Database connection 
    #conn = sqlite3.connect('OPTcallsAnalytics.db')
    Driver="Driver={ODBC Driver 18 for SQL Server};Server=tcp:opt-call-analyzer-server.database.windows.net,1433;Database=OPTCallsAnalytics;Uid="+str(SQLUSER)+";Pwd={"+str(SQLPASS)+"};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=250;"
    conn = pyodbc.connect(Driver)
    c = conn.cursor()


    ####Functions########
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

    def check_file_exists(name):
        """Function to check if a file exists in the database by name."""
        c.execute("SELECT COUNT(*) FROM customersCalls WHERE cast(fileName as nvarchar(max))=?", (name,))
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
    if st.sidebar.button("Log Out"):
        st.session_state.login = True
        st.session_state.qrSecurity = False
        st.session_state.success_signup = False
        st.session_state.secureLogin = False
        st.session_state.access = False
        st.session_state.tempKey = ""
        st.session_state["authentication_status"] = False
        st.cache_resource.clear()
        st.cache_data.clear()
        st.runtime.legacy_caching.clear_cache()
        st.experimental_singleton.clear()
        st.experimental_rerun()


         
    st.sidebar.header('Upload your Data')

    

    if 'sentiment' not in st.session_state:
        st.session_state.sentiment = []
    if 'transcription' not in st.session_state:
        st.session_state.transcription = []
    if 'summary' not in st.session_state:
        st.session_state.summary = []
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = []
    if 'call_type' not in st.session_state:
        st.session_state.call_type = ""
    if 'file_uploader_key' not in st.session_state:
        st.session_state.file_uploader_key = 0
    if 'save_call' not in st.session_state:
        st.session_state.save_call = ""
    if 'fixLabel_form' not in st.session_state:
        st.session_state.fixLabel_form = []

    val_key="test"
    if st.button("Clear uploaded files"):
            st.session_state["file_uploader_key"] += 1
            st.session_state.uploaded_file = ""
            st.experimental_rerun()

    def process_data(file_item,index):
       
        
        st.header(file_item.name)
        # If a file is uploaded, process it
        if file_item:
            #check if file has not been already processed
            if check_file_exists(file_item.name):
                st.success('This file is on database, check historical section', icon="✅")
                #if st.button("Clear uploaded files"):
                #    st.session_state["file_uploader_key"] += 1
                #    st.session_state.uploaded_file = ""
                #    st.experimental_rerun()
                return ""
            else:
                extension = os.path.splitext(file_item.name)[1]
                with open("./static/temp_file"+extension, "wb") as f:
                        f.write(file_item.read())

                try:
                    x = st.session_state.summary[index]
                except:
                    if testing: 
                        st.session_state.sentiment += [pd.read_csv('sentiment.csv')]
                        with open("transcript.txt","r") as f: 
                            st.session_state.transcription  += [f.read()]
                        with open("transcript_summary.txt","r") as f: 
                            st.session_state.summary += [f.read()]

                    else: 
                        #####Relevant functions for transcription######

                        transcriptionClass = vr.voiceTranscription("./static/temp_file"+extension)

                        # Once done, remove the temporary file if you wish
                        ###########
                        st.session_state.transcription +=[(transcriptionClass.cleaned_string()).replace('\n','<br>')]
                        st.session_state.summary += [(transcriptionClass.bart_summarize() + '<br><br>' + transcriptionClass.summarize_from_text(st.session_state.transcription[index],0.1)).replace('\n','<br>')]
                        st.session_state.sentiment += [pd.json_normalize(transcriptionClass.getFullSentimentSpeakersArray())]
                

                ##Save to DB functionality   

                if 'save_mode' not in st.session_state:
                    st.session_state.save_mode = []
                try:
                    x=st.session_state.save_mode[index]
                except:
                    st.session_state.save_mode+=[False]

                col1,col2=st.columns(2)
                if col1.button("Save "+str(index)):
                    st.session_state.save_mode[index] =True

                if col2.button("Cancel "+str(index)): 
                    st.session_state.save_mode[index] = False


                if st.session_state.save_mode[index]:

                    st.write("Select client and enter employee, then press Done")

                    # Fetch client names from the Customers table
                    c.execute("SELECT name FROM Customers")
                    client_names = [item[0] for item in c.fetchall()]

                    c.execute("SELECT name FROM Employees")
                    employees = [item[0] for item in c.fetchall()]

                    call_types = ["Appointment Scheduling","Billing Question","Prescription Call", "Message for Provider"]

                    def save_data(selected_client,employee_name,call_type):
                        #st.write(f'{selected_client} {employee_name} {call_type}')
                        #return ""
                        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        if testing:
                            st.write(("321312",selected_client,current_time,call_type,employee_name,"vdsfswfwegergver",st.session_state.uploaded_file[index].name, st.session_state.transcription[index], st.session_state.summary[index]))
                        else:
                            df =  st.session_state.sentiment[index]
                            df = df.rename(columns={'sentiment_score.neg':'sentiment_score_neg'
                                                    ,'sentiment_score.neu':'sentiment_score_neu'
                                                    ,'sentiment_score.pos':'sentiment_score_pos'
                                                    ,'sentiment_score.overall':'sentiment_score_overall'
                                                    })
                            c.execute("SELECT clientID FROM Customers WHERE cast(name as nvarchar(max))=?", (selected_client,))
                            client_id = str(int(c.fetchone()[0]))
                            recordingID = str(int(generateID()))
                            df['recordingID'] = (recordingID)
                            df['clientID'] = (client_id)
                            df['date'] = current_time
                            df[df.columns] = df[df.columns].astype(str)
                        
                            connection_str= (
                                "mssql+pyodbc://"+str(SQLUSER)+":"+str(SQLPASS)+"@opt-call-analyzer-server.database.windows.net:1433/OPTCallsAnalytics?"
                                "driver=ODBC+Driver+18+for+SQL+Server")

                            engine = create_engine(connection_str)
                            df.to_sql('callsRecords', engine, if_exists='append', index=False)
                            c.execute("INSERT INTO customersCalls (clientID,name,date,callType,EmployeeName,recordingID,fileName, cleanTranscription, Summary) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", 
                                    (client_id,selected_client,current_time,call_type,employee_name,recordingID,st.session_state.uploaded_file[index].name, st.session_state.transcription[index], st.session_state.summary[index]))
                            conn.commit()
                            st.session_state.transcription[index] = ""
                            st.session_state.summary[index] = ""
                            #st.session_state.uploaded_file = []
                            #st.session_state.call_type = ""
                            st.session_state.save_mode[index] = False
                            st.experimental_rerun()

                    client=""
                    name=""
                    t_type=""

                    def set_form_state():
                        st.session_state.save_call=True

                    with st.form(key="testing_form",clear_on_submit=False):
                        # Dropdown to select a client
                        client = st.selectbox('Client', client_names)
                        name = st.selectbox('Employee Name', employees)
                        t_type = st.selectbox('Select call type',call_types)
                        #call_type = selectbox_with_default(f'Select call type', ["Appointment Scheduling","Billing Question","Prescription Call", "Message for Provider"],sidebar=False)

                        # Store selected client and current time in the customers_time table


                        st.form_submit_button('Done',on_click=set_form_state)


                    if st.session_state.save_call:
                        if client and name and t_type:
                            st.session_state.save_call=False
                            save_data(client,name,t_type)
                        else:
                            st.write("Test Failed!!!!")

                    #if st.sidebar.button("Done"):





                ######################################## 
                ##Accordions with all relevant information
                with st.expander("Clear Transcription", expanded=False):
                    if 'fixLabels' not in st.session_state:
                        st.session_state.fixLabels=[]
                    
                    try:
                        x=st.session_state.fixLabels[index]
                    except:
                        st.session_state.fixLabels+=[False]

                    try:
                        x=st.session_state.fixLabel_form[index]
                    except:
                        st.session_state.fixLabel_form+=[False]

                    st.markdown(
                        f"""
                        <div class="warning-container">
                           Please review the if the 'Agent' and 'Caller' labels were correctly assigned.
                            If not, you can fix this by clicking on the button below. This will help to train 
                            our model and have better identification in future updates.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    if not st.session_state.fixLabels[index]:
                        if st.button("Fix labels "+ str(index)):
                             st.session_state.fixLabels[index] = True
                             st.experimental_rerun()  


                        st.markdown(
                            f"""
                            <div class="summary-container">
                                {st.session_state.transcription[index]}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        summ1, summ2 = st.columns(2)
                        label_names = ['Caller', 'Agent']

                        group_speakers = st.session_state.sentiment[index].groupby('speaker')['sentiment_score.overall']
                        current_speakers = list(group_speakers.groups.keys())

                        def update_labels_state():
                            st.session_state.fixLabel_form[index] = True

                        def update_labels(lA,lB):
                            if (lA and lB) and (lA!='Select New Label' and lB!='Select New Label'): 
                                transcriptionFunctions = vr.voiceTranscription()
                                replacements = {current_speakers[0]:lA, current_speakers[1]:lB}
                                st.session_state.sentiment[index] = replace_speakers(st.session_state.sentiment[index],replacements)
                                st.session_state.transcription[index] =(transcriptionFunctions.cleaned_string(rerun_array =(st.session_state.sentiment[index]).to_dict('records'))).replace('\n','<br>')
                                #st.session_state.summary = (transcriptionFunctions.bart_summarize(rerun_array =(st.session_state.sentiment[index]).to_dict('records')) + '<br><br>' + transcriptionFunctions.summarize_from_text(st.session_state.transcription,0.1)).replace('\n','<br>')
                                st.session_state.fixLabels[index] = False
                                st.experimental_rerun()  
                            else:
                                 st.warning("Fill both label fields")

                        if st.button("Exit"):
                            st.session_state.fixLabels[index] = False
                            st.experimental_rerun()  
                        with st.form(key='labels_update'):
                            labelA = selectbox_with_default(f'Current label: {current_speakers[0]}', label_names, default='Select New Label')
                            labelB = selectbox_with_default(f'Current label: {current_speakers[1]}', label_names, default='Select New Label')
                            st.form_submit_button("Update",on_click=update_labels_state)

                        if st.session_state.fixLabel_form[index]:
                            st.session_state.fixLabel_form[index]=False
                            update_labels(labelA,labelB)








                with st.expander("Summary", expanded=False):
                    st.markdown(
                        f"""
                        <div class="summary-container">
                        {st.session_state.summary[index]}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                ##Sentiment Analysis Data
                ################################

                with st.expander("Sentiment Analysis", expanded=False):
                    scores_sentiment = st.session_state.sentiment[index]
                    sentiment_scores = scores_sentiment.groupby('speaker')['sentiment_score.overall']
                    sentiment_speakers = list(sentiment_scores.groups.keys())

                    fig_A = gauge_sentiment_plot(sentiment_scores.mean()[sentiment_speakers[0]],speaker=sentiment_speakers[0])
                    fig_B = gauge_sentiment_plot(sentiment_scores.mean()[sentiment_speakers[1]],speaker=sentiment_speakers[1])


                    st.pyplot(fig_A)
                    st.pyplot(fig_B)


                    negative_sentences = scores_sentiment[scores_sentiment['sentiment_score.overall']< -0.45]

                    if not negative_sentences.empty:
                        st.subheader("Highly negative sentences:")
                    for ind,row in negative_sentences.iterrows():
                            st.markdown(f"""<div><b>&#x2022;Speaker: {row['speaker']} - Sentiment Score: {row['sentiment_score.overall']:.2f}:</b><br>  {row['text']}</div>""",
                            unsafe_allow_html=True)

                    
                    
                    


                ##########Toxicity###########
                toxic_scores = st.session_state.sentiment[index]
                #st.dataframe(toxic_scores)
                toxic_scores = toxic_scores[(toxic_scores['toxicity'] > 0.3) | ( toxic_scores['insult'] > 0.3) |  (toxic_scores['obscene'] > 0.3) |  (toxic_scores['threat'] > 0.3)]

                if not toxic_scores.empty:
                    with st.expander("Toxicity Analysis", expanded=False):

                        for ind,row in toxic_scores.iterrows():
                            st.write("Speaker "+ row['speaker'] +": "+ row['text'])
                            fig1 = linear_gauge("Toxicity", row['toxicity']*100)
                            st.plotly_chart(fig1,use_container_width=True, theme=None)
                            fig2 = linear_gauge("Insult", row['insult']*100)
                            st.plotly_chart(fig2,use_container_width=True, theme=None)
                            fig3 = linear_gauge("Threat", row['threat']*100)
                            st.plotly_chart(fig3,use_container_width=True, theme=None)



                # Bubble plots of word frequency
                speaker_counts = Counter(st.session_state.sentiment[index]['speaker'])
                speakerA_text = st.session_state.sentiment[index][st.session_state.sentiment[index]['speaker'] == list(speaker_counts.keys())[0]]['text']
                speakerA = ' '.join(speakerA_text)

                speakerB_text = st.session_state.sentiment[index][st.session_state.sentiment[index]['speaker'] == list(speaker_counts.keys())[1]]['text']
                speakerB = ' '.join(speakerB_text)


                with st.expander("Word Frequecy", expanded=False):
                    plot_bubble_chart(speakerA, list(speaker_counts.keys())[0])
                    plot_bubble_chart(speakerB,list(speaker_counts.keys())[1])



                os.remove("./static/temp_file"+extension)
                #conn.close()

                # Add any other analytics you think would be relevant for your data.


    #########################RUN HERE#########################################
     # Add a file uploader to the sidebar
    uploaded_files = st.sidebar.file_uploader("Choose an audio file", type=['wav','mp3'],key=st.session_state["file_uploader_key"],accept_multiple_files=True)
    if not st.session_state.uploaded_file:
        st.session_state.uploaded_file = uploaded_files

    if len( st.session_state.uploaded_file) >3:
        st.warning('Only 5 files can be processed at a time', icon="❌")
        if st.button("Clear uploaded files"):
            st.session_state["file_uploader_key"] += 1
            st.session_state.uploaded_file = []
            st.session_state.save_mode = []
            st.experimental_rerun()
    else: 
        if len( st.session_state.uploaded_file) > 0:
            tabs = [item.name for item in  st.session_state.uploaded_file]
            tabs_list = st.tabs(tabs)

            for i, item in enumerate(tabs):
                with tabs_list[i]:
                    process_data( st.session_state.uploaded_file[i],i)





