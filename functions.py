import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd
import plotly.graph_objects as go

DEFAULT = '< select an option >'

def selectbox_with_default(text, values, default=DEFAULT, sidebar=False, col = None):
    if col!= None:
        func = col.sidebar.selectbox if sidebar else col.selectbox
    else:
        func = st.sidebar.selectbox if sidebar else st.selectbox
    return func(text, np.insert(np.array(values, object), 0, default))


def dataframe_with_selections(df):
    df_with_selections = df.copy()
    df_with_selections.insert(0, "Select", False)
    edited_df = st.data_editor(
        df_with_selections,
        hide_index=True,
        column_config={"Select": st.column_config.CheckboxColumn(required=True)},
        disabled=df.columns,
    )
    selected_indices = list(np.where(edited_df.Select)[0])
    selected_rows = df[edited_df.Select]
    return {"selected_rows_indices": selected_indices, "selected_rows": selected_rows}

def gauge_sentiment_plot(mean_score,std = 0, speaker = 'Not Identified'):

    colors = ['#4dab6d', "#72c66e", "#c1da64", "#f6ee54", "#fabd57", "#f36d54", "#ee4d55"]

    values = np.around(np.linspace(1,-1,8),2)

    x_axis_vals = [0, 0.44, 0.88,1.32,1.76,2.2,2.64]

    fig = plt.figure(figsize=(12,12))

    ax = fig.add_subplot(projection="polar")

    ax.bar(x=x_axis_vals, width=0.5, height=0.5, bottom=2,
           linewidth=3, edgecolor="white",
           color=colors, align="edge")

    plt.annotate("Highly Positive", xy=(0.16,2.1), rotation=-75, color="black", fontweight="bold")
    plt.annotate("Positive", xy=(0.65,2.08), rotation=-55, color="black", fontweight="bold")
    plt.annotate("Fairly Positive", xy=(1.14,2.1), rotation=-32, color="black", fontweight="bold")
    plt.annotate("Neutral", xy=(1.62,2.2), color="black", fontweight="bold")
    plt.annotate("Fairly Negative", xy=(2.08,2.25), rotation=20, color="black", fontweight="bold")
    plt.annotate("Negative", xy=(2.46,2.25), rotation=45, color="black", fontweight="bold")
    plt.annotate("Highly Negative", xy=(3.0,2.25), rotation=75, color="black", fontweight="bold")

    for loc, val in zip([0, 0.44, 0.88,1.32,1.76,2.2,2.64, 3.14], values):
        plt.annotate(val, xy=(loc, 2.5), ha="right" if val<=20 else "left")

    plt.annotate(str(np.around(mean_score,2)) + '$\pm$' + str(np.around(std,2)), xytext=(0,0), xy=((-0.5*mean_score+0.5)*np.pi, 2.5),
                 arrowprops=dict(arrowstyle="wedge, tail_width=0.5", color="black", shrinkA=0),
                 bbox=dict(boxstyle="circle", facecolor="black", linewidth=2.0, ),
                 fontsize=45, color="white", ha="center"
                )


    plt.title(f"Speaker: {speaker}", loc="center", pad=20, fontsize=35, fontweight="bold")

    ax.set_axis_off()
   

    return fig


def identify_speaker(transcription):
        # Initialize scores for each speaker
        speaker_A_score = 0
        speaker_B_score = 0

        # Regex patterns to identify agent-like or customer-like phrases
        agent_patterns = [
            r"how may i assist you",
            r"my name is",
            r"welcome to",
            r"thank you for calling",
            r"is there anything else",
            r"how can i help you",
            r"have an excellent day",
            r"have a good day",
            r"can i have the patient's date of birth",
            r"would you be a new patient, or established",
            r"who is your doctor",
            r"what day works best for you",
            r"can I have your phone number"

        ]

        caller_patterns = [
            r"i need help with",
            r"i have a problem",
            r"can you help me",
            r"i'm calling about",
            r"how do i"
        ]

        # Loop through each line in the transcription
        for line in transcription:
            speaker, text = line['speaker'], line['text'].lower()

            # Score speaker A and speaker B based on phrases used
            for pattern in agent_patterns:
                if re.search(pattern, text):
                    if speaker == "A":
                        speaker_A_score += 1
                    else:
                        speaker_B_score += 1

            for pattern in caller_patterns:
                if re.search(pattern, text):
                    if speaker == "A":
                        speaker_B_score += 1
                    else:
                        speaker_A_score += 1

        print(speaker_A_score)
        # Determine who is the agent and who is the caller based on the scores
        if speaker_A_score > speaker_B_score:
            return {"A": "Agent", "B": "Caller"}
        elif speaker_B_score > speaker_A_score:
            return {"A": "Caller", "B": "Agent"}
        else:
            return {"A": "Unknown", "B": "Unknown"}
        

#Function to replace speaker labels 
def replace_speakers(df=pd.DataFrame(), replacements = {}):
    old_data = df.to_dict('records')
    new_data = np.array([{k: replacements.get(v, v) if k == 'speaker' else v for k, v in item.items()} for item in old_data])
    return pd.json_normalize(new_data)


def linear_gauge(title,val):
    fig = go.Figure(go.Indicator(
        mode = "gauge", value = val,
        domain = {'x': [0, 1], 'y': [0, 1]},
        delta = {'reference': 10, 'position': "top"},
        title = {'text':f"<b>{title}</b><br><span style='color: gray; font-size:0.8em'>"+f"{val:.2f}"+"<br></span>", 'font': {"size": 12, "color": "black"}},
        gauge = {
            'shape': "bullet",
            'axis': {'range': [0, 100], 'tickcolor': "black", 'tickfont': {'color': "black", 'size': 10}},  # setting tickfont color here
            'bgcolor': "white",
            'steps': [
                {'range': [0, 100], 'color': "#ec7f79"},
                {'range': [0, 66], 'color': "#f1f192"},
                {'range': [0, 33], 'color': "#81c051"}],
            'bar': {'color': "#355586"}}))
    fig.update_layout(height = 250, width=680, paper_bgcolor='white',  # this sets the entire figure background
                    plot_bgcolor='white')
    
    return fig