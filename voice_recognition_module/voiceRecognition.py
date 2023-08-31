import assemblyai as aai
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import json
import numpy as np
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
nltk.download('punkt')
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
from collections import Counter
from transformers import BartForConditionalGeneration, BartTokenizer, pipeline, AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
import sqlite3
import streamlit as st
import re


conn = sqlite3.connect('OPTcallsAnalytics.db')
cursor = conn.cursor()

class voiceTranscription:
    """
        This package allows to retrieve relevant parameters from call recordings using AssemblyAI API 
    """
    def __init__(self, file_name = ""):
        self.file_name = file_name
        self.api_key = st.secrets.KEY
       

    def getTranscriptionObject(self): 
        aai.settings.api_key = self.api_key
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(self.file_name,config=aai.TranscriptionConfig(speaker_labels=True))
        return transcript
    
    def getSpeakersArray(self):
        speakers_array = []
        utterances = self.getTranscriptionObject().utterances
        for utterance in utterances:
            speakers_array += [{"speaker":utterance.speaker, "confidence":utterance.confidence, "text":utterance.text}]

        speakers_array = np.array(speakers_array)

        return speakers_array
    
    #This function returns sentences from speakers that appear the most. Hopefully this will get rid of und
    def cleanSpeakers(self):
        array = self.getSpeakersArray()
        speaker_counts = (Counter(d['speaker'] for d in array)).most_common(2)
        cleaned_speakers = np.array([i for i in array if i['speaker'] == speaker_counts[0][0] or i['speaker'] == speaker_counts[1][0]] )

        return np.array(cleaned_speakers)
    
    #Speaker identification (We should be able to implement a classification neural network once we have enough data to train a model)
    def identify_speaker(self):
        transcription = self.cleanSpeakers()
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
            r"may i have your date of birth",
            r"i'm still here with you",
            r"your appointment",
            r"the callback number i have",
            r"how can i assist you",
            r"first and last name",
            r"reason for the call",
            r"do you prefer morning or afternoon",
            r"your email address",
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

        replacements = {"A": "", "B": ""}

        # Determine who is the agent and who is the caller based on the scores
        if speaker_A_score > speaker_B_score:
            replacements =  {"A": "Agent", "B": "Caller"}
        elif speaker_B_score > speaker_A_score:
            replacements = {"A": "Caller", "B": "Agent"}
        
        if replacements["A"] != '': 
            new_data = np.array([{k: replacements.get(v, v) if k == 'speaker' else v for k, v in item.items()} for item in transcription])
            return new_data
        else:
            return transcription

    
    
    #Returns a clean text for the call without additional detected speakers 
    def cleaned_string(self,rerun_array:list = []): 
        if rerun_array:
            array = rerun_array
        else:
            array = self.identify_speaker()
        
        return '\n'.join([d["speaker"]+': '+d["text"] for d in array])
    
    def cleaned_string_for_summary(self,rerun_array:list = []): 
        if rerun_array:
            array = rerun_array
        else:
            array = self.identify_speaker()
        return '\n'.join([d["text"] for d in array])

    #This is an intelligent solution, but it's limited by the length of the text 
    def bart_summarize(self,rerun_array:list = []) -> str:
        if rerun_array:
            text = self.cleaned_string_for_summary(rerun_array)
        else:
            text = self.cleaned_string_for_summary()
        # Load pre-trained BART model and tokenizer for summarization
        model_name = "facebook/bart-large-cnn"
        model = BartForConditionalGeneration.from_pretrained(model_name)
        tokenizer = BartTokenizer.from_pretrained(model_name)

        # Using Hugging Face's pipeline for summarization
        summarizer = pipeline(task="summarization", model=model, tokenizer=tokenizer , truncation = True)
        # Summarize the text
        summary = summarizer(text, min_length=40, length_penalty=0.6)[0]['summary_text']

        return summary
    
    #This is a less fancy solution, but it's faster and works with longer text
    @staticmethod
    def summarize_from_text(text, per):
        nlp = spacy.load('en_core_web_sm')
        doc= nlp(text)
        tokens=[token.text for token in doc]
        word_frequencies={}
        for word in doc:
            if word.text.lower() not in list(STOP_WORDS):
                if word.text.lower() not in punctuation:
                    if word.text not in word_frequencies.keys():
                        word_frequencies[word.text] = 1
                    else:
                        word_frequencies[word.text] += 1
        max_frequency=max(word_frequencies.values())
        for word in word_frequencies.keys():
            word_frequencies[word]=word_frequencies[word]/max_frequency
        sentence_tokens= [sent for sent in doc.sents]
        sentence_scores = {}
        for sent in sentence_tokens:
            for word in sent:
                if word.text.lower() in word_frequencies.keys():
                    if sent not in sentence_scores.keys():                            
                        sentence_scores[sent]=word_frequencies[word.text.lower()]
                    else:
                        sentence_scores[sent]+=word_frequencies[word.text.lower()]
        select_length=int(len(sentence_tokens)*per)
        summary=nlargest(select_length, sentence_scores,key=sentence_scores.get)
        final_summary=[word.text for word in summary]
        summary=''.join(final_summary)
        return summary

    
    ##Sentiment analysis using Roberta pretained model 
    ##We need to pass the getSpeakersArray to calculate polarity scores on each sentence 

    def getFullSentimentSpeakersArray(self):
        MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        speakers_array = self.identify_speaker()
        for i, item in enumerate(speakers_array):
            encoded_text = tokenizer(item['text'],return_tensors='pt')
            output = model(**encoded_text)
            scores = np.array(output[0][0].detach())
            scores = softmax(scores)
            scores_dict = {
                'neg':scores[0],
                'neu':scores[1],
                'pos':scores[2], 
                'overall': float(scores[2]) - float(scores[0])
                }
            speakers_array[i]['sentiment_score'] = scores_dict

        
        return speakers_array
    
 

    
