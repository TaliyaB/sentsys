import pandas as pd
import os
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
from  nltk.corpus import stopwords
from  nltk.tokenize import word_tokenize
import spacy
spacy.cli.download("en_core_web_md")
import en_core_web_md

class TextProcessor():
    def __init__(self):
        self.df_parsed_data = []
        self.dict_respondents_block = None
        self.dict_respondents_course = {'BSIT':0 , 'BSCS': 0, 'BSIS':0, 'BSIT-Animation':0}

        self.int_respondents_count = None
        self.text_lemmatized = None
        self.text_removed_stopwords = None
        self.text_cleaned = None

        self.list_texts = []
        self.list_question_number = []
        self.list_tokens_per_text = []
        self.df_for_sentiment_analysis = None

        self.list_tokens_per_question = []

        #final df to be passed
        self.df_final_data_for_sentiment_analysis = None

        #dummy placeholder
        self.df_dummy = None

    def parse_respondents(self, df):
        #count respondents per block
        self.df_dummy = df['Year & Course']
        self.dict_respondents_block = self.df_dummy.value_counts().to_dict()
        #number of respondents
        self.int_respondents_count = len(self.df_dummy)
        #count respondents per course
        for i in range(len(self.df_dummy)):
            print(self.df_dummy[i])
            if 'bsit' in self.df_dummy[i].lower() and 'animation' not in self.df_dummy[i].lower():
                self.dict_respondents_course['BSIT']+=1
            if 'bscs' in self.df_dummy[i].lower():
                self.dict_respondents_course['BSCS']+=1
            if 'bsis' in self.df_dummy[i].lower():
                self.dict_respondents_course['BSIS']+=1
            if 'animation' in self.df_dummy[i].lower():
                self.dict_respondents_course['BSIT-Animation']+=1
        self.df_dummy = None
        return

    def remove_stopwords(self, text):
        stopwords = set(nltk.corpus.stopwords.words('english'))
        words : list[str] = nltk.word_tokenize(text)
        words = [w for w in words if w.lower() not in stopwords and w.isalnum()]
        return words

    def tokens_per_question(self, list):
        tmp_list = []
        for text in list:
            words = self.remove_stopwords(text)
            self.list_tokens_per_text.append(words)
            for w in words:
                tmp_list.append(w)
        self.list_tokens_per_question.append(tmp_list)
        return

    def parse_texts_for_sentiment_analysis(self):
        data = self.df_parsed_data
        for i in range(len(data)):
            text_list = data[i].tolist()
            q_number = i+1
            for txt in text_list:
                self.list_texts.append(txt)
                self.list_question_number.append(q_number)
            self.tokens_per_question(text_list)
        return

    def construct_final_df_for_sentiment_analysis(self):
        cols = ['Text' , 'Tokens', 'Question Number']
        self.df_final_data_for_sentiment_analysis = pd.DataFrame(
            list(zip(self.list_texts, self.list_tokens_per_text, self.list_question_number)),
            columns=cols
        )
        return

    def parser(self,path):
        # get dataframe for each question and append to list
        data = pd.read_csv(path)
        n = self.int_respondents_count
        #answers to questions
        for i in range(4,len(data.columns)):
            self.df_parsed_data.append(data[data.columns[i]])
        #parse respondents data
        self.parse_respondents(data)
        self.parse_texts_for_sentiment_analysis()
        #construct final dataframe
        self.construct_final_df_for_sentiment_analysis()

    def convert_to_csv(self, output_filename):
        return
