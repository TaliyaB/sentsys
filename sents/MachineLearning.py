import pandas as pd
import os
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
#nltk.download('vader_lexicon')
import spacy
spacy.cli.download("en_core_web_md")
import en_core_web_md
#from sentiments import Visualizer
from nltk.sentiment import SentimentIntensityAnalyzer
import collections
from statistics import mean
import gensim
from gensim import  corpora
import pyLDAvis.gensim_models
import  pickle
import numpy as np

class MachineLearning():
    def __init__(self, df_for_sentiment_analysis, tokens_per_question, respondents_by_course, respondents_by_block, total_respondents):
        #inputs
        self.input_df_for_sentiment_analysis = df_for_sentiment_analysis
        self.input_tokens_per_question = tokens_per_question
        self.dict_distribution_of_respondents_per_block = respondents_by_block
        self.dict_distribution_of_respondents_per_course = respondents_by_course
        self.int_total_respondents = total_respondents

        self.df_top_n_adjectives = []
        self.df_top_n_nouns = []

        self.df_sentiment_analysis_result = None
        self.df_positive_predicted_sentiments = None
        self.df_negative_predicted_sentiments = None

        self.list_freq_pos_sentiments_per_question = None
        self.list_freq_neg_sentiments_per_question = None

        self.lda_model = None
        self.dtm_tfidf = None
        self.tfidf_vectorizer = None

    def distribution_of_respondents_per_block(self):
        return self.dict_distribution_of_respondents_per_block

    def distribution_of_respondents_per_course(self):
        return self.dict_distribution_of_respondents_per_course

    def top_n_nouns(self, tokens_per_question, n):
        pos_tag = ['N', 'NN', 'NNS', 'NNP', 'NNPS']
        nouns = []
        tagged_words = nltk.pos_tag(tokens_per_question)
        for (txt, tag) in tagged_words:
            if  (tag in pos_tag):
                nouns.append(txt)
        freq_dist = nltk.FreqDist(n for n in nouns)
        cols = ['Noun' , 'Frequency']
        df_top_n_nouns = pd.DataFrame(freq_dist.most_common(n), columns=cols)
        self.df_top_n_nouns.append(df_top_n_nouns)
        return

    def top_n_adjectives(self, tokens_per_question, n):
        pos_tag = ['JJ', 'JJR', 'JJS']
        adj = []
        tagged_words = nltk.pos_tag(tokens_per_question)
        for (txt, tag) in tagged_words:
            if (tag in pos_tag):
                adj.append(txt)
        freq_dist = nltk.FreqDist(a for a in adj)
        cols = ['Adjective' , 'Frequency']
        df_top_n_adjectives = pd.DataFrame(freq_dist.most_common(n), columns=cols)
        self.df_top_n_adjectives.append(df_top_n_adjectives)
        return

    def exploratory_data_analysis(self, n_words):
        #get respondents per block
        self.distribution_of_respondents_per_block()
        #get respondents per course
        self.distribution_of_respondents_per_course()
        for tokens in self.input_tokens_per_question:
            #get top n words per question
            self.top_n_nouns(tokens,n_words)
            self.top_n_adjectives(tokens, n_words)
        return

    def sentiment_analysis(self):
        cols = ['Texts', 'Polarity', 'Sentiment', 'Question Number']
        category_cols = ['Text', 'Tokens']
        polarity_scores = []
        sentiment_category = []
        sentiment = SentimentIntensityAnalyzer()
        pos_sentiments = []
        pos_sentiments_tokens = []
        neg_sentiments = []
        neg_sentiments_tokens = []

        for i in range(len(self.input_df_for_sentiment_analysis['Text'])):
            scores = [
               sentiment.polarity_scores(sentence)['compound'] for sentence in nltk.sent_tokenize( self.input_df_for_sentiment_analysis.iat[i,0])
            ]
            if mean(scores) > 0:
                sentiment_category.append('positive')
                pos_sentiments.append(self.input_df_for_sentiment_analysis.iat[i,0])
                pos_sentiments_tokens.append(self.input_df_for_sentiment_analysis.iat[i,1])
            else:
                sentiment_category.append('negative')
                neg_sentiments.append(self.input_df_for_sentiment_analysis.iat[i, 0])
                neg_sentiments_tokens.append(self.input_df_for_sentiment_analysis.iat[i,1])
            polarity_scores.append(mean(scores))
        #record sentiment analysis result
        self.df_sentiment_analysis_result = pd.DataFrame(list(zip(
            self.input_df_for_sentiment_analysis['Text'].tolist(),
            polarity_scores, sentiment_category,
            self.input_df_for_sentiment_analysis['Question Number'].tolist()
        )), columns=cols)
        self.df_positive_predicted_sentiments = pd.DataFrame(
            list(zip(pos_sentiments, pos_sentiments_tokens)),
            columns=category_cols
        )
        self.df_negative_predicted_sentiments = pd.DataFrame(
            list(zip(neg_sentiments, neg_sentiments_tokens)),
            columns= category_cols
        )

        pos_sentiments_per_question = np.zeros(3)
        neg_sentiments_per_question = np.zeros(3)
        data = self.df_sentiment_analysis_result

        for i in range(len(data['Question Number'])):
            index = int(data.iat[i,3])
            if data.iat[i,2]=='positive':
                pos_sentiments_per_question[index-1]+=1
            if data.iat[i,2]=='negative':
                neg_sentiments_per_question[index-1]+=1
        self.list_freq_neg_sentiments_per_question = neg_sentiments_per_question
        self.list_freq_pos_sentiments_per_question = pos_sentiments_per_question
        return

    def LatentDirichletAllocation(self, tokens, filename_lda_model, filename_corpus, filename_dictionary):
        dictionary = corpora.Dictionary(tokens)
        corpus = [dictionary.doc2bow(text) for text in tokens]
        num_topics = 3
        lda_model = gensim.models.ldamodel.LdaModel(
            corpus=corpus , num_topics=num_topics, random_state=100, update_every=1,
            id2word=dictionary,passes=15
        )
        #print topics
        topics = lda_model.print_topics(num_words=6)
        for topic in topics:
            print("Topics : {} ".format(topic))

        #save for future use
        pickle.dump(corpus, open(filename_corpus, 'wb'))
        dictionary.save(filename_dictionary)
        lda_model.save(filename_lda_model)
