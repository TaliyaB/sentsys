import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from wordcloud import WordCloud, STOPWORDS
#from sentiments import MachineLearning

class Visualizer():
    def __init__(self):
        return

    def bar(self, x_data, y_data, x_label, title, output_file):
        fontsize = 10
        print("Generating {}".format(title))
        plt.figure(figsize=(8,5))
        plt.bar(x_data,y_data)
        plt.title(title)
        y_label = "Number of Respondents"
        plt.xlabel(x_label, fontsize=fontsize, labelpad=10)
        plt.xticks(rotation=45)
        plt.ylabel(y_label, fontsize=fontsize)
        plt.tight_layout()
        #plt.show()
        plt.savefig(output_file)
        return

    def multiple_bar_for_sentiment_analysis(self, y0_data, y1_data, title, output_file):
        fontsize = 10
        x_label = "Question Number"
        x_data =["1", "2", "3"]
        y0_label = "Positive"
        y1_label = "Negative"
        y_label = "Frequency"
        print("Generating {}".format(title))
        x_axis = np.arange(len(x_data))
        plt.figure(figsize=(8,5))
        plt.bar(x_axis-0.2, y0_data, 0.4, label=y0_label)
        plt.bar(x_axis+0.2, y1_data, 0.4, label=y1_label)
        plt.xticks(x_axis, x_data)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        #plt.show()
        plt.savefig(output_file)
        return

    def pie(self, x_data, y_data, title, output_filename):
        colors = plt.get_cmap('Blues')(np.linspace(0.2, 0.7, len(x_data)))
        #plot
        fig, ax = plt.subplots(figsize=(8,5))
        ax.pie(y_data, labels=x_data, autopct='%1.1f%%')
        ax.axis('equal')  #draw as circle
        fig.savefig(output_filename)
        #plt.show()
        return

    def multiple_bar_for_top_n_words(self, df_top_n_words, word_type, title, output_file):
        fig, axs = plt.subplots(1,3, figsize=(8,5), sharey=True)
        fig.suptitle(title)
        for i in range(3):
            axs[i].plot(df_top_n_words[i][word_type],
                           df_top_n_words[i]['Frequency'])
            axs[i].tick_params(labelrotation=90)
            axs[i].set_title(str(i+1))
            #axs[1][i].plot(df_top_n_words[i+1][word_type],
            #               df_top_n_words[i+1]['Frequency'])
            #axs[1][i].set_title(str(4 + i))
        #axs[2][0].plot(df_top_n_words[6][word_type],
        #              df_top_n_words[6]['Frequency'])
        #axs[2][0].set_title(str(7))
        fig.savefig(output_file)
        return

    def wordCloud(self, text, color, title, output_file):
        w = 1600
        h = 800
        margin = 2
        min_font_size = 20
        figsize = (12, 8)
        text_str = ' '.join(str(e) for e in text)
        wordCloud = WordCloud(
            collocations=False, background_color=color, stopwords=set(STOPWORDS), width=w, height=h, margin=margin,
            min_font_size=min_font_size).generate(text_str)
        plt.figure(figsize= figsize)
        plt.imshow(wordCloud, interpolation='bilinear')
        plt.axis('off')
        plt.figtext(0.5, 0.8, title, fontsize = 20, ha='center')
        plt.savefig(output_file)
        #plt.show()
        return

    def display_img(self):
        return