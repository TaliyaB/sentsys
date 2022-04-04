import os.path

from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.core.files.storage import FileSystemStorage
from subprocess import run, PIPE
from django.http import HttpResponseRedirect
import requests
import sys


def index(request):
    r = requests.get('https://httpbin.org/status/418')
    print(r.text)
    return HttpResponse('<pre>' + r.text + '</pre>')

def primary(path):
    import os
    sys.path.append(os.path.join(os.getcwd(), 'sentiments/sentiments'))
    """
    DATA EXTRACTOR CODE
    """
    from sentiments import TextProcessor, Visualizer, MachineLearning
    import os
    import pandas as pd
    #path = "C:\\Users\\user\\Documents\\Freelance\\sentiments\\sample_data\\Student' s Perception towards Online Learning Questionnaire.csv (1)\\Student' s Perception towards Online Learning Questionnaire.csv"
    path = os.path.join(os.getcwd(),'media', path)
    # create data object
    data = TextProcessor.TextProcessor()
    data.parser(path=path)

    fname2 = 'visuals/templates/preprocessed.html'
    with open(fname2, 'w') as f:
        f.write(data.df_final_data_for_sentiment_analysis.to_html())
        f.close()
    print('Created {} size {}'.format(fname2, os.path.getsize(fname2)))

    print(data.df_final_data_for_sentiment_analysis)
    machine_learning_processes = MachineLearning.MachineLearning(
        df_for_sentiment_analysis=data.df_final_data_for_sentiment_analysis,
        tokens_per_question=data.list_tokens_per_question,
        respondents_by_course=data.dict_respondents_course,
        respondents_by_block=data.dict_respondents_block,
        total_respondents=data.int_respondents_count)
    machine_learning_processes.exploratory_data_analysis(n_words=5)
    machine_learning_processes.sentiment_analysis()

    # adj
    top_n_adj = Visualizer.Visualizer()
    for i in range(len(machine_learning_processes.df_top_n_adjectives)):
        html1 = machine_learning_processes.df_top_n_adjectives[i].to_html()
        fname1 = 'visuals/templates/adj{}.html'.format(i)
        with open(fname1, 'w') as f:
            f.write(html1)
            f.close()
        print('Created {} size {}'.format(fname1, os.path.getsize(fname1)))

    top_n_adj.multiple_bar_for_top_n_words(df_top_n_words=machine_learning_processes.df_top_n_adjectives,
                                           word_type='Adjective',
                                           title="Top Adjectives per Question",
                                           output_file="visuals/templates/top_n_Adjective.png")

    # noun
    top_n_noun = Visualizer.Visualizer()
    for i in range(len(machine_learning_processes.df_top_n_nouns)):
        html = machine_learning_processes.df_top_n_nouns[i].to_html()
        fname = 'visuals/templates/nouns{}.html'.format(i)
        with open(fname, 'w') as f:
            f.write(html)
            f.close()
        print('Created {} size {}'.format(fname, os.path.getsize(fname)))

    top_n_noun.multiple_bar_for_top_n_words(df_top_n_words=machine_learning_processes.df_top_n_nouns,
                                            word_type='Noun',
                                            title="Top Nouns per Question",
                                            output_file="visuals/templates/static/top_n_Noun.png")

    # block
    respondents_per_block = Visualizer.Visualizer()
    respondents_per_block.pie(x_data=data.dict_respondents_block.keys(),
                              y_data=data.dict_respondents_block.values(),
                              title="Distribution of Respondents per Course",
                              output_filename="visuals/templates/static/respondents_per_block.png")

    # course
    respondents_per_course = Visualizer.Visualizer()
    respondents_per_course.pie(x_data=data.dict_respondents_course.keys(),
                               y_data=data.dict_respondents_course.values(),
                               title="Distribution of Respondents per Course",
                               output_filename="visuals/templates/static/respondents_per_course.png")

    # sentiment
    sentiment_analysis_frequency = Visualizer.Visualizer()
    sentiment_analysis_frequency.multiple_bar_for_sentiment_analysis(
        y0_data=machine_learning_processes.list_freq_pos_sentiments_per_question,
        y1_data=machine_learning_processes.list_freq_neg_sentiments_per_question,
        title="Frequency Distribution of Sentiments per Question",
        output_file="visuals/templates/static/sentiment_analysis.png")

    # wordcloud

    positive_wordCloud = Visualizer.Visualizer()
    positive_wordCloud.wordCloud(text=machine_learning_processes.df_positive_predicted_sentiments['Text'].tolist(),
                                 color='white', title='Positive WordCloud',
                                 output_file='visuals/templates/static/positive_wordcloud.png')

    negative_wordCloud = Visualizer.Visualizer()
    negative_wordCloud.wordCloud(text=machine_learning_processes.df_negative_predicted_sentiments['Text'].tolist(),
                                 color='black', title='Negative WordCloud',
                                 output_file='visuals/templates/static/negative_wordcloud.png')

    machine_learning_processes.LatentDirichletAllocation(
        tokens=machine_learning_processes.df_negative_predicted_sentiments['Tokens'].tolist(),
        filename_lda_model='visuals/templates/all_generated/negative.gensim',
        filename_corpus='visuals/templates/all_generated/negative_corpus.pkl',
        filename_dictionary='visuals/templates/all_generated/negative_dictionary.gensim'
    )

    machine_learning_processes.LatentDirichletAllocation(
        tokens=machine_learning_processes.df_positive_predicted_sentiments['Tokens'].tolist(),
        filename_lda_model='visuals/templates/all_generated/positive.gensim',
        filename_corpus='visuals/templates/all_generated/positive_corpus.pkl',
        filename_dictionary='visuals/templates/all_generated/positive_dictionary.gensim'
    )

    import pyLDAvis
    import gensim
    import pickle
    # pyLDAvis.enable_notebook()

    lda = 'visuals/templates/all_generated/positive.gensim'
    dictionary = gensim.corpora.Dictionary.load('visuals/templates/all_generated/positive_dictionary.gensim')
    corpus = pickle.load(open('visuals/templates/all_generated/positive_corpus.pkl', 'rb'))
    lda_model = gensim.models.ldamodel.LdaModel.load(lda)
    # pyLDAvis.enable_notebook()
    lda_display = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(lda_display, 'visuals/templates/pos-lda.html')

    lda = 'visuals/templates/all_generated/negative.gensim'
    dictionary = gensim.corpora.Dictionary.load('visuals/templates/all_generated/negative_dictionary.gensim')
    corpus = pickle.load(open('visuals/templates/all_generated/negative_corpus.pkl', 'rb'))
    lda_model = gensim.models.ldamodel.LdaModel.load(lda)
    # pyLDAvis.enable_notebook()
    lda_display = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(lda_display, 'visuals/templates/neg-lda.html')

def upload(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        primary(filename)
        return render(request, 'visuals.html', {
            'uploaded_file_url': uploaded_file_url
        })
        #out = run([sys.executable,'runall.py'], shell=False, stout=PIPE)
    return render(request, 'up.html')

    #return HttpResponseRedirect(request.path)
    #template = loader.get_template('up.html')
    #return HttpResponse(template.render())


def visuals(request):
    template = loader.get_template('visuals.html')
    return HttpResponse(template.render())

def data(request):
    template = loader.get_template('data.html')
    return HttpResponse(template.render())


def lda(request):
    template = loader.get_template('LDA.html')
    return HttpResponse(template.render())


def nouns0(request):
    template = loader.get_template('nouns0.html')
    return HttpResponse(template.render())


def nouns1(request):
    template = loader.get_template('nouns1.html')
    return HttpResponse(template.render())


def nouns2(request):
    template = loader.get_template('nouns2.html')
    return HttpResponse(template.render())


def adj0(request):
    template = loader.get_template('adj0.html')
    return HttpResponse(template.render())


def adj1(request):
    template = loader.get_template('adj1.html')
    return HttpResponse(template.render())


def adj2(request):
    template = loader.get_template('adj2.html')
    return HttpResponse(template.render())

def poslda(request):
    template = loader.get_template('pos-lda.html')
    return HttpResponse(template.render())

def neglda(request):
    template = loader.get_template('neg-lda.html')
    return HttpResponse(template.render())

