from pathlib import Path
import glob
import numpy as np
import pandas as pd
import collections
import num2words
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.graph_objects as go
import pickle
sid = SentimentIntensityAnalyzer()
#import plotly.graph_objects as px
import collections
import json
import plotly
import plotly.express as px
data = pd.read_pickle('resources/First_file_new.pickle')
from nltk.corpus import stopwords



def preprocessing(text):
    # split into tokens
    tokens = word_tokenize(text)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    import string
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]

    # convert number and remove tokens that are not alphabetic
    words = []
    for word in stripped:
        try:
            if word.isdigit():
                words.append(num2words.num2words(word))
            elif word.isalpha():
                words.append(word)
        except:
            pass
    # remove stop words and single characters
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words and len(w) > 1]

    # stemming and lemmatization
    porter = PorterStemmer()
    words = [porter.stem(word) for word in words]
    doc = ' '.join(words)
    doc = doc.translate(str.maketrans('', '', string.punctuation))
    return doc


def create_vector(name,doc):
    docs = doc.strip('][').split(',')
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vector = tfidf_vectorizer.fit_transform(docs)
    # convert tf-idf into data frame, each row is a tf-idf vector for each document
    tfidf_df = pd.DataFrame(tfidf_vector.toarray(), index=[name], columns=tfidf_vectorizer.get_feature_names_out())
    return tfidf_df









def get_date_time(K):
    Key = [i for i in K]
    Value = [K[i] for i in K]
    df = pd.DataFrame({
        "Date": Key,
        "Reviews": Value,

    })
    df = df.sort_values(by="Date")
    fig = px.line(df, x="Date", y="Reviews", title="Timeline of Reviews")

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    #header =
    return graphJSON

def pree(text):

    new_data = data[data['Phone_names']==text]
    list = len(new_data['extract'])
    K = collections.Counter(new_data['Countries'])
    L = collections.Counter(new_data['lang'])
    D = collections.Counter(new_data['date'])
    R = collections.Counter(new_data['Ratings'])
    R_C = get_sentiments(new_data)
    S = collections.Counter(new_data['source'])
    your_list, graph = aspect_gainer(text)
    unigram_graph,bigram_graph,trigram_graph = n_gram_analysis(new_data)
    return list,K,L,D,R,R_C,S,your_list,graph,unigram_graph,bigram_graph,trigram_graph

def preeCompare(text):

    new_data = data[data['Phone_names']==text]
    list = len(new_data['extract'])
    K = collections.Counter(new_data['Countries'])
    L = collections.Counter(new_data['lang'])
    D = collections.Counter(new_data['date'])
    R = collections.Counter(new_data['Ratings'])
    R_C = get_sentiments(new_data)
    S = collections.Counter(new_data['source'])
    your_list, graph = aspect_gainer(text)
    unigram_graph,bigram_graph,trigram_graph = n_gram_analysis(new_data)
    return list,K,L,D,R,R_C,S,graph,unigram_graph,bigram_graph,trigram_graph


def country_graph(K):

    Key = [i for i in K]
    Value = [K[i] for i in K]
    df = pd.DataFrame({
        "Country": Key,
        "Reviews": Value,

    })

    fig = px.bar(df, x ="Country", y="Reviews", title="Number of Reviews by country")

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    #header =
    return graphJSON


def generate_graph_language(K):
    Key = [i for i in K]
    Value = [K[i] for i in K]
    df = pd.DataFrame({
        "Language": Key,
        "Reviews": Value,

    })

    fig = px.bar(df, x="Language", y="Reviews", title='Number of Reviews by Language')

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    #header = "Number of Reviews by Language"
    return graphJSON



def rating_graph(K):
    Key = [i for i in K]
    Value = [K[i] for i in K]
    df = pd.DataFrame({
        "Ratings": Key,
        "Numbers": Value,

    })
    df = df.sort_values(by="Ratings", ascending=False)
    fig = px.pie(df, names="Ratings", values="Numbers" , title='Ratings Distribution')

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    #header = "Ratings Distribution"
    return graphJSON




def sentiment_graph(K):
    Key = [i for i in K]
    Value = [K[i] for i in K]
    df = pd.DataFrame({
        "Sentiment": Key,
        "Values": Value,

    })

    fig = px.bar(df, x="Sentiment", y="Values", title='Sentiment Analysis')

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    # header = "Number of Reviews by Language"
    return graphJSON

def sellers_graph(K):
    Key = [i for i in K]
    Value = [K[i] for i in K]
    size = [i*10 for i in Value]
    df = pd.DataFrame({
        "Seller": Key,
        "Numbers": Value,

    })

    fig = px.scatter(df, x="Seller", y="Numbers",color="Seller",size="Numbers", title='Sellers Distribution')

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    # header = "Number of Reviews by Language"
    return graphJSON




def get_sentiments(new_df):
    scores = new_df['extract'].apply(lambda review: sid.polarity_scores(str(review)))
    compound = scores.apply(lambda score_dict: score_dict['compound'])
    dic = {'Positive': 0, 'Negative': 0, 'Neutral': 0}

    for i in compound:
        if i < 0 or i == 0:
            dic['Negative'] += 1
        elif 0 < i < 0.5:
            dic['Neutral'] += 1
        else:
            dic['Positive'] += 1
    return dic

def aspect_gainer(text):
    open_file = open(f'Pickles/{text}.pkl', "rb")
    T = pickle.load(open_file)
    your_list = []
    graph = generate_graph(T)
    for i in range(len(T[0])):

        your_list.append((T[0][i],T[1][i]))


    return your_list,graph


def generate_graph(T):
    color = []
    x_bar = []
    y_bar = []
    N = T[0]
    M = T[2]
    for i in range(len(T[0])):
        color.append('Positive')
        color.append('Negative')
        x_bar.append(N[i])
        x_bar.append(N[i])
        y_bar.append(M[i][1])
        y_bar.append(M[i][0])
    df = pd.DataFrame({
        'Aspect': x_bar,
        'Sentiment_numbers': y_bar,
        'sentiment': color

    })
    df = df.sample(frac=1).reset_index(drop=True)

    fig = px.bar(df, x='Aspect', y='Sentiment_numbers', color='sentiment', title='Aspect based Sentiments', )


    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graph_json

def get_new_vector(tfidf):
    #with open('vector_data_dict.pickle', 'rb') as handle:
        #dic = pickle.load(handle)
    open_file = open('resources/names.pkl', "rb")
    loaded_list = pickle.load(open_file)
    name_list = list(loaded_list)
    with open('resources/all_text.pickle', 'rb') as handle:
        documents = pickle.load(handle)
    name_list.append('search')
    documents.append(tfidf)
    tfidf_vect = TfidfVectorizer()
    vector_matrix = tfidf_vect.fit_transform(documents)
    cosine_similarity_matrix = cosine_similarity(vector_matrix)
    df = pd.DataFrame(cosine_similarity_matrix, index=name_list, columns=name_list)
    return df['search'].nlargest(16).index.tolist()


def create_ngrams(token_list, nb_elements):
    ngrams = zip(*[token_list[index_token:] for index_token in range(nb_elements)])
    return (" ".join(ngram) for ngram in ngrams)


def frequent_words(list_words, ngrams_number=1, number_top_words=10):
    frequent = []
    if ngrams_number == 1:
        pass
    elif ngrams_number >= 2:
        list_words = create_ngrams(list_words, ngrams_number)
    else:
        raise ValueError("number of n-grams should be >= 1")
    counter = collections.Counter(list_words)
    frequent = counter.most_common(number_top_words)
    return frequent

def generate_n_gram_graph(K,title):
    Key = [i[0] for i in K]
    Value = [i[1] for i in K]
    stops = set(stopwords.words('english'))
    df = pd.DataFrame({
        "Ngrams": Key,
        "ocurrances": Value,

    })

    fig = px.bar(df, x="Ngrams", y="ocurrances", title=title)

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    # header = "Number of Reviews by Language"
    return graphJSON


def n_gram_analysis(new_df):
    text = ''
    for i in new_df['extract']:
        text += str(i).lower()
    doc1 = text.split()
    doc = []
    stops = set(stopwords.words('english'))
    for i in doc1:
        if i not in stops:
            doc.append(i)
    unigram = frequent_words(doc, ngrams_number=1, number_top_words=10)
    bigrams = frequent_words(doc, ngrams_number=2, number_top_words=10)
    trigrams = frequent_words(doc, ngrams_number=3, number_top_words=10)
    unigram_graph = generate_n_gram_graph(unigram,'Unigrams')
    bigram_graph = generate_n_gram_graph(bigrams, 'Bigrams')
    trigram_graph = generate_n_gram_graph(trigrams, 'trigrams')
    return unigram_graph,bigram_graph,trigram_graph