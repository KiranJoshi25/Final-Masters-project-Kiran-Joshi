from flask import Flask, request, render_template
import processing
import pandas as pd
import numpy as np
import pickle
app = Flask(__name__,template_folder='template')

open_file = open('resources/names.pkl', "rb")
loaded_list = pickle.load(open_file)

@app.route('/')
def my_form():
    return render_template('form.html')


@app.route('/', methods=['POST'])
def my_form_post():
    text= request.form['fname']


    if str(text) in loaded_list:
        #print('now in if loop processing', text)
        new_var, K, L, D, R, R_C, S, your_list, graph,unigram_graph,bigram_graph,trigram_graph = processing.pree(str(text))
        graphJSON = processing.country_graph(K)
        L_graph = processing.generate_graph_language(L)
        D_graph = processing.get_date_time(D)
        R_graph = processing.rating_graph(R)
        sentiment_graph = processing.sentiment_graph(R_C)
        sellers_graph = processing.sellers_graph(S)

        return render_template('form.html', new_var=text, graphJSON=graphJSON, num_of_review=new_var,
                               L_graph=L_graph, D_graph=D_graph, R_graph=R_graph, RC_graph=sentiment_graph,
                               sellers_graph=sellers_graph, your_list=your_list, geo_graph=graph,
                               unigram_graph = unigram_graph, bigram_graph = bigram_graph, trigram_graph = trigram_graph
                               )
    else:

        docs1 = ''
        text = processing.preprocessing(text)
        for i in text:
            docs1+=str(i)
        result = processing.get_new_vector(docs1)
        return render_template('form.html', results = result[1:])

app.run()