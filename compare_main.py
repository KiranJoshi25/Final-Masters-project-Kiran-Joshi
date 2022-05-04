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
    return render_template('compare.html')


@app.route('/', methods=['POST'])
def my_form_post():
    text= request.form['fname']
    new_text = request.form['fname2']
    #if not text or not new_text:
    #return render_template('compare.html',message = 'please select two products')

    message = 'you have selected '+text+' and '+new_text+' for comparison'
    new_var, K, L, D, R, R_C, S,graph, unigram_graph, bigram_graph, trigram_graph = processing.preeCompare(str(text))
    graphJSON = processing.country_graph(K)
    L_graph = processing.generate_graph_language(L)
    D_graph = processing.get_date_time(D)
    R_graph = processing.rating_graph(R)
    sentiment_graph = processing.sentiment_graph(R_C)
    sellers_graph = processing.sellers_graph(S)

    new_var1, K1, L1, D1, R1, R_C1, S1, graph1, unigram_graph1, bigram_graph1, trigram_graph1 = processing.preeCompare(str(new_text))
    graphJSON1 = processing.country_graph(K1)
    L_graph1 = processing.generate_graph_language(L1)
    D_graph1 = processing.get_date_time(D1)
    R_graph1 = processing.rating_graph(R1)
    sentiment_graph1 = processing.sentiment_graph(R_C1)
    sellers_graph1 = processing.sellers_graph(S1)





    return render_template('compare.html', text = text, new_text = new_text,
                               message = message,graphJSON=graphJSON,num_of_review = new_var ,
                               L_graph=L_graph, D_graph=D_graph, R_graph=R_graph, RC_graph=sentiment_graph,
                               sellers_graph=sellers_graph,  geo_graph=graph,
                               unigram_graph=unigram_graph, bigram_graph=bigram_graph, trigram_graph=trigram_graph,

                           graphJSON1=graphJSON1,
                           num_of_review1=new_var1,
                           L_graph1=L_graph1, D_graph1=D_graph1, R_graph1=R_graph1, RC_graph1=sentiment_graph1,
                           sellers_graph1=sellers_graph1, geo_graph1=graph1,
                           unigram_graph1=unigram_graph1, bigram_graph1=bigram_graph1, trigram_graph1=trigram_graph1

                               )




app.run()