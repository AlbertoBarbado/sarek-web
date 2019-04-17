# -*- coding: utf-8 -*-
"""
@author: Alberto Barbado González
@mail: alberto.barbado.gonzalez@gmail.com

"""

from flask import Flask, render_template, redirect, url_for, request
from query_web import embedding_query_stanza
            
app = Flask(__name__)  


    
@app.route('/')
def home():
   return render_template('home.html')

@app.route('/result',methods = ['POST', 'GET'])
def result_student():
   if request.method == 'POST':
      result = request.form
      print("query_text introduced", result['query_text']) # results passes the input param as a dict
      sonnet = embedding_query_stanza(result['query_text'], composition_type="joint", metric="cosine", use_prefilter=False, log=False)
      print("Sonnet: ", sonnet['text'])
      return render_template("result.html",result = sonnet)
  

if __name__ == "__main__":
    app.run(debug=True)
