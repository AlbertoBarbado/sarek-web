# -*- coding: utf-8 -*-
"""
@author: Alberto Barbado González
@mail: alberto.barbado.gonzalez@gmail.com

"""
import glob
import pickle
import xmltodict
import json
import re
from nltk.corpus import stopwords
from numpy.linalg import norm
from nltk import ngrams
import spacy
nlp = spacy.load('es_core_news_sm')
import pandas as pd

def get_files(PATH, extension):
    """
    # TODO
    """
    return glob.glob(PATH + "/" + "*." + extension)

def file_presistance(file_path, file_type, doc, mode):
    """
    # TODO
    """
    
    if mode == "load":
        doc = None
        if file_type == "xml":
            with open(file_path, encoding="utf-8") as fd:
                doc = xmltodict.parse(fd.read())
        elif file_type=='generic':
            with open(file_path, "rb") as fd:
                doc = pickle.load(fd)
        elif file_type=='json':
            with open(file_path, "r", encoding="utf8") as fd:
                doc = json.load(fd)
        return doc
    
    elif mode == "save":
        if file_type == "xml":
            with open(file_path, 'w', encoding="utf-8") as result_file:
                result_file.write(xmltodict.unparse(doc, pretty=True))
        elif file_type=='generic':
            with open(file_path, "wb") as fd:
                pickle.dump(doc, fd)
        elif file_type=='json':
            with open(file_path, "w", encoding="utf8") as fd:
                json.dump(doc, fd)
        return None
    

def word_grams(words, lim_min=2, lim_max=5):
    """
    # TODO
    Function to obtain different ngrams from a word. It gives back the list containing those ngrams as
    well as the original word.
    
    """
    s = []
    for n in range(lim_min, lim_max):
        for ngram in ngrams(words, n):
            s.append(''.join(str(i) for i in ngram))
            break # para coger solo el ngrama de inicio
    return s


    
def word_preprocessing(text):
    """
    # TODO
    Generic function that recieves a str array ato be preproccesed. It performs:
        - tokenization
        - decapitalization
        - stop words removal
        - filters non-words characters
        - lemmatization
    """
    
    # Tokenize each sentence
    words = re.findall(r'\w+', text,flags = re.UNICODE)
    # Upper case to lowercase
    words = [w.lower() for w in words]
    # Remove stopwords
    words = [w for w in words if w not in stopwords.words('spanish')]
    # Remove non alphanumeric characters
    words = [w for w in words if w.isalpha()]
    # Lemmatize words for its use in affective features
    words_lem = [token.lemma_ if (token.tag_.split('=')[-1] != 'Sing') else w for w in words for token in nlp(w)] # lemmatize only not-singular words and verbs
#    words_lem = [token.lemma_ for w in words for token in nlp(w)] # lemmatize all
    # n-grams for those words
    words_lem_ngrams = list(set([x for w in words_lem for x in word_grams(w, len(w)-1, len(w)+1)]))
    # words lem with stop words and with uppercases
    words_lem_complete = [token.lemma_ if (token.tag_.split('=')[-1] != 'Sing') else w for w in re.findall(r'\w+', text,flags = re.UNICODE) for token in nlp(w)] # lemmatize only not-singular words and verbs
    
    return words, words_lem, words_lem_ngrams, words_lem_complete


def joint_function(v1,v2):
    """
    TODO
    """

    numer = v1+v2
#    v1_v2 = norm(np.column_stack((v1.values,v2.values)))
    
    v1_norm = norm(v1)
    v2_norm = norm(v2)
#    v1_2 = norm(np.column_stack((v1.values,v1.values)))
#    v2_2 = norm(np.column_stack((v2.values,v2.values)))
#    denom = norm(numer)
#    denom = norm(np.column_stack((numer.values,numer.values)))
    denom = v1_norm + v2_norm + 2*v1_norm*v2_norm*cosine_similarity(pd.DataFrame(v1).T, pd.DataFrame(v2).T)[0][0]
    v_joint = (numer/denom)*np.sqrt((v1_norm)**2 + (v2_norm)**2 - v1_norm*v2_norm*cosine_similarity(pd.DataFrame(v1).T, pd.DataFrame(v2).T)[0][0])
    
    return v_joint