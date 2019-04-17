# -*- coding: utf-8 -*-
"""
@author: Alberto Barbado González
@mail: alberto.barbado.gonzalez@gmail.com
"""

import re
import pandas as pd
import numpy as np
import spacy
nlp = spacy.load('es_core_news_sm')

from program.config import (PATH_POEMS, PATH, WORD_COL, NAME1, NAME2, 
                    NAME3, NAME4, WORD_EMBEDDING_FILE, FILE_NAME_DCT_RESULTS,
                    DOCS_NAME_DCT, TYPE_EMBEDDING, FILE_NAME_VOCAB)
from program.tools import file_presistance, word_preprocessing, joint_function
from numpy.random import randint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import norm
from bert_embedding import BertEmbedding

### LOAD CSV
#DF_COMPOSITION_EMBEDDING_STANZA = pd.read_csv(PATH + '/' + 'df_composition_embeddings_stanza_{0}_{1}.csv'.format("joint", TYPE_EMBEDDING))
DF_COMPOSITION_EMBEDDING_STANZA = pd.read_csv(PATH + '/' + 'df_composition_embeddings_stanza_{0}_{1}_part1.csv'.format("joint", TYPE_EMBEDDING))
DF_COMPOSITION_EMBEDDING_STANZA = DF_COMPOSITION_EMBEDDING_STANZA.append(pd.read_csv(PATH + '/' + 'df_composition_embeddings_stanza_{0}_{1}_part2.csv'.format("joint", TYPE_EMBEDDING)))

try:
    del DF_COMPOSITION_EMBEDDING_STANZA['Unnamed: 0']
except:
    pass

#DCT_SONNETS = file_presistance(PATH + "/" + DOCS_NAME_DCT + ".p", "generic", None, "load")
DCT_SONNETS = file_presistance(PATH + "/" + DOCS_NAME_DCT + ".json", "json", None, "load")

try:
    DF_COMPOSITION_EMBEDDING_STANZA = DF_COMPOSITION_EMBEDDING_STANZA.set_index('index')
except:
    pass

def tfidf_query_similairty(query_text, strict=False, return_list=True):
    """
    # TODO
    
    Receive a query and retrieves the most similar sonnet using tf-idf; it builds 
    the tf-idf matrix using the stored documents and the new query and then it checks
    the feature word most similar to the query (last row used) and retrieves the index
    corresponding to that document.
    
    This tfidf function considers both the lemmatized words and the possible ngrams
    as well up to 2grams.
    
    """
    
    # Obtain documents
    documents = [f['ngrams2str'] for f in DCT_SONNETS.values()]
    
    # Obtain ngrams of new text
    _, _, query_words_ngrams, _ = word_preprocessing(query_text)
    text_ngrams = ""
    for w in query_words_ngrams:
        text_ngrams += w
        text_ngrams += " "
    
    # Add retrieval query as a new document
    documents.append(text_ngrams)
    
    # docs2tfidf
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(documents)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    
    X_sim = (X_train_tfidf * X_train_tfidf.T).A
#    X_dense = pd.DataFrame(X_train_tfidf.todense())
    words_used = count_vect.get_feature_names()
    
    # Last row corresponds to query similarity to docs
    results =  pd.DataFrame(X_sim[:,-1])
    results.drop(results.tail(1).index, inplace=True) # drop autosimilarity
    
    
    if not return_list:
        # If there's no similarity to any doc, give a random one
        if results[0].max()==results[0].min():
            results = randint(low=0, high=len(results))
        else:
            results = results[0].values.argmax()
        doc_sonnet = DCT_SONNETS[results]
        
    else:
        # By default give all the dict as it is
        doc_sonnet = DCT_SONNETS
        
        if results[0].max() != results[0].min():
            # Preselect only docs with maximum TF-IDF
#            max_value = results[0].max()
#            results = results[results[0]==max_value]
#            mean_value = results[0].mean()
#            results = results[results[0]>=mean_value]
            if strict:
                intermedian_value = results[0].max()
            else:
                intermedian_value = (2*results[0].max() + results[0].mean())/3
            results = results[results[0]>=intermedian_value]
#            results = results.sort_values(by=0, ascending=False)
            doc_sonnet = {key:doc_sonnet[key] for key in list(results.index)} # no need to sort anymore; use filtered dict
    
    return doc_sonnet, words_used



def icm(v1, v2):
    """
    # TODO
    """
    c = 1.2
    c1 = 1
    c2 = 1
    
    v1_norm = norm(v1)
    v2_norm = norm(v2)
    v1_v2 = v1_norm*v2_norm*cosine_similarity(v1,v2)[0][0]
    result = c*v1_v2 + (c1 - c)*v1_norm**2 + (c2 - c)*v2_norm**2
    return result


def embedding_query_stanza(query_text, composition_type, metric, use_prefilter=True, log=False):
    """
    # TODO
    Doc retrieval using BERT embedding against an input text query
    
    """
    
    # Process the query text if needed
    # Transfor str to str withoupt uppercases, stopwords and lemmatized
    if TYPE_EMBEDDING == "words_lem_nonstopwords":
        _, list_words, _, _ = word_preprocessing(query_text)
        
        if len(list_words) > 1:
            s = ''
            for w in list_words:
                s += w
                s += ' '
            query_text = s
        else:
            query_text = list_words[0]
         
        # Using BERT embedding library
        bert_embedding = BertEmbedding(model='bert_12_768_12', dataset_name='wiki_multilingual')
        result = bert_embedding(list_words)
        df_features = pd.DataFrame([x[1][0] for x in result])
    
    elif TYPE_EMBEDDING == "whole_text":
        list_words = re.findall(r'\w+', query_text,flags = re.UNICODE)
        list_words = [w.lower() for w in list_words]
        s = ''
        for w in list_words:
            s += ' '
            s += w
            
        query_text = s
        
        # Using BERT embedding library
        bert_embedding = BertEmbedding(model='bert_12_768_12', dataset_name='wiki_multilingual')
        result = bert_embedding(list_words)
        df_features = pd.DataFrame([x[1][0] for x in result])
    
    elif TYPE_EMBEDDING == "fasttext":
        NotImplementedError("Composition type used is not recognised")
    else:
        raise NotImplementedError("Embedding defined in config is not recognised")
    
    # Sum of vectors
    if composition_type=="sum":
        raise NotImplementedError("Embedding defined in config is not recognised")
    elif composition_type=="mean":
        raise NotImplementedError("Embedding defined in config is not recognised")
    elif composition_type=="joint":
        vector_compos_query = pd.DataFrame(df_features)
        # Inverse order (RL_SEQ)
        vector_compos_query.reset_index(inplace=True)
        del vector_compos_query['index']
        vector_compos_query.sort_index(ascending=False, inplace=True)
        key_max = vector_compos_query.index.max()
        v_follow = [0]*vector_compos_query.shape[1]
        # Iterate per word
        for key, vi in vector_compos_query.iterrows():
            # First vector is not joint to anyone yet
            if key == key_max:
                v_follow = vi
            else:
                v_follow = joint_function(v_follow,vi)
        vector_compos_query = pd.DataFrame(v_follow).T
        dct_embedding_all = DF_COMPOSITION_EMBEDDING_STANZA
    else:
        raise NotImplementedError("Composition type not included. Try sum, mean or joint")
    
    # If condition pre-filter with TF-IDF
    list_prefilter = list(DCT_SONNETS.keys())
    if use_prefilter:
        doc_aux, _ = tfidf_query_similairty(query_text, strict=False, return_list=True)
        list_prefilter = list(doc_aux.keys())
        dct_embedding_all = dct_embedding_all[dct_embedding_all.index.isin(list_prefilter)]
    
    metric_sim = -np.inf
    key_final = -np.inf

    for key, sonnet in DCT_SONNETS.items():
        if key not in list_prefilter:
            continue
        
        vector_compos_sonnet = pd.DataFrame(dct_embedding_all.loc[int(key)])
        
        for _, vector_compos_stanza in vector_compos_sonnet.iterrows():
            vector_compos_stanza = pd.DataFrame(vector_compos_stanza).T

            # Compute metric
            if metric == "cosine":
                metric_sim_aux = cosine_similarity(vector_compos_query, vector_compos_stanza)[0][0]
            elif metric == "icm":
                metric_sim_aux = icm(v1=vector_compos_stanza, v2=vector_compos_query)
            
            # Compare metrics
            if metric_sim_aux >= metric_sim:
                metric_sim = metric_sim_aux
                key_final = key
                if log:
                    print("Best {0} metric until now {1} for doc nº {2}".format(metric, np.round(metric_sim_aux,4), key))
                    print()
                    print(DCT_SONNETS[key_final]['text'])
                    print("*"*50)
                    print("")
        
    return DCT_SONNETS[key_final]
    
 