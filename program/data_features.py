# -*- coding: utf-8 -*-
"""
@author: Alberto Barbado González
@mail: alberto.barbado.gonzalez@gmail.com
"""

import xmltodict
import re
import os
from nltk.corpus import stopwords
from nltk import ngrams
import pandas as pd
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import spacy
nlp = spacy.load('es_core_news_sm')

from program.config import (PATH_POEMS, PATH, WORD_COL, NAME1, NAME2, 
                    NAME3, NAME4, WORD_EMBEDDING_FILE, FILE_NAME_DCT_RESULTS,
                    DOCS_NAME_DCT, BERT_BATCH_SIZE, OTHER_SONNETS, SONNETS_ADDITIONAL, 
                    TYPE_EMBEDDING, LOAD_SPLITS, COMPOSITION_TYPE)
from program.tools import get_files, file_presistance
from numpy.random import randint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics.pairwise import cosine_similarity

import argparse
import collections
import logging
import json
from itertools import islice
from numpy.linalg import norm

from program.bert_utils import read_text, convert_examples_to_features
from bert_embedding import BertEmbedding


########### Pre load of embeddings
## Load vectors
#vocab_prueba = ["autoridad", "bajo", "aumento", "alta"]
#quantity = 1000
#wordvectors = KeyedVectors.load_word2vec_format(os.path.join(os.getcwd(), PATH, WORD_EMBEDDING_FILE), limit=quantity)
## Obtain all n-grams
#vocab_ext = list(set([x for w in vocab_prueba for x in word_grams(w, 2, len(w) + 1)]))
#list_vocab = list(wordvectors.vocab.keys())
#df_word_embedding = pd.DataFrame.from_dict(dict((w, wordvectors[w]) for w in vocab_ext if w in list_vocab)).T
## Add UNK column
#df_word_embedding = df_word_embedding.append(pd.DataFrame({'UNK':[0]*df_word_embedding.shape[1]}).T)
#dct_results_prueba = {}
#dct_results_prueba['vocab_ngrams_total'] = list_vocab
#dct_results_prueba['df_word_embedding'] = df_word_embedding
#dct_results_prueba['n_grams'] = vocab_ext
#dct_results_prueba['words_featured'] = list(df_word_embedding.index)
#file_presistance("dct_results_prueba.p", "generic", dct_results_prueba, "save")
# query_text = "La autoridad está bajo una alta presión en el alma"

#DCT_RESULTS = file_presistance("dct_results_prueba.p", "generic", None, "load")
#WORDS_FEATURED = DCT_RESULTS['words_featured']
#DF_WORD_EMBEDDING = DCT_RESULTS['df_word_embedding']
#
#DCT_RESULTS = file_presistance("datasets/dct_results.p", "generic", None, "load")
##########################



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



def parse_poem(document):
    """
    # TODO
    """
    text = ''
    for paragraph in  document["lg"]:
        for line in paragraph["l"]:
            # Line with content
            try:
                text += line["#"+"text"]
            # Empty line
            except:
                text += ''
            text += '\n'
        text += '\n'
    return text


def parse_stanza(document):
    """
    # TODO
    """
    
    key = 0
    stanzas = {}
    for paragraph in  document["lg"]:
        text = ''
        for line in paragraph["l"]:
            # Line with content
            try:
                text += line["#"+"text"]
            # Empty line
            except:
                text += ''
            text += '\n'
        stanzas[key] = {'text':text}
        key += 1
    return stanzas
    


def doc2text(doc, docs, type_doc):
    """
    # TODO
    This function recieves an xml document from the DISCO per-author files and retrieves the different
    sonnets content as well as author, sonnet name, id and year metadata. It gives back a dict with the different
    sonnets and metadata available for that xml document.
    
    """
    
    if type_doc=="author":
        dct_aux = {}
        j = 0
        # Iterate through poems in that doc
        iter_dict = doc["TEI"]["text"]["body"]
        # Metadata
        author = doc['TEI']['text']['front']['div']['head']
        year = doc['TEI']['text']['front']['div']['p']
                        
        for doc_poem in iter_dict.values():
            ################
            # One Sonnet per Author
            ################
            if 'lg' in doc_poem:
                # Metadata
                title = doc_poem['head']
                id_doc = doc_poem['@'+'xml:id']
                
                # Sonnet sequence
                if doc_poem['@'+'type'] == 'sonnet-sequence':
                    # with two or more parts
                    if type(doc_poem["lg"])==list: 
                        for sonnet in doc_poem["lg"]:
                            text = parse_poem(sonnet)
                            dct_stanzas = parse_stanza(sonnet)
                            docs.append(text)
                            dct_aux[j] = {'title':title, 'text':text, 'id_doc':id_doc, 'dct_stanzas':dct_stanzas}
                            j += 1
                    # with one part
                    else:
                        for key, value in doc_poem.items():
                            if key=='lg':
                                sonnet=value
                                text = parse_poem(sonnet)
                                dct_stanzas = parse_stanza(sonnet)
                                docs.append(text)
                                dct_aux[j] = {'title':title, 'text':text, 'id_doc':id_doc, 'dct_stanzas':dct_stanzas}
                                j += 1  
                                
                # In other case
                else:
                    text = parse_poem(doc_poem)
                    dct_stanzas = parse_stanza(doc_poem)
                    docs.append(text)
                    dct_aux[j] = {'title':title, 'text':text, 'id_doc':id_doc, 'dct_stanzas':dct_stanzas}
                    j += 1
            
            ###############
            # More than one sonnet per Author
            ##############
            else:
                for doc_sonnet in doc_poem:
                    # Metadata
                    title = doc_sonnet['head']
                    id_doc = doc_sonnet['@'+'xml:id']
                    
                    # Unique sonnets
                    if 'lg' in doc_sonnet and doc_sonnet['@'+'type']=='sonnet':
                        text = parse_poem(doc_sonnet)
                        dct_stanzas = parse_stanza(doc_sonnet)
                        docs.append(text)
                        dct_aux[j] = {'title':title, 'text':text, 'id_doc':id_doc, 'dct_stanzas':dct_stanzas}
                        j += 1
                
                    # Sonnet sequence  
                    elif 'lg' in doc_sonnet and doc_sonnet['@'+'type']=='sonnet-sequence':
                        # with two or more parts
                        if type(doc_sonnet["lg"])==list: 
                            for sonnet in doc_sonnet["lg"]:
                                text = parse_poem(sonnet)
                                dct_stanzas = parse_stanza(sonnet)
                                docs.append(text)
                                dct_aux[j] = {'title':title, 'text':text, 'id_doc':id_doc, 'dct_stanzas':dct_stanzas}
                                j += 1
                        # with one part
                        else:
                            for key, value in doc_sonnet.items():
                                if key=='lg':
                                    sonnet=value
                                    text = parse_poem(sonnet)
                                    dct_stanzas = parse_stanza(sonnet)
                                    docs.append(text)
                                    dct_aux[j] = {'title':title, 'text':text, 'id_doc':id_doc, 'dct_stanzas':dct_stanzas}
                                    j += 1                    
    
            for key, value in dct_aux.items():
                dct_aux[key].update({'author':author, 'year':year})
                
    else:
        author = doc['TEI']['teiHeader']['fileDesc']['titleStmt']['author']['#text']
        year = doc['TEI']['teiHeader']['fileDesc']['sourceDesc']['bibl']['hi']['#text']
        title = doc['TEI']['teiHeader']['fileDesc']['titleStmt']['title']['#text']
        id_doc = doc['TEI']['teiHeader']['fileDesc']['titleStmt']['@about']
                   
        # Iterate through stanzas in that doc
        sonnet = doc["TEI"]["text"]["body"]
        text = parse_poem(sonnet)
        dct_stanzas = parse_stanza(sonnet)
        docs.append(text)
        dct_aux = {0:{'title':title, 'text':text, 'id_doc':id_doc, 'dct_stanzas':dct_stanzas, 'author':author, 'year':year}}
                
            
    return docs, dct_aux
                

                
def docs2dict():
    """
    # TODO
    This function loads all the available documents and extracts the texts of the sonnets,
    and metadata info such as id, author and year. It also obtains a list of the lemmatized words 
    for each sonnet as well as the list extended with ngrams and creates a string with all the ngrams
    of each text for the latter use of tfidf.
    
    """
    
    # Load list of poems
    list_docs =  get_files(PATH_POEMS + '/per-sonnet', "xml")
    dct_sonnets = {}
    num_doc = 0
    num_sonnet = 0
    
    #### Iterate through all documents (DISCO)
    for file_path in list_docs:
        num_doc += 1
        print("{0}".format("*"*40))
        print("Iteration {0}/{1}".format(num_doc, len(list_docs)))
        print("{0}".format("*"*40))
 
        # Load file
        doc = file_presistance(file_path, "xml", None, "load")
        docs = []
        docs, dct_data = doc2text(doc, docs, "sonnet")
        
        #### Iterate per sonnet
        # Word preprocesing
        for index, data in dct_data.items():
            
            try:
                dct_sonnets = file_presistance(PATH + "/"+ DOCS_NAME_DCT + ".p", "generic", None, "load")
                dct_sonnets[num_sonnet]
                print("Already exist -- pass")
                num_sonnet += 1
                continue
            except:
                pass
                
            ##### Whole Text
            words, list_lem, list_lem_ngrams, list_lem_complete = word_preprocessing(data['text'])
            # Add affective features
            data['affective_features'], _, _, _ = affective_features(data['text'], list_lem, list_lem_complete, list_lem_ngrams)
            # Rename cols
            data.update({'words':words,
                         'words_lem':list_lem, 
                         'words_lem_ngrams':list_lem_ngrams,
                         'words_lem_complete':list_lem_complete,
                         'file_path':file_path})
            
            # string of words (ngram)
            s = ''
            for w in list_lem_ngrams:
                s += ' '
                s += w
            data.update({'ngrams2str':s})
            
            # string of words (lem)
            s = ''
            for w in list_lem:
                s += ' '
                s += w
            data.update({'lem2str':s})
            
            
            ##### Per Stanza
            for key, dct_stanza in data['dct_stanzas'].items():
                words, list_lem, list_lem_ngrams, list_lem_complete = word_preprocessing(dct_stanza['text'])
                # Add affective features
                data['dct_stanzas'][key]['affective_features'], _, _, _ = affective_features(dct_stanza['text'], list_lem, list_lem_complete, list_lem_ngrams)
                # Rename cols 
                data['dct_stanzas'][key].update({'words':words,
                                                 'words_lem':list_lem, 
                                                 'words_lem_ngrams':list_lem_ngrams,
                                                 'words_lem_complete':list_lem_complete})
                # string of words (ngram)
                s = ''
                for w in list_lem_ngrams:
                    s += ' '
                    s += w
                data['dct_stanzas'][key].update({'ngrams2str':s})
                
                # string of words (lem)
                s = ''
                for w in list_lem:
                    s += ' '
                    s += w
                data['dct_stanzas'][key].update({'lem2str':s})
            
            #### Store Sonnet Info
            # Append info and go to new iteration
            dct_sonnets[num_sonnet] = data
            num_sonnet += 1
            
            # Store in disk
            file_presistance(PATH + "/"+ DOCS_NAME_DCT + ".p", "generic", dct_sonnets, "save")
            
            
            
    #### Iterate through all documents (S.XX custom)
    dct_add = file_presistance(OTHER_SONNETS + "/" + SONNETS_ADDITIONAL + ".json", "json", None, "load")
    j = 0
    for key, data in dct_add.items():
        print("{0}".format("*"*40))
        print("Iteration {0}/{1}".format(j, len(dct_add)))
        print("{0}".format("*"*40))
        j += 1

        # Lemmat. words
        words, list_lem, list_lem_ngrams, list_lem_complete = word_preprocessing(data['text'])
        # Add affective features
        data['affective_features'], _, _, _ = affective_features(data['text'], list_lem, list_lem_complete, list_lem_ngrams)
        data['id_doc'] = ''
        data['file_path'] = ''
        data['year'] = data['date']
        del data['date']
        # Rename cols
        data.update({'words':words,
                     'words_lem':list_lem, 
                     'words_lem_ngrams':list_lem_ngrams,
                     'words_lem_complete':list_lem_complete,
                     'file_path':''})
        
        # string of words (ngram)
        s = ''
        for w in list_lem_ngrams:
            s += ' '
            s += w
        data.update({'ngrams2str':s})
        
        # string of words (lem)
        s = ''
        for w in list_lem:
            s += ' '
            s += w
        data.update({'lem2str':s})
        
        # Parse stanzas
        list_aux = [s.strip() for s in data['text'].splitlines()]
        stanza = ''
        dct_stanzas = {}
        key = 0
        for line in list_aux:
            if line != '':
                stanza += line
                stanza += '\n'
            else:
                dct_stanzas[key] = {'text':stanza}
                key += 1
                stanza = ''
        
        # Last row
        if len(dct_stanzas) <= 3:
            dct_stanzas[key] = {'text':stanza}
            key += 1
            stanza = ''
        
        del key 
        data['dct_stanzas'] = dct_stanzas

        ##### Per Stanza
        for key, dct_stanza in data['dct_stanzas'].items():
            words, list_lem, list_lem_ngrams, list_lem_complete = word_preprocessing(dct_stanza['text'])
            # Add affective features
            data['dct_stanzas'][key]['affective_features'], _, _, _ = affective_features(dct_stanza['text'], list_lem, list_lem_complete, list_lem_ngrams)
            
            # Rename cols
            data['dct_stanzas'][key].update({'words':words,
                                             'words_lem':list_lem, 
                                             'words_lem_ngrams':list_lem_ngrams,
                                             'words_lem_complete':list_lem_complete
                                             })
            # string of words (ngram)
            s = ''
            for w in list_lem_ngrams:
                s += ' '
                s += w
            data['dct_stanzas'][key].update({'ngrams2str':s})
            
            # string of words (lem)
            s = ''
            for w in list_lem:
                s += ' '
                s += w
            data['dct_stanzas'][key].update({'lem2str':s})

        # Append info and go to new iteration
        dct_sonnets[num_sonnet] = data
        num_sonnet += 1
        
    
    # Store in disk
    file_presistance(PATH + "/"+ DOCS_NAME_DCT + ".p", "generic", dct_sonnets, "save")
    
    return dct_sonnets
            

def generate_vocab():
    """
    # TODO
    Function to generate vocabulary from the available sonnets
    
    """
    
    # Load sonnets
    try:
        dct_sonnets = file_presistance(PATH + "/"+ DOCS_NAME_DCT + ".p", "generic", None, "load")
        print("Sonnets loaded")
    except:
        print("Sonnets not yet extracted... extracting them")
        dct_sonnets = docs2dict()
        
    vocab_total = []
    vocab_lem_total = []
    vocab_lem_ngrams_total = []
    vocab_lem_with_stopwords = []
    
    num_doc = 0
    for key,sonnet in dct_sonnets.items():
        num_doc += 1
        print("{0}".format("*"*40))
        print("Iteration {0}/{1}".format(num_doc, len(dct_sonnets)))
        print("{0}".format("*"*40))
        
        words, _, _, _ = word_preprocessing(sonnet['text'])
        vocab_total = list(set(vocab_total + words))
        vocab_lem_total = list(set(vocab_lem_total + sonnet['words_lem']))
        vocab_lem_ngrams_total = list(set(vocab_lem_ngrams_total + sonnet['words_lem_ngrams']))
        vocab_lem_with_stopwords = list(set(vocab_lem_with_stopwords + sonnet['words_lem_complete']))
        
    dct_results = {'vocab_total':vocab_total, 
                   'vocab_lem_total':vocab_lem_total, 
                   'vocab_lem_ngrams_total':vocab_lem_ngrams_total,
                   'vocab_lem_with_stopwords':vocab_lem_with_stopwords}
    
    file_presistance(os.path.join(os.getcwd(), PATH, FILE_NAME_DCT_RESULTS + '.p'), "generic", dct_results, "save")
    
    



def affective_features(text, vocab, vocab_lem, vocab_lem_ngrams):
    """
    # TODO
    Function to modify the xml files and add affective features
    """
    words, words_lem, words_lem_ngrams, _ = word_preprocessing(text)
    
    # Add to vocab
    vocab = list(set(vocab + words))
    vocab_lem = list(set(vocab_lem + words_lem))
    vocab_lem_ngrams = list(set(vocab_lem_ngrams + words_lem_ngrams))
    
    #######
    # Affective Feature Extractor
    #######
    word_features_tot = {}
    
    ### Emotions
    
    # Load data
    name = PATH + '/' +  NAME1 + '.csv' 
    df = pd.read_csv(name, sep=';', encoding='utf-8', header=0).rename(columns={'Spanish_Word':WORD_COL}).drop(["English_Translation", "N"], axis=1).set_index(WORD_COL)
    
    # Add same features for ngrams
    df_aux = pd.DataFrame()
    for j,row in df.iterrows():
        list_aux = list(set([x for w in [j] for x in word_grams(w, len(w)-1, len(w))])) # Lower ngrams only of one word
        for val in list_aux:
            aux = pd.DataFrame(row).T
            aux = aux.reset_index()
            aux['index'] = val
            aux = aux.set_index('index')
            df_aux = df_aux.append(aux)
    
    # Concat
    df = df.append(df_aux).reset_index().drop_duplicates(subset='index').set_index('index').sort_index()
    
    # Obtain features for that poem
    list_words_df = list(df.index)
    for w in words_lem_ngrams:
        # Word in df
        if w in list_words_df:
            word_features = df[df.index== w].to_dict('index')[w]
        else:
#                    print("word '{}' not in df".format(w))
            continue # not consider this word
        # Add features values for each word
        if not bool(word_features_tot):
            word_features_tot = word_features.copy()
        else:
            for key,value in word_features.items():
                word_features_tot[key] += value
                word_features_tot[key] = np.round(word_features_tot[key], 2)
                
     
    ### Valence/Arousal metrics
    name = PATH + '/' +  NAME4 + '.csv' 
    df = pd.read_csv(name, sep=';', encoding='utf-8', header=0).rename(columns={'Word':WORD_COL}).drop(["ValenceRaters", "ArousalRaters"], axis=1).fillna(0)
    
    # Add same features for ngrams
    df_aux = pd.DataFrame()
    for j,row in df.set_index(WORD_COL).iterrows():
        list_aux = list(set([x for w in [j] for x in word_grams(w, len(w)-1, len(w))])) # Lower ngrams only of one word
        for val in list_aux:
            aux = pd.DataFrame(row).T
            aux = aux.reset_index()
            aux['index'] = val
            aux = aux.set_index('index')
            df_aux = df_aux.append(aux)
    
    # Concat
    df_aux = df_aux.reset_index().rename({'index':WORD_COL}, axis=1)
    df = df.append(df_aux).drop_duplicates(subset=WORD_COL).sort_index()
    
    df_words = pd.DataFrame({WORD_COL:words_lem_ngrams})
    df_words_feat = df.merge(df_words, how='right').dropna()
    
    ## Features: Mean(Aro), SD(Aro), Mean(Val), SD(Val)
    df = df.set_index(WORD_COL)
    list_words_df = list(df.index)
    word_features_aux = {}
    for w in words_lem_ngrams:
        # Word in df
        if w in list_words_df:
            word_features = df[df.index== w].to_dict('index')[w]
        else:
#                    print("word '{}' not in df".format(w))
            continue # not consider this word
        
        # Add features values for each word
        if not bool(word_features_aux):
            word_features_aux = word_features.copy()
        else:
            for key,value in word_features.items():
                word_features_aux[key] += value
                word_features_aux[key] = np.round(word_features_aux[key], 2)
    
    # Add features to total
    word_features_tot = {**word_features_tot, **word_features_aux}
    
    ## Features: Min(Aro), Max(Aro), Min(Val), Max(Val), ValenceSpan, ArousalSpan
    word_features_tot['Max_Aro'] = df_words_feat['ArousalMean'].max()
    word_features_tot['Min_Aro'] = df_words_feat['ArousalMean'].min()
    word_features_tot['ArousalSpan'] = np.round(abs(word_features_tot['Max_Aro'] - word_features_tot['Min_Aro']), 4)
    word_features_tot['Max_Val'] = df_words_feat['ValenceMean'].max()
    word_features_tot['Min_Val'] = df_words_feat['ValenceMean'].min()
    word_features_tot['ValenceSpan'] = np.round(abs(word_features_tot['Max_Val'] - word_features_tot['Min_Val']), 4)
    
    ## Features: Vector Correlation (Aro), Vector Correlation (Val)
    word2idx = [k for k in range(len(words))]
    arousal_idx = [df_words_feat[df_words_feat[WORD_COL]==w]['ArousalMean'].values[0]  if w in list(df_words_feat[WORD_COL]) else np.nan for w in words_lem]
    valence_idx = [df_words_feat[df_words_feat[WORD_COL]==w]['ValenceMean'].values[0]  if w in list(df_words_feat[WORD_COL]) else np.nan for w in words_lem]
    df_aux = pd.DataFrame({'words':words_lem, 'idx':word2idx, 'arousal':arousal_idx, 'valence':valence_idx}).dropna()
    CorAro = np.round(df_aux[['idx', 'arousal']].corr()['arousal']['idx'], 4)
    CorVal = np.round(df_aux[['idx', 'valence']].corr()['valence']['idx'], 4)
    
    word_features_tot['CorAro'] = CorAro
    word_features_tot['CorVal'] = CorVal
    
    ## Features: Nº words
    N_Words = len(words)
    word_features_tot['N_Words'] = N_Words
    
    
    ### AoA (Age of Acquisition, Oral Freq, Written Freq)
    
    # Load data
    name = PATH + '/' +  NAME3 + '.csv' 
    df = pd.read_csv(name, sep=';', encoding='utf-8', header=0).rename(columns={'word':WORD_COL})
    df = df[[WORD_COL, "averageAoA", "SD", "Min", "Max", "OralFreq_Log", " WrittenFreq_LEXESP_Log"]].rename(columns={'SD':'sdAOA', 'Min':'MinAoA', 'Max':'MaxAoA', ' WrittenFreq_LEXESP_Log': 'WrittenFreq_Log'})
    
    # Add same features for ngrams
    df_aux = pd.DataFrame()
    for j,row in df.set_index(WORD_COL).iterrows():
        list_aux = list(set([x for w in [j] for x in word_grams(w, len(w)-1, len(w))])) # Lower ngrams only of one word
        for val in list_aux:
            aux = pd.DataFrame(row).T
            aux = aux.reset_index()
            aux['index'] = val
            aux = aux.set_index('index')
            df_aux = df_aux.append(aux)
    
    # Concat
    df_aux = df_aux.reset_index().rename({'index':WORD_COL}, axis=1)
    df = df.append(df_aux).drop_duplicates(subset=WORD_COL).sort_index()
    
    df_words = pd.DataFrame({WORD_COL:words_lem_ngrams})
    df_words_feat = df.merge(df_words, how='right').dropna()
    
    
    ## Features: Min(AoA), Max(AoA), AoASpan
    word_features_tot['Max_AoA'] = df_words_feat['averageAoA'].max()
    word_features_tot['Min_AoA'] = df_words_feat['averageAoA'].min()
    word_features_tot['AoASpan'] = np.round(abs(word_features_tot['Max_AoA'] - word_features_tot['Min_AoA']), 4)
    
    ## Features: Min(sdAOA), Max(sdAOA), sdAOA
    word_features_tot['Max_sdAOA'] = np.round(df_words_feat['sdAOA'].max(), 4)
    word_features_tot['Min_sdAOA'] = np.round(df_words_feat['sdAOA'].min(), 4)
    word_features_tot['sdAOASpan'] = np.round(abs(word_features_tot['Max_sdAOA'] - word_features_tot['Min_sdAOA']), 4)
    
    ## Features: Min(OralFreq), Max(OralFreq), OralFreqSpan
    word_features_tot['Max_logOralFreq'] = np.round(df_words_feat['OralFreq_Log'].max(), 4)
    word_features_tot['Min_logOralFreq'] = np.round(df_words_feat['OralFreq_Log'].min(), 4)
    word_features_tot['logOralFreqSpan'] = np.round(abs(word_features_tot['Max_logOralFreq'] - word_features_tot['Min_logOralFreq']), 4)
    
    ## Features: Min(WrittenFreq), Max(WrittenFreq), WrittenFreqSpan
    word_features_tot['Max_logWrittenFreq'] = df_words_feat['WrittenFreq_Log'].max()
    word_features_tot['Min_logWrittenFreq'] = df_words_feat['WrittenFreq_Log'].min()
    word_features_tot['logWrittenFreqSpan'] = np.round(abs(word_features_tot['Max_logWrittenFreq'] - word_features_tot['Min_logWrittenFreq']), 4)
    
    ## Features: Vector Correlation (AoA)
    word2idx = [k for k in range(len(words))]
    aoa_idx = [df_words_feat[df_words_feat[WORD_COL]==w]['averageAoA'].values[0]  if w in list(df_words_feat[WORD_COL]) else np.nan for w in words_lem]
    df_aux = pd.DataFrame({'words':words_lem, 'idx':word2idx, 'aoa':aoa_idx}).dropna()
    CorAoA = np.round(df_aux[['idx', 'aoa']].corr()['aoa']['idx'], 4)
    word_features_tot['CorAoA'] = CorAoA
  
    ## Totals
    df = df.set_index(WORD_COL)[["averageAoA", "sdAOA"]]
    list_words_df = list(df.index)
    word_features_aux = {}
    for w in words_lem_ngrams:
        # Word in df
        if w in list_words_df:
            word_features = df[df.index== w].to_dict('index')[w]
        else:
#                    print("word '{}' not in df".format(w))
            continue # not consider this word
        
        # Add features values for each word
        if not bool(word_features_aux):
            word_features_aux = word_features.copy()
        else:
            for key,value in word_features.items():
                word_features_aux[key] += value
                word_features_aux[key] = np.round(word_features_aux[key], 2)
    
    word_features_tot = {**word_features_tot, **word_features_aux}

    return word_features_tot, vocab, vocab_lem, vocab_lem_ngrams



def feature_extractor_affective():
    """
    Function to process all the documents available in DISCO per-author and include in them
    a list of affective features such as:
        # TODO
    
    
    Parameters
    -------------------
    
    Returns
    -------------------
    
    
    """
    # List of documents
    try:
        list_docs_original = file_presistance(os.path.join(os.getcwd(), PATH, 'list_files.p'), 'generic', None, 'load')
        list_docs =  get_files(PATH_POEMS, "xml")
        list_docs = [x for x in list_docs if x not in list_docs_original]
    except:
        print("Nothing used yet, loading all files")
        list_docs =  get_files(PATH_POEMS, "xml")
        list_docs_original = []
#        
#    vocab_total = []
#    vocab_lem_total = []
#    vocab_lem_ngrams_total = []
#    
#    # Load stored sonnets
#    try:
#        print("Loading stored sonnets")
#        dct_sonnets = file_presistance(PATH + "/"+ DOCS_NAME_DCT + ".p", "generic", None, "load")
#        print("Sonnets loaded")
#        
#    except:
#        print("Sonnet dictionary is not generated yet... generating it first")
#        dct_sonnets = docs2dict()
#        print("Sonnet dictionary generated!")
#

    num_poems = 0
    # Iterate through all documents
    for file_path in list_docs:
        num_poems += 1
        print("{0}".format("*"*40))
        print("Iteration {0}/{1}".format(num_poems, len(list_docs)))
        print("{0}".format("*"*40))
        
        # Load file
        doc = file_presistance(file_path, "xml", None, "load")
        try:
            print("author", doc["TEI"]["text"]["front"]["div"]["head"])
            print("country & date", doc["TEI"]["text"]["front"]["div"]["p"])
        except:
            print("Some meta fields not present in this document")
        
        # Poems
        i = 0
        
        docs = [] # total documents
        vocab = []
        vocab_lem = []
        vocab_lem_ngrams = []
        
        # Iterate through poems in that doc
        iter_dict = doc["TEI"]["text"]["body"]
        j = 0
        
        # Metadata
        author = doc['TEI']['text']['front']['div']['head']
        year = doc['TEI']['text']['front']['div']['p']
        
        for doc_poem in iter_dict.values():
            ################
            # One Sonnet per Author
            ################
            if 'lg' in doc_poem:
                # Metadata
                title = doc_poem['head']
                id_doc = doc_poem['@'+'xml:id']
                
                # Sonnet sequence
                if doc_poem['@'+'type'] == 'sonnet-sequence':
                    # with two or more parts
                    if type(doc_poem["lg"])==list: 
                        for sonnet in doc_poem["lg"]:
                            text = parse_poem(sonnet)
                            word_features_tot, vocab, vocab_lem, vocab_lem_ngrams = affective_features(text, vocab, vocab_lem, vocab_lem_ngrams)
                            docs.append(text)
                            k = 0
                            for key, value in word_features_tot.items():
                                doc["TEI"]["text"]["body"]["lg"]["lg"][k]["param"] = {"@name": "AffectiveFeatures", "attRef":[]}
                                doc["TEI"]["text"]["body"]["lg"]["lg"][k]["param"]["attRef"].append({"@name":key, "content":str(value)})
                            k += 1
                    # with one part
                    else:
                        for key, value in doc_poem.items():
                            if key=='lg':
                                sonnet=value
                                text = parse_poem(sonnet)
                                word_features_tot, vocab, vocab_lem, vocab_lem_ngrams = affective_features(text, vocab, vocab_lem, vocab_lem_ngrams)
                                docs.append(text)
                                doc["TEI"]["text"]["body"]["lg"]["param"] = {"@name": "AffectiveFeatures", "attRef":[]}
                                for key, value in word_features_tot.items():
                                    doc["TEI"]["text"]["body"]["lg"]["param"]["attRef"].append({"@name":key, "content":str(value)})
                              
                # In other case
                else:
                    text = parse_poem(doc_poem)
                    word_features_tot, vocab, vocab_lem, vocab_lem_ngrams = affective_features(text, vocab, vocab_lem, vocab_lem_ngrams)
                    docs.append(text)
                    doc["TEI"]["text"]["body"]["lg"]["param"] = {"@name": "AffectiveFeatures", "attRef":[]}  
                    for key, value in word_features_tot.items():
                        doc["TEI"]["text"]["body"]["lg"]["param"]["attRef"].append({"@name":key, "content":str(value)})
                         
            
            ###############
            # More than one sonnet per Author
            ##############
            else:
                for doc_sonnet in doc_poem:
                    # Metadata
                    title = doc_sonnet['head']
                    id_doc = doc_sonnet['@'+'xml:id']
                    
                    # Unique sonnets
                    if 'lg' in doc_sonnet and doc_sonnet['@'+'type']=='sonnet':
                        text = parse_poem(doc_sonnet)
                        docs.append(text)
                        word_features_tot, vocab, vocab_lem, vocab_lem_ngrams = affective_features(text, vocab, vocab_lem, vocab_lem_ngrams)
                        doc["TEI"]["text"]["body"]["lg"][j]["param"] = {"@name": "AffectiveFeatures", "attRef":[]}
                        for key, value in word_features_tot.items():
                            doc["TEI"]["text"]["body"]["lg"][j]["param"]["attRef"].append({"@name":key, "content":str(value)})
                         
                        j += 1
                
                    # Sonnet sequence  
                    elif 'lg' in doc_sonnet and doc_sonnet['@'+'type']=='sonnet-sequence':
                        # with two or more parts
                        if type(doc_sonnet["lg"])==list: 
                            k = 0
                            for sonnet in doc_sonnet["lg"]:
                                text = parse_poem(sonnet)
                                word_features_tot, vocab, vocab_lem, vocab_lem_ngrams = affective_features(text, vocab, vocab_lem, vocab_lem_ngrams)
                                docs.append(text)
                                for key, value in word_features_tot.items():
                                    doc["TEI"]["text"]["body"]["lg"][j]["lg"][k]["param"] = {"@name": "AffectiveFeatures", "attRef":[]}
                                    doc["TEI"]["text"]["body"]["lg"][j]["lg"][k]["param"]["attRef"].append({"@name":key, "content":str(value)})
                                k += 1
                            j += 1
                        # with one part
                        else:
                             k = 0
                             for key, value in doc_sonnet.items():
                                if key=='lg':
                                    sonnet=value
                                    text = parse_poem(sonnet)
                                    word_features_tot, vocab, vocab_lem, vocab_lem_ngrams = affective_features(text, vocab, vocab_lem, vocab_lem_ngrams)
                                    docs.append(text)
                                    for key, value in word_features_tot.items():
                                        doc["TEI"]["text"]["body"]["lg"][k]["param"] = {"@name": "AffectiveFeatures", "attRef":[]}
                                        doc["TEI"]["text"]["body"]["lg"][k]["param"]["attRef"].append({"@name":key, "content":str(value)})
                                    k += 1
                             j += 1                    
            i += 1
        
            ### Save new XML
            file_presistance(file_path, "xml", doc, "save")
        
            # Append doc name to general 
            list_docs_original.append(file_path)
            file_presistance(PATH + '/list_files.p', "generic", list_docs_original, "save")
            
            
            
        
def obtain_embedding_matrix(file_path, vocab_lem, whole=False):
    """
    # TODO
    Function to obtain the embedding matrix from FastText corresponding to the vocabulary (extended with ngrams) from the poems
    stored in the DB.
    
    It can also obtain the whole matrix corresponding to all the words available in FastText.
    
    """
    ### Load Embeddings
    # Load vectors
    wordvectors = KeyedVectors.load_word2vec_format(os.path.join(os.getcwd(), PATH, WORD_EMBEDDING_FILE))
    
    # Obtain all n-grams
    vocab_ext = list(set([x for w in vocab_lem for x in word_grams(w, 2, len(w) + 1)]))
    list_vocab = list(wordvectors.vocab.keys())
    df_word_embedding = pd.DataFrame.from_dict(dict((w, wordvectors[w]) for w in vocab_ext if w in list_vocab)).T
    df_word_embedding = df_word_embedding.append(pd.DataFrame({'UNK':[0]*df_word_embedding.shape[1]}).T) # add UNK
    
    # Save new embedding
    dct_results = file_presistance(file_path, "generic", None, "load")
    dct_results['vocab_ngrams_total_available'] = list_vocab
    dct_results['df_word_embedding'] = df_word_embedding
    dct_results['df_word_embedding_whole'] = pd.DataFrame()
    
    if whole:
        df_word_embedding = pd.DataFrame.from_dict(dict((w, wordvectors[w]) for w in list_vocab)).T
        df_word_embedding = df_word_embedding.append(pd.DataFrame({'UNK':[0]*df_word_embedding.shape[1]}).T) # add UNK
        dct_results['df_word_embedding_whole'] = df_word_embedding

    file_presistance(file_path, "generic", dct_results, "save")


def text2bert_embedding(model, examples, tokenizer, local_rank, batch_size, device, layer_indexes):
    """
    # TODO
    Obtain embeddings for a single piece of text
    """
    
    # Obtain features
    max_length = max([len(examples[i].text_a) for i in range(len(examples))])
    # Upper limit to avoid overflow errors
    if max_length >=512:
        max_length = 512
    
    features = convert_examples_to_features(
          examples=examples, seq_length=max_length, tokenizer=tokenizer)
    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature
    
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)
    if local_rank == -1:
        eval_sampler = SequentialSampler(eval_data)
    else:
        eval_sampler = DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)
    model.eval()
    
    # Obtain Word Embeddings
    j = 0
    output_json_all = collections.OrderedDict()
    for input_ids, input_mask, example_indices in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        all_encoder_layers, _ = model(input_ids, token_type_ids=None, attention_mask=input_mask)
        all_encoder_layers = all_encoder_layers
        
        # Iterate per piece of text considered
        for b, example_index in enumerate(example_indices):
            feature = features[example_index.item()]
            unique_id = int(feature.unique_id)
            output_json = collections.OrderedDict()
            output_json["linex_index"] = unique_id
            all_out_features = []
            
            # Iterate per word in that piece of text and obtain embeddings for it
            for (i, token) in enumerate(feature.tokens):
                all_layers = []
                for (j, layer_index) in enumerate(layer_indexes):
                    layer_output = all_encoder_layers[int(layer_index)].detach().cpu().numpy()
                    layer_output = layer_output[b]
                    layers = collections.OrderedDict()
                    layers["index"] = layer_index
                    layers["values"] = [
                        round(x.item(), 6) for x in layer_output[i]
                    ]
                    all_layers.append(layers)
                out_features = collections.OrderedDict()
                out_features["token"] = token
                out_features["layers"] = all_layers
                all_out_features.append(out_features)
                
            # Add all embedings for that piece of text
            output_json["features"] = all_out_features
        # Append to general dict
        output_json_all[j] = output_json
        j += 1
        
    return output_json_all
    


def _chunks(data, SIZE=10000):
    """
    This function does a yield of the original dict (casted as an iteritem) in different
    batches. In the first i-th iteration it yields (taking it out from the original dict) the
    first num=SIZE elements, in the next i-th iteration since the original dict is without those
    first num=SIZE elements it takes out the next num=SIZE elements, and so on.
    """
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k:data[k] for k in islice(it, SIZE)} 
        
    

def obtain_bert_embeddings(file_path, type_embedding, generate_new=False):
    """
    # TODO
    Function to obtain the embeddings for each sonnet according to BERT. Each sonnet will
    have their own embedding for each of the words involved in it.
    
    """   
    
    ###########################################################################
    # NOT USED
    
    ### Define Bert model
    # Use only one layer
    layers = "-1"
    layer_indexes = [int(x) for x in layers.split(",")]
    # Define tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
    local_rank = -1
    batch_size = 10
    # Train model
    model = BertModel.from_pretrained('bert-base-multilingual-cased')
    device = torch.device("cpu")
    model.to(device)
    ###########################################################################
    
    # List of documents (original)
    dct_sonnets_all = file_presistance(PATH + "/"+ DOCS_NAME_DCT + ".p", "generic", None, "load")
    
    # Split original documents
    batch_size = BERT_BATCH_SIZE
    batch = int(np.round(len(dct_sonnets_all)/batch_size))
    list_splits = []
    
    # Using bert word embedding recovery library
    bert_embedding = BertEmbedding(model='bert_12_768_12', dataset_name='wiki_multilingual')
#    bert_embedding = BertEmbedding()

            
    for item in _chunks(dct_sonnets_all, batch):
        list_splits.append(item)
    
    # Generate new parts
    if generate_new:
        # Save splited documents
        file_presistance(PATH + "/"+ DOCS_NAME_DCT + '_split_' + ".p", "generic", list_splits, "save")
        
        # Create and save original splitted parts
        j = 0
        for dct_sonnets in list_splits:
            print("{0}".format("-"*40))
            print("Iteration {0}/{1}".format(j, len(list_splits)-1))
            print("{0}".format("-"*40))
            
            # Save doc
            file_presistance(PATH + "/"+ DOCS_NAME_DCT + '_bert_{0}_part_{1}'.format(type_embedding, j) + ".p", "generic", dct_sonnets, "save")
            j += 1

    # Embeddings for each subset
    j=0
    for list_aux in list_splits:
        print("{0}".format("-"*40))
        print("Iteration {0}/{1}".format(j, len(list_splits)-1))
        print("{0}".format("-"*40))

        # List of documents
        try:
            list_docs_used = file_presistance(PATH + "/"+ "list_sonnets" + '_bert_{0}_part_{1}'.format(type_embedding, j) + ".p", 'generic', None, 'load')
            dct_sonnets = file_presistance(PATH + "/"+ DOCS_NAME_DCT + '_bert_{0}_part_{1}'.format(type_embedding, j) + ".p", "generic", None, "load")
        except:
            print("Nothing used yet, loading all files")
            dct_sonnets = file_presistance(PATH + "/"+ DOCS_NAME_DCT + '_bert_{0}_part_{1}'.format(type_embedding, j) + ".p", "generic", None, "load")
            list_docs_used = []
               
        k = 0
        ### Apply to texts
        for key, sonnet in dct_sonnets.items():
            print("{0}".format("*"*40))
            print("Sub-iteration {0}/{1}".format(k, len(dct_sonnets)))
            print("{0}".format("*"*40))
            
            # If it's already embedded, skip this iteration
            if key in list_docs_used:
                k += 1
                continue
            
            if type_embedding=="whole_text":
                # Embedding of whole text
#                text = sonnet['text']
#                examples = read_text(text, multiple_lines=False)
#                output_json_all = text2bert_embedding(model, examples, tokenizer, local_rank, batch_size, device, layer_indexes)
#                dct_sonnets[key]["bert_embedding_text"] = output_json_all
#                
                
                # Embedding without stopwords and lemmatized [Stanza]
                for key_stanza,stanza in sonnet['dct_stanzas'].items():
                    result = bert_embedding(stanza['words']) 
                    dct_sonnets[key]['dct_stanzas'][key_stanza]["whole_text"] = result
                    
                    
                result = bert_embedding(sonnet['words'])                                    
                dct_sonnets[key]["whole_text"] = result
                
            elif type_embedding=="words_lem_complete":
                # Embedding of text lemmatized with stopwords and uppercase
                text = ''
                for w in sonnet['words_lem_complete']:
                    text += w
                    text += ' '
                examples = read_text(text, multiple_lines=False)
                output_json_all = text2bert_embedding(model, examples, tokenizer, local_rank, batch_size, device, layer_indexes)
                dct_sonnets[key]["bert_embedding_lem_complete"] = output_json_all
            
            elif type_embedding=="words_lem_nonstopwords":
                
                # Embedding without stopwords and lemmatized [Stanza]
                for key_stanza,stanza in sonnet['dct_stanzas'].items():
                    text = ''
                    for w in stanza['words_lem']:
                        text += w
                        text += ' '
                    
#                    examples = read_text(text, multiple_lines=False)
#                    output_json_all = text2bert_embedding(model, examples, tokenizer, local_rank, batch_size, device, layer_indexes)
                    
                    # Using BERT embedding library
                    
                    result = bert_embedding(stanza['words_lem'])                    
                    dct_sonnets[key]['dct_stanzas'][key_stanza]["bert_embedding_lem_nonstopwords"] = result
                    
                
                # Embedding without stopwords and lemmatized [Whole Text]
                text = ''
                for w in sonnet['words_lem']:
                    text += w
                    text += ' '
#                examples = read_text(text, multiple_lines=False)
#                output_json_all = text2bert_embedding(model, examples, tokenizer, local_rank, batch_size, device, layer_indexes)
                result = bert_embedding(sonnet['words_lem'])                                    
                dct_sonnets[key]["bert_embedding_lem_nonstopwords"] = result
            
            # Append doc name to general 
            list_docs_used.append(key)
            file_presistance(PATH + '/list_sonnets' + '_bert_{0}_part_{1}'.format(type_embedding, j) + '.p', "generic", list_docs_used, "save")
        
            # Save sonnets information
            file_presistance(PATH + "/"+ DOCS_NAME_DCT + '_bert_{0}_part_{1}'.format(type_embedding, j) + ".p", "generic", dct_sonnets, "save")
            
            k += 1
            
        # Next split
        j += 1
        
        
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
    
  
def bert_embedding_composition(type_embedding, composition_type):
    """
    # TODO
    Perform composition over the embeddings of a text
    """
    
    if LOAD_SPLITS:
        # Load sonnets embeddings
        list_bert_sonnets = get_files(PATH, "p")
        if type_embedding=="whole_text":
            list_bert_sonnets = [doc for doc in list_bert_sonnets if ("whole_text" in doc) and ("dct_sonnets" in doc)]
        elif type_embedding=="words_lem_nonstopwords":
            list_bert_sonnets = [doc for doc in list_bert_sonnets if ("nonstopwords" in doc) and ("dct_sonnets" in doc) and ("affective" in doc)]
            # list_bert_sonnets = [doc for doc in list_bert_sonnets if ("nonstopwords" in doc) and ("dct_sonnets" in doc) and ("affective" not in doc)]
        else:
            raise NotImplementedError()
            
        # Join all splitted in a single variable
        dct_sonnets_all = {}
        for path in list_bert_sonnets:
            dct_sonnets_aux = file_presistance(path, "generic", None, "load")
            if len(dct_sonnets_all)==0:
                dct_sonnets_all = dct_sonnets_aux
            else:
                dct_sonnets_all.update(dct_sonnets_aux)
    else:
        dct_sonnets_all = file_presistance(PATH + "/"+ DOCS_NAME_DCT + '_bert_{0}_all_words'.format(type_embedding) + ".p", "generic", None, "load")
            
    # Obtain compositions for each sonnet
    df_embeddings_all_sonnets = pd.DataFrame()
    for key, sonnet in dct_sonnets_all.items():
        df_embeddings_all = pd.DataFrame()
        
        if type_embedding=="whole_text":
            word_emb = sonnet['bert_embedding_text'][0]['features']
        elif type_embedding=="words_lem_nonstopwords":
            word_emb = sonnet['bert_embedding_lem_nonstopwords'][0]['features']
        else:
            raise NotImplementedError()
            
        for word in word_emb:
            # Obtain embeddings for that word
            df_embeddings = pd.DataFrame(word['layers'][0]['values']).T
            # Append embeddings of each word
            if df_embeddings_all.empty:
                df_embeddings_all = df_embeddings
            else:
                df_embeddings_all = df_embeddings_all.append(df_embeddings)
                
        # Apply composition function
        if composition_type=="sum":
            df_embeddings_all = pd.DataFrame(df_embeddings_all.sum()).T
        elif composition_type=="mean":
            df_embeddings_all = pd.DataFrame(df_embeddings_all.mean()).T
        elif composition_type=="joint":
            # Inverse order (RL_SEQ)
            df_embeddings_all.reset_index(inplace=True)
            del df_embeddings_all['index']
            df_embeddings_all.sort_index(ascending=False, inplace=True)
            key_max = df_embeddings_all.index.max()
            v_follow = [0]*df_embeddings_all.shape[1]
            # Iterate per word
            for key1, vi in df_embeddings_all.iterrows():
                # First vector is not joint to anyone yet
                if key1 == key_max:
                    v_follow = vi
                else:
                    v_follow = joint_function(v_follow,vi)
            df_embeddings_all = pd.DataFrame(v_follow).T

        # Set index as key value for that sonnet
        df_embeddings_all = df_embeddings_all.reset_index()
        df_embeddings_all['index'] = key
        df_embeddings_all = df_embeddings_all.set_index('index')
        # Append to general dict
        if df_embeddings_all_sonnets.empty:
            df_embeddings_all_sonnets = df_embeddings_all
        else:
            df_embeddings_all_sonnets = df_embeddings_all_sonnets.append(df_embeddings_all)
            
    
    # Load if existing
    try: 
        dct_composition_embeddings= file_presistance(PATH + '/' + 'dct_composition_embeddings_{0}_{1}.p'.format(composition_type, type_embedding), "generic", None, "load")
    except:
        dct_composition_embeddings = {}
    # Update
    dct_composition_embeddings[composition_type] = df_embeddings_all_sonnets
    # Save
    file_presistance(PATH + '/' + 'dct_composition_embeddings_{0}_{1}.p'.format(composition_type, type_embedding), "generic", dct_composition_embeddings, "save")
    
    
    # Obtain compositions for each stanza
    df_embeddings_all_stanzas = pd.DataFrame()


def embed_bert_text(word_emb, key, composition_type):
    """
    # TODO
    """
    df_embeddings_all = pd.DataFrame()
    
    for word in word_emb:
        # Obtain embeddings for that word
#        df_embeddings = pd.DataFrame(word['layers'][0]['values']).T
        df_embeddings = pd.DataFrame(word[0]).T
        # Append embeddings of each word
        if df_embeddings_all.empty:
            df_embeddings_all = df_embeddings
        else:
            df_embeddings_all = df_embeddings_all.append(df_embeddings)
            
    # Apply composition function
    if composition_type=="sum":
        df_embeddings_all = pd.DataFrame(df_embeddings_all.sum()).T
    elif composition_type=="mean":
        df_embeddings_all = pd.DataFrame(df_embeddings_all.mean()).T
    elif composition_type=="joint":
        # Inverse order (RL_SEQ)
        df_embeddings_all.reset_index(inplace=True)
        del df_embeddings_all['index']
        df_embeddings_all.sort_index(ascending=False, inplace=True)
        key_max = max(list(df_embeddings_all.index))
        v_follow = [0]*df_embeddings_all.shape[1]
        # Iterate per word
        for key1, vi in df_embeddings_all.iterrows():
            # First vector is not joint to anyone yet
            if key1 == key_max:
                v_follow = vi
            else:
                v_follow = joint_function(v_follow,vi)
        df_embeddings_all = pd.DataFrame(v_follow).T

    # Set index as key value for that sonnet
    df_embeddings_all = df_embeddings_all.reset_index()
    df_embeddings_all['index'] = key
    df_embeddings_all = df_embeddings_all.set_index('index')
    
    return df_embeddings_all


def bert_embedding_composition_iter(type_embedding, composition_type):
    """
    # TODO
    Perform composition over the embeddings of a text
    """

    if LOAD_SPLITS:
        # Load sonnets embeddings
        list_bert_sonnets = get_files(PATH, "p")
        if type_embedding=="whole_text":
            list_bert_sonnets = [doc for doc in list_bert_sonnets if ("whole_text" in doc) and ("dct_sonnets" in doc) and ("affective" not in doc) and ('all_words' not in doc)]
        elif type_embedding=="words_lem_nonstopwords":
    #        list_bert_sonnets = [doc for doc in list_bert_sonnets if ("nonstopwords" in doc) and ("dct_sonnets" in doc) and ("affective" in doc)]
#             list_bert_sonnets = [doc for doc in list_bert_sonnets if ("nonstopwords" in doc) and ("dct_sonnets" in doc) and ("affective" not in doc) and ('all_words' not in doc)]
             list_bert_sonnets = [doc for doc in list_bert_sonnets if ("whole_text" in doc) and ("dct_sonnets" in doc) and ("affective" not in doc) and ('all_words' not in doc)]
        else:
            raise NotImplementedError()
        
        df_embeddings_all_sonnets = pd.DataFrame()
        df_embeddings_all_stanzas = pd.DataFrame()
        j = 0
        # Join all splitted in a single variable
        for path in list_bert_sonnets:
            dct_sonnets_all = file_presistance(path, "generic", None, "load")
            
            # Obtain compositions for each sonnet
            for key, sonnet in dct_sonnets_all.items():
                print("{0}".format("-"*40))
                print("Iteration {0}/{1}".format(j, len(dct_sonnets_all)*len(list_bert_sonnets)))
                print("{0}".format("-"*40))
                j += 1
                
                #########
                # Sonnets
                #########
                
                if type_embedding=="whole_text":
#                    word_emb = sonnet['bert_embedding_text'][0]['features']
                    # word_preprocessing
                    word_emb = [x[1] for x in sonnet['whole_text']]
                elif type_embedding=="words_lem_nonstopwords":
#                    word_emb = sonnet['bert_embedding_lem_nonstopwords'][0]['features']
                    # word_preprocessing(text)
                    sonnet['bert_embedding_lem_nonstopwords'] = []
                    
                    for k in range(len(sonnet['whole_text'])):
                        if word_preprocessing(sonnet['whole_text'][k][0][0])[1] != []:
                            sonnet['bert_embedding_lem_nonstopwords'].append(sonnet['whole_text'][k])
                            
                    word_emb = [x[1] for x in sonnet['bert_embedding_lem_nonstopwords']]
                else:
                    raise NotImplementedError()
                    
                # Text embedding
                df_embedding_all = embed_bert_text(word_emb, key, composition_type=composition_type)
    
                # Append to general dict
                if df_embeddings_all_sonnets.empty:
                    df_embeddings_all_sonnets = df_embedding_all
                else:
                    df_embeddings_all_sonnets = df_embeddings_all_sonnets.append(df_embedding_all)
            
            
                #########
                # Stanzas
                #########
                for key_stanza, stanza in sonnet['dct_stanzas'].items():
                    
                    if type_embedding=="whole_text":
#                        word_emb = stanza['bert_embedding_text'][0]['features']
                        word_emb = [x[1] for x in stanza['whole_text']]
                    elif type_embedding=="words_lem_nonstopwords":
                        
                        stanza['bert_embedding_lem_nonstopwords'] = []
                        
                        for k in range(len(stanza['whole_text'])):
                            if word_preprocessing(stanza['whole_text'][k][0][0])[1] != []:
                                stanza['bert_embedding_lem_nonstopwords'].append(stanza['whole_text'][k])
                            
                        word_emb = [x[1] for x in stanza['bert_embedding_lem_nonstopwords']]
                    else:
                        raise NotImplementedError()
                    
                    # Text embedding
                    df_embedding_all = embed_bert_text(word_emb, key, composition_type=composition_type)
                    df_embedding_all['stanza'] = key_stanza
                    
                    # Append to general dict
                    if df_embeddings_all_stanzas.empty:
                        df_embeddings_all_stanzas = df_embedding_all
                    else:
                        df_embeddings_all_stanzas = df_embeddings_all_stanzas.append(df_embedding_all)
                
        
        # Load if existing
        try: 
            dct_composition_embeddings= file_presistance(PATH + '/' + 'dct_composition_embeddings_{0}_{1}.p'.format(composition_type, type_embedding), "generic", None, "load")
        except:
            dct_composition_embeddings = {}
        # Update
        dct_composition_embeddings[composition_type] = df_embeddings_all_sonnets
        # Save
        file_presistance(PATH + '/' + 'dct_composition_embeddings_{0}_{1}.p'.format(composition_type, type_embedding), "generic", dct_composition_embeddings, "save")
    
    
        # Load if existing
        try: 
            dct_composition_embeddings= file_presistance(PATH + '/' + 'dct_composition_embeddings_stanza_{0}_{1}.p'.format(composition_type, type_embedding), "generic", None, "load")
        except:
            dct_composition_embeddings = {}
        # Update
        dct_composition_embeddings[composition_type] = df_embeddings_all_stanzas
        # Save
        file_presistance(PATH + '/' + 'dct_composition_embeddings_stanza_{0}_{1}.p'.format(composition_type, type_embedding), "generic", dct_composition_embeddings, "save")
    


    
    
def obtain_fasttext_embedding(file_path, type_embedding, generate_new=False):
    """
    # TODO
    
    """
    
    # List of documents (original)
    dct_sonnets_all = file_presistance(PATH + "/"+ DOCS_NAME_DCT + ".p", "generic", None, "load")
    
    # Split original documents
    batch_size = BERT_BATCH_SIZE
    batch = int(np.round(len(dct_sonnets_all)/batch_size))
    list_splits = []
            
    for item in _chunks(dct_sonnets_all, batch):
        list_splits.append(item)
        
    # Generate new parts
    if generate_new:
        # Save splited documents
        file_presistance(PATH + "/"+ DOCS_NAME_DCT + '_split_' + ".p", "generic", list_splits, "save")
        
        # Create and save original splitted parts
        j = 0
        for dct_sonnets in list_splits:
            print("{0}".format("-"*40))
            print("Iteration {0}/{1}".format(j, len(list_splits)-1))
            print("{0}".format("-"*40))
            
            # Save doc
            file_presistance(PATH + "/"+ DOCS_NAME_DCT + '_fasttext_part_{0}'.format(j) + ".p", "generic", dct_sonnets, "save")
            j += 1
            
    ## Load vectors
    wordvectors = KeyedVectors.load_word2vec_format(os.path.join(os.getcwd(), PATH, WORD_EMBEDDING_FILE))
    list_vocab = wordvectors.vocab
    df_word_embedding = pd.DataFrame.from_dict(dict((w, wordvectors[w]) for w in list_vocab)).T
    del wordvectors
    
    # Split 
    j=0
    for dct_sonnets in list_splits:
        print("{0}".format("-"*40))
        print("Iteration {0}/{1}".format(j, len(list_splits)-1))
        print("{0}".format("-"*40))

        # List of documents
        try:
            list_docs_used = file_presistance(PATH + "/"+ "list_sonnets" + '_fasttext_part_{0}'.format(j) + ".p", 'generic', None, 'load')
            dct_sonnets = file_presistance(PATH + "/"+ DOCS_NAME_DCT + '_fasttext_part_{0}'.format(j) + ".p", "generic", None, "load")
        except:
            print("Nothing used yet, loading all files")
            dct_sonnets = file_presistance(PATH + "/"+ DOCS_NAME_DCT + '_fasttext_part_{0}'.format(j) + ".p", "generic", None, "load")
            list_docs_used = []
            
        k = 0
        ### Apply to texts
        df_embeddings_all = pd.DataFrame()
        for key, sonnet in dct_sonnets.items():
            print("{0}".format("*"*40))
            print("Sub-iteration {0}/{1}".format(k, len(dct_sonnets)))
            print("{0}".format("*"*40))
            k += 1

            # If it's already embedded, skip this iteration
            if key in list_docs_used:
                continue
            
            df_embeddings_all = pd.DataFrame()
            # Obtain embeddings
            words_lem = sonnet['words_lem']
            for word in words_lem:
                aux = word
                # If it does not exist use an ngram and embed with the longest word
                if word not in list_vocab:
                    list_ngrams = word_grams(word, lim_min=1, lim_max=len(word)+1)
                    list_ngrams.reverse()
                    for word in list_ngrams:
                        if word in list_vocab:
                            break
                # Embedding
                try:
                    df_aux = pd.DataFrame(df_word_embedding.loc[word]).T
                except:
                    print("word {0} not in vocab".format(aux))
                
                if df_embeddings_all.empty:
                    df_embeddings_all = df_aux
                else:
                    df_embeddings_all = df_embeddings_all.append(df_aux)
                    
                dct_sonnets[key]['df_embedding_fasttext'] = df_embeddings_all
                list_docs_used.append(key)
                
        file_presistance(PATH + "/"+ "list_sonnets" + '_fasttext_part_{0}'.format(j) + ".p", 'generic', list_docs_used, 'save')
        file_presistance(PATH + "/"+ DOCS_NAME_DCT + '_fasttext_part_{0}'.format(j) + ".p", 'generic', dct_sonnets, 'save')
        j += 1
                
    # File persistance
#    file_presistance(PATH + "/dct_embedding_fasttext.p", "generic", dct_sonnets_all, "save")
    file_presistance(PATH + "/df_fasttext_matrix.p", "generic", df_word_embedding, "save")
    file_presistance(PATH + "/df_fasttext_vocab.p", "generic", list_vocab, "save")
    
    
    ## Obtain all n-grams
    #vocab_ext = list(set([x for w in vocab_prueba for x in word_grams(w, 2, len(w) + 1)]))
#    list_vocab = list(wordvectors.vocab.keys())
#    df_word_embedding = pd.DataFrame.from_dict(dict((w, wordvectors[w]) for w in vocab_ext if w in list_vocab)).T
#    ## Add UNK column
#    df_word_embedding = df_word_embedding.append(pd.DataFrame({'UNK':[0]*df_word_embedding.shape[1]}).T)
    #dct_results_prueba = {}
    #dct_results_prueba['vocab_ngrams_total'] = list_vocab
    #dct_results_prueba['df_word_embedding'] = df_word_embedding
    #dct_results_prueba['n_grams'] = vocab_ext
    #dct_results_prueba['words_featured'] = list(df_word_embedding.index)
    #file_presistance("dct_results_prueba.p", "generic", dct_results_prueba, "save")
    
    
def fasttext_embedding_composition(composition_type):
    """
    # TODO
    Perform composition over the embeddings of a text
    """
    
    # Load sonnets embeddings
    list_fasttext_sonnets = get_files(PATH, "p")
    list_fasttext_sonnets = [doc for doc in list_fasttext_sonnets if ("fasttext" in doc) and ("dct_sonnets" in doc)]

    # Join all splitted in a single variable
    dct_sonnets_all = {}
    for path in list_fasttext_sonnets:
        dct_sonnets_aux = file_presistance(path, "generic", None, "load")
        if len(dct_sonnets_all)==0:
            dct_sonnets_all = dct_sonnets_aux
        else:
            dct_sonnets_all.update(dct_sonnets_aux)
               
    # Obtain compositions for each sonnet
    df_embeddings_all_sonnets = pd.DataFrame()
    j = 1
    for key, sonnet in dct_sonnets_all.items():
        
        print("{0}".format("-"*40))
        print("Iteration {0}/{1}".format(j, len(dct_sonnets_all)-1))
        print("{0}".format("-"*40))
        j +=1
    
        df_embeddings_all = sonnet['df_embedding_fasttext']
        # Apply composition function
        if composition_type=="sum":
            df_embeddings_all = pd.DataFrame(df_embeddings_all.sum()).T
        elif composition_type=="mean":
            df_embeddings_all = pd.DataFrame(df_embeddings_all.mean()).T
        elif composition_type=="joint":
            # Inverse order (RL_SEQ)
            df_embeddings_all.reset_index(inplace=True)
            del df_embeddings_all['index']
            df_embeddings_all.sort_index(ascending=False, inplace=True)
            key_max = df_embeddings_all.index.max()
            v_follow = [0]*df_embeddings_all.shape[1]
            # Iterate per word
            for key1, vi in df_embeddings_all.iterrows():
                # First vector is not joint to anyone yet
                if key1 == key_max:
                    v_follow = vi
                else:
                    v_follow = joint_function(v_follow,vi)
            df_embeddings_all = pd.DataFrame(v_follow).T

        # Set index as key value for that sonnet
        df_embeddings_all = df_embeddings_all.reset_index()
        df_embeddings_all['index'] = key
        df_embeddings_all = df_embeddings_all.set_index('index')
        # Append to general dict
        if df_embeddings_all_sonnets.empty:
            df_embeddings_all_sonnets = df_embeddings_all
        else:
            df_embeddings_all_sonnets = df_embeddings_all_sonnets.append(df_embeddings_all)
    
    # Load if existing
    try: 
        dct_composition_embeddings= file_presistance(PATH + '/' + 'dct_composition_embeddings_{0}_{1}.p'.format(composition_type, "fasttext"), "generic", None, "load")
    except:
        dct_composition_embeddings = {}
    # Update
    dct_composition_embeddings[composition_type] = df_embeddings_all_sonnets
    # Save
    file_presistance(PATH + '/' + 'dct_composition_embeddings_{0}_{1}.p'.format(composition_type, "fasttext"), "generic", dct_composition_embeddings, "save")
    