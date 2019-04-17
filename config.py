# -*- coding: utf-8 -*-
"""
@author: Alberto Barbado González
@mail: alberto.barbado.gonzalez@gmail.com
"""

PATH_POEMS = "poems"
PATH = 'datasets'
PATH_USER = 'user_results'
WORD_COL = 'word'

NAME1 = 'Ferre2017_Article_MovedByWordsAffectiveRatings'
NAME2 = 'Guasch2016_Article_SpanishNormsForAffective'
NAME3 = 'SpanishAoA'
NAME4 = 'Stadthagen_Gonzalez2017_Article_NormsOfValenceAndArousal'
WORD_EMBEDDING_FILE = 'fasttext-sbwc.3.6.e20.vec'

FILE_NAME_DCT_RESULTS = 'df_fasttext_matrix'
FILE_NAME_VOCAB = 'df_fasttext_vocab'
DOCS_NAME_DCT = 'dct_sonnets'
OTHER_SONNETS = 'poems/other_sonnets'
SONNETS_ADDITIONAL = "dct_sonnets_sxx"
BERT_BATCH_SIZE = 10
COMPOSITION_TYPE="sum"
#TYPE_EMBEDDING = "fasttext"
#TYPE_EMBEDDING = "whole_text"
TYPE_EMBEDDING = "words_lem_nonstopwords"
LOAD_SPLITS = False
JOBLIB_PARAMS = {'n_jobs':2, 'verbose':0, 'backend':"loky"}
