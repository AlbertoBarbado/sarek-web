# SAREK (Sonnet Affective Recommender with Embedding Knowledge)
This is Sarek, a personal sonnet recommender for spanish poetry!

This repo is a web app that allows users to search for poems with a query that could consist in either a unique word or a sentence. After receiving it, Sarek searchs for the most relevant sonnet from a corpus of Spanish poetry sonnets (dating from S.XV to S.XX) and shows that poem in the web.

The retrieval is done using a joint function of the individual words of a stanza and with BERT embedding to assign each word an individual vector. The similarity among stanzas and the query text is computed using cosine similarity.

## 1. Requirements
* Python 3.6.1 
* Pip

## 2. Setup
After cloning/downloading the repo install the requirements
```
$ pip install -r requirements.txt
```

Then execute the app.py file

```
$ python app.py
```

And then use any web browser to access the following webpage:

```
localhost:5000
```
There you can interact with the system. Sometimes it takes a while to retrieve the sonnets (the code could be further optimized), so be patient.

Enjoy!
