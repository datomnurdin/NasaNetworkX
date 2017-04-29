import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string

# Importing Gensim
import gensim
from gensim import corpora

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()
doc_complete = []

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

data = pd.read_json('data.json')

data.columns

data.head()

file = open("doc_complete.txt", "w")

for result in data['dataset']:
    result[u'description'] = result[u'description']
    doc_complete.append(result[u'description'])
    
file.write(str(doc_complete))

doc_clean = [clean(doc).split() for doc in doc_complete]  

# Creating the term dictionary of our courpus, where every unique term is assigned an index. 
dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=221, id2word = dictionary, passes=50)

print(ldamodel.print_topics(num_topics=221, num_words=9))