import nltk
import string
from nltk.corpus import reuters, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from string import punctuation

def editing(raw_doc):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    spc = punctuation

    tdoc = word_tokenize(raw_doc)
    sdoc = []
    reword = []
    for word in tdoc:
        if word not in stop_words and word not in spc:
            reword.append(word)
    for word in reword:
        sdoc.append(stemmer.stem(word))
    return sdoc

dictionary = {}
ddocs = reuters.fileids()
train_docs_ids = [doc for doc in ddocs if doc.startswith('training/')]
ndocs = len(train_docs_ids)
for i in range(1, ndocs):
    rawdoc = reuters.raw(train_docs_ids[i])
    words_doc = editing(rawdoc)
    for words in words_doc:
        if words not in dictionary:
            dictionary[words] = [train_docs_ids[i]]
        else:
            if train_docs_ids[i] not in dictionary[words]:
                dictionary[words].append(i)

print(f"Το ευρετήριο φτιάχτηκε! Έχει {len(dictionary)} λέξεις.")