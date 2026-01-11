import nltk
import string
from nltk.corpus import reuters, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import FreqDist
from string import punctuation

docs = reuters.fileids()
train_docs_ids = [doc for doc in docs if doc.startswith('training/')]
frs_doc = train_docs_ids[0]
raw_doc = reuters.raw(frs_doc)

print('Raw sample from 1st document: ')
print(raw_doc[:300] + "...")


print("\n\n\n")

#tokenization, stemming/lemmatization και stop-word removal και αφαίρεση ειδικών χαρακτήρων

# Tokenization

tkn_doc = word_tokenize(raw_doc[:300])
print("Sample from tokenized document: ")
print(tkn_doc)

# Stemming/Lemmatization

print("\n\n\n")

stemmer = PorterStemmer()
print("Stemming on a tokenized document: ")
for word in tkn_doc:
    print(stemmer.stem(word))

print("\n\n")

lemmatizer = WordNetLemmatizer()
print("Lemmatization on words in document: ")
for word in tkn_doc:
    lem = lemmatizer.lemmatize(word)
    print(lem)

# Stop-Word and special characters removal

print("\n\n\n")

print("Stop-words: ")
stop_words = set(stopwords.words('english'))
reword = [word for word in tkn_doc if word not in stop_words]
print(reword)

print("\n\n")
print("Special characters removal: ")
spc_c = punctuation
spcword = [word for word in tkn_doc if word not in spc_c]
print(spcword)