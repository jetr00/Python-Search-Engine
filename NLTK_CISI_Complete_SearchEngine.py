import nltk
import string
from nltk.corpus import reuters, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

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


def blsc(tstem, dictionary):
    stemmer = PorterStemmer()
    if 'and' in tstem:
        cstem = []
        astem = []
        for i in tstem:
            if i != 'and':
                cstem.append(i)
        for j in cstem:
            astem.append(stemmer.stem(j))
        if len(astem) > 0 and astem[0] in dictionary:
            setc = set(dictionary[astem[0]])
            for k in range(1, len(astem)):
                wcheck = astem[k]
                if wcheck in dictionary:
                    setc = setc & set(dictionary[wcheck])
                else:
                    print("\nΔεν βρέθηκαν αποτελέσματα (AND).")
            if setc != []:
                print(list(setc))

    elif 'or' in tstem:
        cstem = []
        astem = []
        for i in tstem:
            if i != 'or':
                cstem.append(i)
        for j in cstem:
            astem.append(stemmer.stem(j))
        ff = False
        setc = set()
        for word in astem:
            if word in dictionary:
                setc = set(dictionary[word])
                ff = True
                break
        if ff:
            for wcheck in astem:
                if wcheck in dictionary:
                    setc = setc | set(dictionary[wcheck])
            print(list(setc))
        else:
            print("Δεν βρέθηκαν αποτελέσματα (OR).")
            
    elif 'not' in tstem:
        cstem = []
        astem = []
        for i in tstem:
            if i != 'not':
                cstem.append(i)
        for j in cstem:
            astem.append(stemmer.stem(j))
        if len(astem) > 0 and astem[0] in dictionary:
            setc = set(dictionary[astem[0]])
            for k in range(1, len(astem)):
                wcheck = astem[k]
                if wcheck in dictionary:
                    setc = setc - set(dictionary[wcheck])
            print(list(setc))
        else:
            print("Δεν βρέθηκαν αποτελέσματα.")

def comp(sr, tfvect, vdoc, dids):
    fd = False
    sq = tfvect.transform([sr])
    cosim = cosine_similarity(sq, vdoc)
    results = cosim.flatten()
    sort = results.argsort()[::-1]
    print("Αποτέλεσματα VSM:")
    for i in sort[:10]:
        if results[i] > 0.1:
            print(f"Document: {dids[i]} with a result of: {results[i]:.4}")
            fd = True
    if not fd:
        print("Δεν βρέθηκαν αποτελέσματα.")

def rank(tdoc, sr, dids):
    tsr = word_tokenize(sr)
    scores = bm25.get_scores(tsr)
    sort = scores.argsort()[::-1]
    print("\nΑποτελέσματα Rank_bm25: ")
    fd = False
    for i in sort[:10]:
        if scores[i] > 0:
            print(f"Document: {dids[i]} με Score: {scores[i]:.4f}")
            fd = True
    if not fd:
        print("Δεν βρέθηκαν αποτελέσματα.")

stemmer = PorterStemmer()
dictionary = {}
ddocs = reuters.fileids()
tfvect = TfidfVectorizer()
raw_text = []
documents = []

data = input("\nΕπιλογή συνόλου δεδομένων (1. Reuters-21578 Corpus, 2. CISI Dataset.): ")

if int(data) == 1:
    print("Προετοιμασία Reuters-21578 Corpus κειμένων...")
    train_docs_ids = [doc for doc in ddocs if doc.startswith('training/')]
    documents = train_docs_ids
    ndocs = len(documents)

    for i in range(len(documents)):
        raw_text.append(reuters.raw(documents[i]))
    print("\nΤα κείμενα είναι έτοιμα.")
elif int(data) == 2:
    print("Προετοιμασία CISI κειμένων...")
    with open("CISI.ALL", 'r') as f:
        lines = f.readlines()
    cisi_dict = {}
    doc_id = ""
    text = ""

    for line in lines:
        if line.startswith(".I"):
            if doc_id:
                cisi_dict[doc_id] = text
            doc_id = line.strip().split(" ")[1]
            text = ""
        elif line.startswith(".X"):
            pass
        elif line.startswith(".T") or line.startswith(".A") or line.startswith(".W"):
            continue
        else:
            text += line.strip() + " "
    if doc_id:
        cisi_dict[doc_id] = text

    for did, txt in cisi_dict.items():
        documents.append(did)
        raw_text.append(txt)
    ndocs = len(documents)
    print("\nΤα κείμενα είναι έτοιμα.")

print("\nΠροετοιμασία VSM (TF-IDF)...")
ldoc = tfvect.fit_transform(raw_text)
print("\nΤο VSM είναι έτοιμο.")

print("\nΠροετιμασία Rank_bm25...")
trt = []
for i in range(len(raw_text)):
    trt.append(word_tokenize(raw_text[i]))
bm25 = BM25Okapi(trt)
print("\nΤο Rank_bm25 είναι έτοιμο.")

print("\nΠροετοιμασία Boolean Dictionary...")
for i in range(ndocs):
    words_doc = editing(raw_text[i])
    for words in words_doc:
        if words not in dictionary:
            dictionary[words] = [documents[i]]
        else:
            if documents[i] not in dictionary[words]:
                dictionary[words].append(documents[i])
print("\nΤο Boolean Dictionary είναι έτοιμο.")

app = input("\nΕπιλογή για λειτουργία (1. Boolean Search (AND/OR/NOT), 2. VSM, 3. Κανονική λειτουργία και 4. Rank_bm25). ")
search = input("\nΑναζήτηση: ")

while search != "ss" or app != "ss":
    tstem = word_tokenize(search)
    sstem = [w.lower() for w in tstem]
    if int(app) == 1:
        if 'and' in sstem or 'or' in sstem or 'not' in sstem:
            blsc(sstem, dictionary)
        else:
            print("\nΔεν βρέθηκε τελεστής (AND/OR/NOT).")
    elif int(app) == 2:
        comp(search, tfvect, ldoc, documents)
    elif int(app) == 3:
        if len(tstem) > 0:
            sstem = stemmer.stem(tstem[0].lower())
        if sstem in dictionary:
            print(f"\nΒρέθηκαν {len(dictionary[sstem])} έγγραφα.")
            print(dictionary[sstem])
        else:
            print(f"\nΗ λέξη {search} δεν υπάρχει στο ευρυτήριο.")
    elif int(app) == 4:
        rank(bm25, search, documents)
    else:
        print("\nΗ επιλογή που κάνατε δεν υπάρχει. Επιλέξτε 1. Boolean Search (AND/OR/NOT), 2. VSM, 3. Κανονική λειτουργία ή 4. Rank_bm25 και ss για έξοδο. ")
    app = input("\nΕπιλογή για νέα λειτουργία (1. Boolean Search (AND/OR/NOT), 2. VSM, 3. Κανονική λειτουργία και 4. Rank_bm25). ")
    search = input("\nΑναζήτηση: ")