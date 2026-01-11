import nltk
from nltk.corpus import reuters

try:
    nltk.data.find('corpora/reuters')
    print("Το σύνολο δεδομένων Reuters βρέθηκε ήδη.")
except LookupError:
    print("Γίνεται λήψη του συνόλου δεδομένων Reuters...")
    nltk.download('reuters')
    print("Η λήψη ολοκληρώθηκε.")

print("-" * 30)

documents = reuters.fileids()
train_docs_ids = [doc for doc in documents if doc.startswith('training/')]

print(f"Συνολικά έγγραφα: {len(documents)}")
print(f"Έγγραφα εκπαίδευσης (Training): {len(train_docs_ids)}")

sample_doc_id = train_docs_ids[0] 

print(f"\n--- Επιθεώρηση Εγγράφου: {sample_doc_id} ---")

categories = reuters.categories(sample_doc_id)
print(f"Κατηγορίες: {categories}")

raw_text = reuters.raw(sample_doc_id)

print("\nΑπόσπασμα Κειμένου:")
print(raw_text[:300] + "...")