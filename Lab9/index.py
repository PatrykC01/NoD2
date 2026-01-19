from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

# Wczytywanie danych z określonych kategorii
data = fetch_20newsgroups(subset='train', categories=['rec.autos', 'rec.sport.baseball'], 
                          remove=('headers', 'footers', 'quotes'))
texts = data.data

# Wektoryzacja TF-IDF z usuwaniem stop-words
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(texts)

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Oczyszczona tokenizacja dla lepszych efektów Word2Vec i LDA
sentences = [[t for t in word_tokenize(doc.lower()) if t.isalpha() and t not in stop_words] 
             for doc in texts if len(doc) > 5]

# Trenowanie modelu Word2Vec
model_w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=5)
print(f"Podobieństwo 'car' i 'engine': {model_w2v.wv.similarity('car', 'engine')}")

from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel

# Przygotowanie korpusu bag-of-words
dictionary = Dictionary(sentences)
corpus = [dictionary.doc2bow(s) for s in sentences]

# Trenowanie modelu LDA 
lda = LdaModel(corpus, num_topics=2, id2word=dictionary, passes=15)

# Prezentacja słów kluczowych dla tematów
for idx, topic in lda.print_topics():
    print(f"Temat {idx}: {topic}")

from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

# Redukcja do 2 komponentów
svd = TruncatedSVD(n_components=2)
X_2d = svd.fit_transform(tfidf_matrix)

# Wizualizacja z kolorowaniem według rzeczywistych kategorii
plt.figure(figsize=(10, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=data.target, cmap='viridis', alpha=0.5)
plt.title("Reprezentacja dokumentów w 2D (SVD)")
plt.colorbar(label="Kategoria (0=Autos, 1=Baseball)")
plt.show()
