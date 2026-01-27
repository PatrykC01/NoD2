import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

# 1. Wczytywanie danych 
data = fetch_20newsgroups(subset='train', categories=['sci.space', 'sci.med'], 
                          remove=('headers', 'footers', 'quotes'))
texts = data.data

# 2. Wektoryzacja TF-IDF 
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(texts)

# 3. Przygotowanie narzędzi do czyszczenia tekstu
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

additional_stop_words = {'get', 'also', 'like', 'think', 'would', 'know', 'people', 'one', 'could', 'use', 'many', 'say', 'make', 'may', 'year', 'time', 'see', 'good', 'new', 'even', 'want', 'way', 'still', 'take', 'come', 'first', 'two', 'well', 'much', 'right', 'look', 'give', 'day', 'work', 'seem', 'need', 'system', 'never', 'part', 'little', 'help', 'find', 'something', 'lot', 'better', 'nothing', 'might', 'since', 'last', 'thing', 'put', 'ask', 'great', 'world', 'tell', 'end', 'try', 'run', 'keep', 'different', 'high', 'big', 'start', 'long', 'point', 'without', 'around', 'however', 'another', 'example', 'fact', 'give', 'set', 'group', 'number', 'course', 'several', 'always', 'rather', 'although', 'yet', 'often', 'away', 'least', 'however'}
stop_words.update(additional_stop_words)

# 4. Tokenizacja z lematyzacją
sentences = []
for doc in texts:
    if len(doc) > 5:
      
        tokens = [lemmatizer.lemmatize(t.lower()) for t in word_tokenize(doc) 
                  if t.isalpha() and t.lower() not in stop_words]
        sentences.append(tokens)

# 5. Trenowanie modelu Word2Vec
model_w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=5)
print(f"Podobieństwo 'space' i 'orbit': {model_w2v.wv.similarity('space', 'orbit')}")

# 6. Trenowanie modelu LDA
dictionary = Dictionary(sentences)
corpus = [dictionary.doc2bow(s) for s in sentences]
lda = LdaModel(corpus, num_topics=2, id2word=dictionary, passes=15)

print("\n--- TEMATY LDA ---")
for idx, topic in lda.print_topics():
    print(f"Temat {idx}: {topic}")

# 7. Wizualizacja SVD
svd = TruncatedSVD(n_components=2)
X_2d = svd.fit_transform(tfidf_matrix)

plt.figure(figsize=(10, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=data.target, cmap='viridis', alpha=0.5)
plt.title("Reprezentacja dokumentów w 2D (SVD)")
plt.colorbar(label="Kategoria (0=Space, 1=Medicine)")
plt.show()
