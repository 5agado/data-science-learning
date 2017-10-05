import re
import os
import numpy as np
from bs4 import BeautifulSoup

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def bags_of_words(docs, max_features=2000):
    vectorizer = CountVectorizer(analyzer = "word", max_features = max_features,
                                 stop_words="english")
    data = vectorizer.fit_transform(docs).toarray()
    return vectorizer, data

# For now used to get the words with highest value from a TfidfVectorizer
def get_top_features(vectorizer, fit_matrix, n):
    feature_names = np.array(vectorizer.get_feature_names())
    matrix_sorted = np.argsort(fit_matrix)

    indexes = matrix_sorted[(-n):]#[::-1] if you want precisely ordered by relevance
    return feature_names[indexes]

def clean_text(in_txt, rem_html=True, rem_punct=True, rem_numbers=True, to_lower=True, rem_stopwords=True):
    # Remove HTML content with BeautifulSoup
    if rem_html:
        in_txt = BeautifulSoup(in_txt, "html.parser").get_text()
    # Remove non-letters
    out_txt = re.sub("[^a-zA-Z]", " ", in_txt)
    # Convert to lower case words
    words = out_txt.lower().split()
    # Remove stop words
    if rem_stopwords:
        stops = set(stopwords.words("english"))
        out_words = [w for w in words if not w in stops]

    return out_words

def sentence_generator(dirpath, filenames=None):
    if not filenames:
        filenames = os.listdir(dirpath)

    for name in filenames:
        with open(os.path.join(dirpath, name)) as f:
            for line in f:
                yield line.split()


from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
# Get most relevant words using TF-IDF
# For this statistic we need additional pieces of text to compare with our speech transcript
# we can simply load some corpora from NLTK
def get_vectorizer(text, n):
    # Load corpora for different genres
    c1 = nltk.corpus.gutenberg.raw('carroll-alice.txt')
    c2 = nltk.corpus.inaugural.raw("2009-Obama.txt")
    c3 = nltk.corpus.webtext.raw("firefox.txt")
    # Load english stopwords
    stops = set(stopwords.words("english"))

    # Compute TF-IDF matrix and print top results for our speech
    vectorizer = TfidfVectorizer(analyzer='word',stop_words=stops)
    tfIdf = vectorizer.fit_transform([text, c1, c2, c3]).toarray()
    indices = np.argsort(tfIdf[0])[::-1]
    features = vectorizer.get_feature_names()
    top_features = [features[i] for i in indices[:n] if tfIdf[0][i]!=0]
    print(top_features)