# =============================================================================
# Olumlu veya olumsuz incelemelerdeki kelimelerin ozellik listelerini derleyelim
# =============================================================================

import nltk
import random
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

def findFeatures(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

print((findFeatures(movie_reviews.words('neg/cv000_29416.txt'))))
print('\n\n')
featuresets = [(findFeatures(rev), category) for (rev, category) in documents]

