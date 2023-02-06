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

# Naive Bayes Classifier
training_set = featuresets[:1900]
testing_set = featuresets[1900:]

# Original Naive Bayes
classifier = nltk.NaiveBayesClassifier.train(training_set)
print('Original Naive Bayes Accuracy Percent: ',\
      (nltk.classify.accuracy(classifier, testing_set)) * 100)
classifier.show_most_informative_features(15)

# Butun sklearn siniflandiricilarini kullanabiliriz
from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.naive_bayes import MultinomialNB, BernoulliNB

# MultinomialNB
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print('\nMultinomialNB Classifier Accuracy Percent: ',\
      (nltk.classify.accuracy(MNB_classifier, testing_set)) * 100)

# BernoulliNB
BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print('BernoulliNB Classifier Accuracy Percent: ',\
      (nltk.classify.accuracy(BNB_classifier, testing_set)) * 100)

# Daha fazla sklearn siniflandiricilarini kullanalim
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

# Logistic Regression
log_reg_classifier = SklearnClassifier(LogisticRegression())
log_reg_classifier.train(training_set)
print('Logistic Regression Accuracy Percent: ',\
      (nltk.classify.accuracy(log_reg_classifier, testing_set)) * 100)

# SGD
sgd_classifier = SklearnClassifier(SGDClassifier())
sgd_classifier.train(training_set)
print('SGD Classifier Accuracy Percent: ',\
      (nltk.classify.accuracy(sgd_classifier, testing_set)) * 100)

# SVC
svc_classifier = SklearnClassifier(SVC())
svc_classifier.train(training_set)
print('SVC Classifier Accuracy Percent: ',\
      (nltk.classify.accuracy(svc_classifier, testing_set)) * 100)

# LinearSVC
linear_svc_classifier = SklearnClassifier(LinearSVC())
linear_svc_classifier.train(training_set)
print('Linear SVC Classifier Accuracy Percent: ',\
      (nltk.classify.accuracy(linear_svc_classifier, testing_set)) * 100)

# NuSVC
nusvc_classifier = SklearnClassifier(NuSVC())
nusvc_classifier.train(training_set)
print('NuSVC Classifier Accuracy Percent: ',\
      (nltk.classify.accuracy(nusvc_classifier, testing_set)) * 100)
