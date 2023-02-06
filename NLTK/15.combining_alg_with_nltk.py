import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from nltk.classify import ClassifierI
from statistics import mode # En populer oyu secme yontemimiz 'Mod'

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, NuSVC

class VoteClassifier(ClassifierI):
    
    def __init__(self, *classifiers):
        self._classifiers = classifiers
    
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf
    

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    
    for w in word_features:
        features[w] = (w in words)
    
    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

training_set = featuresets[:1900]
testing_set = featuresets[1900:]

# Naive Bayes Classifier Load
classifier_f = open('naivebayes.pickle', 'rb')
classifier = pickle.load(classifier_f)
classifier_f.close()
print('Original Naive Bayes Accuracy Percent: ',\
      (nltk.classify.accuracy(classifier, testing_set)) * 100)

# Multinomial Classifier
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print('Multinomial Classifier Accuracy Percent: ',\
      (nltk.classify.accuracy(MNB_classifier, testing_set)) * 100)

# Bernoulli Classifier
BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print('Bernoulli Classifier Accuracy Percent: ',\
      (nltk.classify.accuracy(BNB_classifier, testing_set)) * 100)

# Logistic Regression Classifier
log_reg_classifier = SklearnClassifier(LogisticRegression())
log_reg_classifier.train(training_set)
print('Logistic Regression Classifier Accuracy Percent: ',\
      (nltk.classify.accuracy(log_reg_classifier, testing_set)) * 100)
    
# SGD Classifier
sgd_classifier = SklearnClassifier(SGDClassifier())
sgd_classifier.train(training_set)
print('SGD Classifier Accuracy Percent: ',\
      (nltk.classify.accuracy(sgd_classifier, testing_set)) * 100)

# LinearSVC Classifier
linear_svc_classifier = SklearnClassifier(LinearSVC())
linear_svc_classifier.train(training_set)
print('LinearSVC Classifier Accuracy Percent: ',\
      (nltk.classify.accuracy(linear_svc_classifier, testing_set)) * 100)

# NuSVC Classifier
nu_svc_classifier = SklearnClassifier(NuSVC())
nu_svc_classifier.train(training_set)
print('NuSVC Classifier Accuracy Percent: ',\
      (nltk.classify.accuracy(nu_svc_classifier, testing_set)) * 100)

voted_classifier = VoteClassifier(classifier,
                                  MNB_classifier,
                                  BNB_classifier,
                                  log_reg_classifier,
                                  sgd_classifier,
                                  linear_svc_classifier,
                                  nu_svc_classifier)

print('Voted Classifier Accuracy Percent: ',\
      (nltk.classify.accuracy(voted_classifier, testing_set)) * 100)

print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:",voted_classifier.confidence(testing_set[0][0])*100)
print("Classification:", voted_classifier.classify(testing_set[1][0]), "Confidence %:",voted_classifier.confidence(testing_set[1][0])*100)
print("Classification:", voted_classifier.classify(testing_set[2][0]), "Confidence %:",voted_classifier.confidence(testing_set[2][0])*100)
print("Classification:", voted_classifier.classify(testing_set[3][0]), "Confidence %:",voted_classifier.confidence(testing_set[3][0])*100)
print("Classification:", voted_classifier.classify(testing_set[4][0]), "Confidence %:",voted_classifier.confidence(testing_set[4][0])*100)
print("Classification:", voted_classifier.classify(testing_set[5][0]), "Confidence %:",voted_classifier.confidence(testing_set[5][0])*100)
