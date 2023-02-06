# =============================================================================
# Metin Siniflandirmada kullanilan oldukca populer olan Naive Bayes Classifier
# ile bir text classification yapalim.
# Dikkat etmememiz gereken husus; ayni verilerle hem train hem test yapmamaliyiz.
# Yaparsak, bu bize ciddi onyargi sorunlari sunar.
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

# Naive Bayes Classifier
training_set = featuresets[:1900]
test_set = featuresets[1900:]

classifier = nltk.NaiveBayesClassifier.train(training_set)

print('Classifier accuracy percent: ', (nltk.classify.accuracy(classifier, test_set)) * 100)
print('\n\n')

# Her kelime icin olumsuzdaki olusumlarin pozitife orani veya tam tersidir.
# Mesela 'hakaret' teriminin olumlu yorumlara gore olumsuz yorumlarda 10,6 kat daha
# fazla gectigini gorebiliriz.
classifier.show_most_informative_features(15)


import pickle

# Modelimizi Kaydedelim
save_classifier = open('naivebayes.pickle', 'wb')
pickle.dump(classifier, save_classifier)
save_classifier.close()


# Kaydettigimiz modelimizi kullanalim
classifier_f = open('naivebayes.pickle', 'rb')
classifier = pickle.load(classifier_f)
classifier_f.close()
