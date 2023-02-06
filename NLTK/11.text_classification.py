# =============================================================================
# Metin Siniflandirilmasini, ornegin; metni siyasi veya ordu ile ilgili olarak
# siniflandirmaya calisiyoruz. Belki de onu yazarin cinsiyetine gore siniflandirmaya
# calisiyoruz.
# Populer bir metin siniflandirmaya ornek verecek olursak, e-posta filtreleri gibi
# seyler icin bir metin govdesinin spam olup olmadigini belirtmek.
# =============================================================================

# Duyarlilik Analizi Algoritmasi olusturmaya calisalim
# film inceleme veritabanini kullanacagiz

import nltk
import random
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

print(documents[1])
print('\n\n')
all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
print(all_words.most_common(15)) # En cok gecen kelimeler ve sayisi
print('\n\n')
print(all_words['open']) # Kac tane open kelimesi geciyor?
