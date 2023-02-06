# =============================================================================
# Lemmatizing, Stemming e cok benzer.
# Aralarindaki en buyuk fark, kok cikarmak genellikle var olmayan sozcukler
# olusturabilir oysa Lemmalar gercek sozcuklerdir.
# =============================================================================

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize('cats'))
print(lemmatizer.lemmatize('skirts'))
print(lemmatizer.lemmatize('cacti'))
print(lemmatizer.lemmatize('rocks'))
print(lemmatizer.lemmatize('python'))
print(lemmatizer.lemmatize('better', pos='a'))
print(lemmatizer.lemmatize('best', pos='a'))
print(lemmatizer.lemmatize('run'))
print(lemmatizer.lemmatize('running', 'v'))
