# =============================================================================
# Wordnet, kelimelerin, tanimlarin, kullanim orneklerinin, esanlamlilarin,
# zit anlamlilarin ve daha fazlasinin bir koleksiyonudur.
# Princeton tarafindan olusturulan ve nltk kulliyatinin bir parcasi olan
# Ingilizce dili icin sozcuksel bir veritabanidir.
# =============================================================================

from nltk.corpus import wordnet

# program kelimesinin sentez kumeleri
syns = wordnet.synsets('program')
print(syns[0].name())

# Sadece kelime
print(syns[0].lemmas()[0].name())

# ilk synset'in tanimi
print(syns[0].definition())

# kullanilan kelime ornekleri
print(syns[0].examples())

# Bir kelimenin es anlamlisini ve zit anlamlisini nasil ayirt edebiliriz?
synonyms = [] # es anlamlilari icin
antonyms = [] # zit anlamlilari icin

for syn in wordnet.synsets('bad'):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print('\n', set(antonyms))

