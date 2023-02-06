# =============================================================================
# NLP de en onemli parcalama bicimleriden biri Adlandirilmis Varlik Tanima
# olarak adlandirilir. Burada fikir, makinenin insanlar, yerler, parasal rakamlar
# ve daha fazlasi gibi varliklari hemen cikarabilmesidir.
# NLTK nin Adlandirilmis Varlik Tanimasinda iki ana secenek vardir.
# 1. Ya tum adlandirilmis varliklari taniyin.
# 2. Adlandirilmis varliklari insanlar, yerler, konumlar vb. ilgili tur olarak taniyin. 
# =============================================================================

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw('2005-GWBush.txt')
test_text = state_union.raw('2006-GWBush.txt')

class NamedEntityRecognition(object):
    
    def __init__(self):
        pass
    
    def ner(self, train_text, test_text):
        custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
        
        tokenized = custom_sent_tokenizer.tokenize(test_text)
        
        try:
            for i in tokenized[5:]:
                words = word_tokenize(i)
                tagged = nltk.pos_tag(words)
                named_entity = nltk.ne_chunk(tagged, binary=True)
                named_entity.draw()
        
        except Exception as e:
            print(str(e))

nltk.download('words')
ner = NamedEntityRecognition()
ner.ner(train_text, test_text)
