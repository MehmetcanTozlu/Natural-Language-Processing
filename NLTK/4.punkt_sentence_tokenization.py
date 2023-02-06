import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer


class PunktSentToken(object):
# =============================================================================
#     Konusma Bolumu etiketlemesi, bir cumledeki kelimeler, isimler, sifatlar,
#     fiiller vb. olarak etiketlemek anlamina gelir. Zamana gorede etiketler.
#     Punkt Sentence Tokenizer = cumle belirteci
#     Bu belirtec denetimsiz ML yetenegine sahiptir.
# =============================================================================
    def __init__(self):
        self.tokenized = None

    def punktSentToken(self, train_text, test_text):
        # Punkt Tokenizer'i egitelim
        custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
        
        self.tokenized = custom_sent_tokenizer.tokenize(test_text)
        
    def process_content(self):
        print('Punkt Sentences Tokenization:')
        try:
            for i in self.tokenized[:5]:
                words = word_tokenize(i)
                tagged = nltk.pos_tag(words)
                print(tagged)
        
        except Exception as e:
            print(str(e))


train_text = state_union.raw('2005-GWBush.txt')
test_text = state_union.raw('2006-GWBush.txt')

punkt_sent_tok = PunktSentToken()
punkt_sent_tok.punktSentToken(train_text, test_text)
punkt_sent_tok.process_content()