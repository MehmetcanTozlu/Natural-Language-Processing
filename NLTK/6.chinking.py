import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw('2005-GWBush.txt')
test_text = state_union.raw('2006-GWBush.txt')


class Chinking(object):
    
    def __init__(self):
        pass
    
    def process_content(self, train_text, test_text):
        custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

        tokenized = custom_sent_tokenizer.tokenize(test_text)
        
        try:
            for i in tokenized[5:]:
                words = word_tokenize(i)
                tagged = nltk.pos_tag(words)
                
                chunkGram = r"""Chunk: {<.*>+}
                                            }<VB.?|IN|DT|TO>+{"""
                
                chunkParser = nltk.RegexpParser(chunkGram)
                chunked = chunkParser.parse(tagged)
                
                chunked.draw()
        
        except Exception as e:
            print(str(e))


chinking = Chinking()
chinking.process_content(train_text, test_text)