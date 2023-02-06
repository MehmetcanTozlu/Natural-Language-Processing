import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer


train_text = state_union.raw('2005-GWBush.txt')
test_text = state_union.raw('2006-GWBush.txt')


class Chunking(object):
    
    def __init__(self):
        pass
    
    def process_content(self, train_text, test_text):
        custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
        
        tokenized = custom_sent_tokenizer.tokenize(test_text)
        
        try:
            for i in tokenized:
                words = word_tokenize(i)
                tagged = nltk.pos_tag(words)
                chunkGram = r"""Chunk: {<RB.?>*<NNP>+<NN>?}"""
                chunkParser = nltk.RegexpParser(chunkGram)
                chunked = chunkParser.parse(tagged)
                print('Chunking: ', chunked)
                
                for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):
                    print(subtree)
                
                chunked.draw()
        
        except Exception as e:
            print(str(e))


chunking = Chunking()
chunking.process_content(train_text, test_text)