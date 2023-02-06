from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer

class Stemming(object):
    
    def __init__(self):
        pass
    
    def stemEx(self):
        ps = PorterStemmer()
        
        example_words = ["python","pythoner","pythoning","pythoned","pythonly"]
        
        for i in example_words:
            print(ps.stem(i))
    
    def stemEx2(self):
        ps = PorterStemmer()
        
        example_words = "It is important to by very pythonly while you are pythoning with python. All pythoners have pythoned poorly at least once."
        
        words = word_tokenize(example_words)
        
        for i in words:
            print(ps.stem(i))
        

stem = Stemming()
stem.stemEx()
print('***********')
stem.stemEx2()