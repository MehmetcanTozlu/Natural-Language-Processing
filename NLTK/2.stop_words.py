from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

text = """Recent research has increasingly focused on unsupervised and semi-supervised learning algorithms. Such algorithms can learn from data that has not been hand-annotated with the desired answers or using a combination of annotated and non-annotated data. Generally, this task is much more difficult than supervised learning, and typically produces less accurate results for a given amount of input data. However, there is an enormous amount of non-annotated data available(including, among other things, the entire content of the World Wide Web), which can often make up for the inferior results if the algorithm used has a low enough time complexity to be practical."""



class StopWords(object):
    
    def __init__(self):
        self.stopwords = None
        self.word_token = None
        self.filtered_sentences = []
    
    def stop_words(self, text):
        self.stop_words = set(stopwords.words('english'))
        self.word_token = word_tokenize(text)
        
        self.filtered_sentences = [token for token in self.word_token if not token in self.stop_words]
        
        print('\nWord Tokenization:\n', self.word_token)
        print('Word Token Length: ', len(self.word_token))
        
        print('\n\nFiltered Sentences:\n', self.filtered_sentences)
        print('Filtered Sentences Length: ', len(self.filtered_sentences))
    

stop_wrd = StopWords()
stop_wrd.stop_words(text)