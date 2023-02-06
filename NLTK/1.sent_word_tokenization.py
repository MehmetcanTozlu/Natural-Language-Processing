from nltk.tokenize import word_tokenize, sent_tokenize

text = """Recent research has increasingly focused on unsupervised and semi-supervised learning algorithms. Such algorithms can learn from data that has not been hand-annotated with the desired answers or using a combination of annotated and non-annotated data. Generally, this task is much more difficult than supervised learning, and typically produces less accurate results for a given amount of input data. However, there is an enormous amount of non-annotated data available(including, among other things, the entire content of the World Wide Web), which can often make up for the inferior results if the algorithm used has a low enough time complexity to be practical."""


class SentWordTokenization(object):
    
    def __init__(self):
        self.sent_token = None
        self.word_token = None
    
    def sentTokenization(self, text):
        self.sent_token = sent_tokenize(text)
        print('\nSent Tokenization:\n', self.sent_token)
    
    def workTokenization(self, text):
        self.word_token = word_tokenize(text)
        print('\n\nWork Tokenization:\n', self.word_token)


sent_word_tok = SentWordTokenization()
sent_word_tok.sentTokenization(text)
sent_word_tok.workTokenization(text)
