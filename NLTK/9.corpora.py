# =============================================================================
# NLTK toplulugundaki dosyalarin tumune nltk modulunu kullanarak erisebiliriz.
# Bu dosyalar cogunlukla duz metin dosyalaridir, bazilari XML ve diger bicimlerdir.
# Ancak hepsine kendi tarafimizdan manuel olarak erisebiliriz.
# =============================================================================

# nltk_data dizinimizin bilgisayarimizda nerede saklandigina manuel olarak bakalim
import nltk
print(nltk.__file__)

# =============================================================================
# Windows kullaniyorsak;
# C:\Users\yourpcname\AppData\Roaming\nltk_data\corpora
# nltk_data icin cesitli dizinlerimiz bu klasorde saklaniyor olmali.
# Bu klasorde, kitaplar, film incelemeleri ve cok daha fazlasi mevcuttur.
# =============================================================================

# Simdi bu belgelere nltk uzerinden erismeye calisalim.
# Bu belgeler normal metin verileri old. normal Python kodu kullanabiliriz.
# Bununla birlikte nltk modulunun birkac guzel yontemi de var.
from nltk.tokenize import sent_tokenize
from nltk.corpus import gutenberg

nltk.download('gutenberg')
# sample text
sample = gutenberg.raw('bible-kjv.txt')

token = sent_tokenize(sample)

for i in range(5):
    print(token[i])
