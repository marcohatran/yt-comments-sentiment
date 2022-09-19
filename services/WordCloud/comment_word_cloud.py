from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import nltk


def word_cloud(document):
    """ function to get the word cloud """
    stop_words = stopwords.words("english")
    tokenizer = RegexpTokenizer(r"\w+")
    word_tokens = list(tokenizer.tokenize(document))
    final_tokens = [w for w in word_tokens if not w in stop_words]
    most_frequent = Counter(final_tokens).most_common(40)
    data = []
    for (word, frequency) in most_frequent:
        data.append({"word": word, "frequency": frequency})
    return data
