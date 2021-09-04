import pandas as pd 
import numpy as np
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string
import re
import stopwordsiso as sw

# create stemmer for bahasa
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# stop words consist of malay, indo, english
stop_words_main = list(sw.stopwords(["ms", "id", "en"]))
# custom stopwords such as shortform
stop_words_custom = ['kau', 'yg', 'mcm', 'gak', 'nak', 'ni', 'tu', 'la', 'je', 'kat', 'ya', 'dgn', 'tau', 'org', 'rt', 'aja', 
                    'nk', 'dah', 'orang', 'sy', 'ga', 'kalo', 'kena']
STOP_WORDS = np.unique(stop_words_main+stop_words_custom)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F" # emoticons
                           u"\U0001F300-\U0001F5FF" # symbols & pictographs
                           u"\U0001F680-\U0001F6FF" # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF" # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def text_preprocessing(text):

    # remove numbers
    text = re.sub(r'\d+', '', text)

    # remove links
    text = re.sub('http[s]?://\S+', '', text)

    # remove word with tweethandle @name
    text = re.sub('[^ ]*@[^ ]*', '', text)

    # remove emoji
    text = remove_emoji(text)

    # tokennization
    tokens = word_tokenize(text)

    # stemmer and remove punctuation
    words = []
    for token in tokens:
        if token not in string.punctuation:
            temp = stemmer.stem(token)
            words.append(temp)

    # remove stopwords
    cleaned = []
    for word in words:
        if word not in STOP_WORDS:
            cleaned.append(word)

    # join all words into a complete sentence 
    complete_sentence = ' '.join([str(word) for word in cleaned])

    # remove extra line spaces between words in a sentence
    complete_sentence = " ".join(complete_sentence.split())
    
    return complete_sentence