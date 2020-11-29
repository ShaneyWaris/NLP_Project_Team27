import numpy as np
import re, nltk
from nltk import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
nltk.download('punkt')

base = ''
def mark_negation(sentence):
    negation = r"""(?:^(?:never|no|nothing|nowhere|noone|none|not|havent|hasnt|hadnt|cant|couldnt|shouldnt|wont|wouldnt
        |dont|doesnt|didnt|isnt|arent|aint)$)|n't """
    neg_re = re.compile(negation, re.VERBOSE)
    punctuation = r"^[.:;!?]$"
    puncts = re.compile(punctuation)

    doc = word_tokenize(sentence)
    neg_scope = False
    count = 0
    for i, word in enumerate(doc):
        if neg_re.search(word):
            if not neg_scope:
                neg_scope = not neg_scope
                continue
            else:
                doc[i] += "_NEG"
                count += 1
        elif neg_scope and puncts.search(word):
            neg_scope = not neg_scope
        elif neg_scope and not puncts.search(word):
            doc[i] += "_NEG"
            count += 1
    return count


# Bing lui lexicons - no. of positive & negative words in a tweet.
def bing_lui(tweet):
    positive_words = 0
    negative_words = 0
    positive_minus_negative = 0
    with open(base + 'Lex/bing lui lexicon/positive-words.txt') as file:
        contents = file.read()
        for word in tweet.split():
            if word in contents:
                positive_words += 1
    with open(base + 'Lex/bing lui lexicon/negative-words.txt') as file:
        contents = file.read()
        for word in tweet.split():
            if word in contents:
                negative_words += 1

    positive_minus_negative = positive_words - negative_words
    lexicon_vector = [positive_words, negative_words, positive_minus_negative]
    return lexicon_vector


def polarity(sentence):
    sid = SentimentIntensityAnalyzer()
    pol = sid.polarity_scores(sentence)
    return [pol['pos'], pol['neg'], pol['neu']]


def get_shaney_features(sentence):
    negation_words = mark_negation(sentence)
    bingLui = bing_lui(sentence)
    polari = polarity(sentence)
    return np.array([negation_words] + bingLui + polari).reshape((1, -1))