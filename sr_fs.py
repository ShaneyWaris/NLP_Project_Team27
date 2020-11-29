import numpy as np
import re, nltk
# nltk.download('vader_lexicon')
# nltk.download('punkt')


# Sriza Features

"""##No of URLS"""

base = ''
def no_of_URLs(listoftweets):
    count = []
    for tweet in listoftweets:
        l = re.findall(r'((www\.[\S]+)|(https?:\/\/[\S]+))', tweet)
        # print(l)
        count.append(len(l))
    return np.array(count);


"""##No of Hashtags"""


def no_of_hashtags(listoftweets):
    count = []
    for tweet in listoftweets:
        l = re.findall(r'#(\w+)', tweet)
        # print(l)
        count.append(len(l))
    return np.array(count);


"""##Lexicon sentiment of hastags"""


def aggregatepolarityscores_hashtags(listoftweets):
    with open(base + 'Lex/unigrams-pmilexiconNRC_HashtagsSentiment.txt', 'r') as document:
        hashtagscore = {}
        for line in document:
            line = line.split()
            if (line[0][0] == '@'):
                continue
            hashtagscore[line[0]] = float(line[1:][0])

    vector = []
    for tweet in listoftweets:
        tweet = tweet.split(' ')
        # print(tweet)

        val1 = 0
        val = 0
        for word in tweet:
            if len(word) > 1 and word[0] != '#':
                continue
            if word in hashtagscore.keys():
                val = hashtagscore[word]
            val1 += val

        vector.append(val1)
    return np.array(vector)


"""##No of Emojis"""


def no_of_emojis(listoftweets):
    with open(base + 'Lex/AffinnEmoticons.txt', 'r', encoding='UTF-8') as document:
        emoticons_score1 = {}
        # print(document)
        for line in document:
            # print(line)
            words = line.split()
            emoticons_score1[words[0]] = int(words[1])
    emocount = []
    for tweet in listoftweets:
        emo = 0
        t = tweet.split(' ')

        for word in t:
            if word in emoticons_score1.keys():
                emo += 1
                # print(word)
        emocount.append(emo)
    return np.array(emocount)


"""##Emoji Sentiment Average"""


def emoticons_score(listoftweets):
    with open(base + 'Lex/AffinnEmoticons.txt', 'r', encoding='UTF-8') as document:
        emoticons_score1 = {}
        # print(document)
        for line in document:
            # print(line)
            words = line.split()
            emoticons_score1[words[0]] = int(words[1])
    vector = []
    for tweet in listoftweets:
        tweet = tweet.split(' ')
        emoscore = 0
        for word in tweet:
            if word in emoticons_score1.keys():
                emoscore += emoticons_score1[word]
        # print(emoscore)
        # feature_vector[i][17]+=emoscore
        vector.append(emoscore)
    return np.array(vector)


def getfeaturearray(listoftweets):
    v1 = no_of_URLs(listoftweets)
    v2 = no_of_hashtags(listoftweets)
    v3 = aggregatepolarityscores_hashtags(listoftweets)
    v4 = no_of_emojis(listoftweets)
    v5 = emoticons_score(listoftweets)

    return np.concatenate(
        (v1.reshape(-1, 1), v2.reshape(-1, 1), v3.reshape(-1, 1), v4.reshape(-1, 1), v5.reshape(-1, 1)),
        axis=1).reshape((1, -1))