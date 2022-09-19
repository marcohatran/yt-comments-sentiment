from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re


def cleanTxt(text):
    text = re.sub(r'[^\w]', ' ', text)

    return text


# get subjectivity
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity


# get polarity
def getPolarity(text):
    return TextBlob(text).sentiment.polarity


# function to compute analysis
def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'


def sentiments(data):
    data['Comment'] = data['Comment'].apply(cleanTxt)
    data['Subjectivity'] = data['Comment'].apply(getSubjectivity)
    data['Polarity'] = data['Comment'].apply(getPolarity)
    data['Analysis'] = data['Polarity'].apply(getAnalysis)
    pcomments = data[data.Analysis == 'Positive']
    pcomments = pcomments['Comment']
    positive = str(round((pcomments.shape[0] / data.shape[0]) * 100, 1)) + '%'
    ncomments = data[data.Analysis == 'Negative']
    ncomments = ncomments['Comment']
    negative = str(round((ncomments.shape[0] / data.shape[0]) * 100, 1)) + '%'
    nucomments = data[data.Analysis == 'Neutral']
    nucomments = nucomments['Comment']
    neutral = str(round((nucomments.shape[0] / data.shape[0]) * 100, 1)) + '%'
    return data, positive, negative, neutral
