import pandas as pd
from nltk import pos_tag

genre_and_lyrics = pd.read_csv('genre_and_lyrics.csv')

def pos_tag_word(word):
    '''
    try to pos tag a single word
    if word is nan then return nan
    nan check done by seeing if word is equal to self
    '''
    if word != word:
        return word
    else:
        return pos_tag([word])[0][1]

genre_and_lyrics['pos'] = genre_and_lyrics['word'].apply(pos_tag_word)

genre_and_lyrics.to_csv('genre_and_lyrics.csv')
