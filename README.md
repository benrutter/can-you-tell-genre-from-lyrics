# Can you guess Genre from Lyrics?

*(tldr: kinda)*

If you're into music and data, then the [Million Song Dataset](http://millionsongdataset.com/) is a pretty awesome resource. There's a bunch of meta data there on a total of (unsurprisingly) a million songs which works out at about 280GB if you decide to download the full dataset. There's also a bunch of linked datasets, like the [musiXmatch dataset](http://millionsongdataset.com/musixmatch/) and [tagtraum genre annotations](https://www.tagtraum.com/msd_genre_datasets.html) so I figures it would be a cool use of this to try and see if it was possible to build a model to predict the genre of music based off of a set of lyrics.

There's hundreds of genre, include a bunch without lyrics like 'House' or 'Instrumental Jazz' so I decided just to look at the top four:
* Rock
* Pop
* Hip-Hop
* Country

(the weirdest finding out of this is that Country is the fourth most common genre, which I don't understand at all and think is maybe just down to people not splitting it out into a bunch of subgenres like 'alt-house-dub-country' or 'extreme-death-country-core')

Pos-tagging the entire dataset took a reeeeaaallly long time, but after I'd done that I was able to take a look at the most used nouns and verbs in each genre. The code is simple enough, since I just used used NLTK's built in pos_tag function, but this uses peceptrons, so running it over a really big dataset, especially just using pandas apply function like I did, took about 40 minutes or something.

```python
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
```

In terms of word usage, the most used noun is 'I' which kinda figures, but it was also cool to see that 'love' is the second most used, especially in country where more that 1 in 500 nouns are the word 'love' (I guess the whole 'every song is a love song' thing is true).

![Bar chart showing use of the word love by genre](https://github.com/benrutter/million-song-dataset-exploration/blob/master/images/love_comparison.png)

Other than that, the main thing that stood out was the fact that Hip-Hop seems really different from the other three genres - words like 'shit' and 'yo' are some of the most used nouns (is 'yo' a noun?) for Hip-Hop but are pretty small in the other genres. On the flip side, a lot of the most used words in Country, Rock and Pop, like 'day', 'time' and 'love', are a lot less frequent in Hop-Hop.

![Bar chart showing a selection of the most used nouns by genre](https://github.com/benrutter/million-song-dataset-exploration/blob/master/images/noun_comparison.png)

I saw the same kinda thing looking at verbs, but to a lesser extent. Hop-Hop still looks like the outlier, with the other three genres more or less matching exactly in usage of the most frequent verbs.

![Bar chart showing a selection of the most used verbs by genre](https://github.com/benrutter/million-song-dataset-exploration/blob/master/images/verb_comparison.png)

I also thought it would be interesting to see which genre uses the 'rarest' nouns (i.e. of all words used, what's the distribution of usage). So I started evaluating counts of words as a total of their word group.

```python
select_nouns['word_proportion'] = select_nouns.apply(
    lambda row: row['count'] / noun_count[row['genre']], axis=1)
```

Rock looked like it had a significantly narrower pool of words, while Country is a lot broader than the other three- it's really interesting to think of the reason for that although I have no idea, maybe it's something to do with place names which feature a lot in country music (and a lot less in other genres), or maybe there's a a really big subgenre of country music singing songs about Cuttlefish and Aglets and stuff.

```python
kde = sns.kdeplot(
    x='count',
    data=grouped_nouns,
    log_scale=10,
    hue='genre',
)
```

![KDE plot of the distribution of noun usage](https://github.com/benrutter/million-song-dataset-exploration/blob/master/images/word_rareness.png)

After taking a look through the dataset, I trained up a Naive Bayes and a Support Vector machine model. After training a Support Vector machine on the entire dataset (which takes a really long time) I got really excited by seeing a 75% accuracy rate, then immediately disappointed when I realised the reason for this was just that it was predicting everything was Rock (the most common genre)

![Confusion matrix of the SVM model](https://github.com/benrutter/million-song-dataset-exploration/blob/master/images/svm-confusion-matrix.png)

The Naive Bayes model performed ok-ish / surprisingly well and made the correct classification about 60% of the time. That's not a lot but it's better than I expected given that, apart from Hip-Hop, the genre word usage didn't show a lot of difference from a high level comparison. It also did well precision wise rather than just placing everything into the most common category.

```python
from sklearn.naive_bayes import MultinomialNB
nb_model = MultinomialNB()
nb_model.fit(train_features, train_labels)
```

The confusion matrix is really cool for this as well- as the cross over points map a lot with what you might expect. Pop, which is a pretty vague genre, gets lumped in with Rock (for all the Coldplay fans) and Country (for all the Swifties) a lot. Rock pretty much only gets confused with Country. And strangely, Country is predicted really accurately.

![Confusion matrix of the NB model](https://github.com/benrutter/million-song-dataset-exploration/blob/master/images/nb-confusino-matrix.png)

Feel free to check out any of the code which is mainly split out into a couple of Jupyter notebooks, or if you feel like using the POS tagged dataset, without waiting 45 minutes for NLTK to run a positron network on hundreds of thousands of rows, [be my guest](https://ufile.io/2z1tfu5r)!
