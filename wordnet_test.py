import numpy
import csv
import nltk
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.text import TextCollection
import string

from nltk.corpus import reuters

from dataclasses import dataclass
from typing import List
from copy import copy
from scipy import stats

import argparse

from utils import readCSV
from hierarcial_reasoning import wordnetPosTag

from nltk.corpus import reuters

from sklearn.feature_extraction.text import TfidfVectorizer



STSS_131_DATA = "data/STSS-131.csv"

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("maxent_ne_chunker")
nltk.download("words")
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download('reuters')

#Using fancy sklearn lib machine learning to calculate tf_idf using reuters as corpus
tf_idf_model = TfidfVectorizer()
#Fit tf_idf model using reuters corpus
tf_idf_model.fit([reuters.raw(fileids) for fileids in reuters.fileids()])
tf_idf = tf_idf_model.transform([reuters.raw()])

"""
# Synset

Synset is a special kind of a simple interface that is present in NLTK to look up words in WordNet.
Synset instances are the groupings of synonymous words that express the same concept. 
Some of the words have only one Synset and some have several. 

Hypernyms: More abstract terms
Hyponyms: More specific terms. 

Source: https://www.geeksforgeeks.org/nlp-synsets-for-a-word-in-wordnet/

"""

def synset_tag(word, tag):
    wn_tag = wordnetPosTag(tag)
    if wn_tag is None:
        return None
    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None


def wordNetSimilarity(s1, s2, perform_lemmatization = False, perform_stemming = False, use_wup = False, use_lch = False, use_idf = False):
    """ 
    An attempt to measure similarity of sentences using Wordnet for single sentence pair. 
    
    Heavily based on the method presented by Mihalcea et al. https://www.aaai.org/Papers/AAAI/2006/AAAI06-123.pdf

    :param s1: First sentence
    :param s2: Second sentence
    :rtype: Similarity score of sentences
    :param perform_stemming: Set to True to perform stemmings, default False
    :param use_wup: Set to True to use wup_similarity to measure similarity, default False.
    :param use_lch: Set to True to use lch_similarity to measure similarity, default False.
    :param use_idf: Set to True to use idf in calculation.
    """
    
    #Mandatory preprosessing: remove punctuation and tokenize sentences.

    # Sentences are broken into words, symbols and other potential meaningful elements
    s1_no_punct = "".join([p.lower() for p in s1 if p not in string.punctuation])
    s2_no_punct = "".join([p.lower() for p in s2 if p not in string.punctuation])

    s1_tokens = nltk.word_tokenize(s1_no_punct)
    s2_tokens = nltk.word_tokenize(s2_no_punct)

    stop_words = set(stopwords.words('english'))
    filtered_words1 = [w for w in s1_tokens if not w in stop_words]
    filtered_words2 = [w for w in s2_tokens if not w in stop_words]
    s1_tokens = filtered_words1
    s2_tokens = filtered_words2

    # Optional Pre-processing

    # Option to perform stemming
    # Stemmers remove morphological affixes from words, leaving only the word stem.

    if perform_stemming == True:
        stemmer = SnowballStemmer("english", ignore_stopwords=True)
        s1_stemmed = [stemmer.stem(w) for w in s1_tokens]
        s2_stemmed = [stemmer.stem(w) for w in s2_tokens]
        s1_tokens = s1_stemmed
        s2_tokens = s2_stemmed

    # Option to perform lemmatization
    # Lemmatization is the process of grouping together the different inflected forms of a word so they can be analysed as a single item.
    if perform_lemmatization == True:
        lemmatizer=WordNetLemmatizer()
        s1_lemmatized = [lemmatizer.lemmatize(w) for w in s1_tokens]
        s2_lemmatized = [lemmatizer.lemmatize(w) for w in s2_tokens]
        s1_tokens = s1_lemmatized
        s2_tokens = s2_lemmatized

    # Step 2, pos tag
    s1_tokens = pos_tag(s1_tokens)
    s2_tokens = pos_tag(s2_tokens)

    # Get synsets of each word (looping list of tokens)
    s1_synsets = [synset_tag(*tagged_word) for tagged_word in s1_tokens]
    s2_synsets = [synset_tag(*tagged_word) for tagged_word in s2_tokens]
    #Filter out any possible None values.
    s1_synsets = [i for i in s1_synsets if i]
    s2_synsets = [i for i in s2_synsets if i]

    final_score = []

    #Calculating Inverse-Document-Frequency
    # Use TF-IDF to determine frequency of each word in our article, relative to the
    # word frequency distributions in corpus of 11k Reuters news articles.
    #tfidf = TfidfVectorizer(tokenizer=self.tokenize_and_stem, stop_words='english', decode_error='ignore')
    #tdm = self._tfidf.fit_transform(token_dict.values())  # Term-document matrix 

    #mytexts = TextCollection([])
    #Algorithm not yet Inverse-Document-Frequency weighted


    #Using fancy sklearn lib machine learning to calculate tf_idf using reuters as corpus
    #stop_words = stopwords.words('english') + list(punctuation)
    s1_idfs, s2_idfs = [], []
    for word in s1_tokens:
        try:
            s1_idfs.append(tf_idf[0, tf_idf_model.vocabulary_[word[0]]])
        except:
            #print("KeyError %s"%(word[0]))
            pass

    for word in s2_tokens:
        try:
            s2_idfs.append(tf_idf[0, tf_idf_model.vocabulary_[word[0]]])
        except:
            #print("KeyError %s"%(word[0]))
            pass

    for i in range(2):
        score, count, similarity_values = 0.0, 0, []
        for w1synset in s1_synsets:
            for w2synset in s2_synsets:
                #Possibility to use Wu-Palmer Similarity.
                if use_wup == True:
                    path_sim = w1synset.wup_similarity(w2synset)
                #Possibility to use Leacock-Chodorow Similarity.
                if use_lch == True:
                    path_sim = w1synset.lch_similarity(w2synset)
                else:
                    path_sim = w1synset.path_similarity(w2synset)  
                if path_sim != None:
                    # Make larger scale values for matching range in STSS-131 data set
                    similarity_values.append(path_sim * 4)
            try:
                score += max(similarity_values)
                count += 1
            except:
                return 0

        score /= count
        if use_idf == True:
            final_score.append((score * sum(s1_idfs)) / sum(s2_idfs))
        else:
            final_score.append(score)
        s1_synsets, s2_synsets = s2_synsets, s1_synsets

    return (final_score[0] + final_score[1])/2

def STSS_tests():

    '''Some tests for wordNetSimilarity using STSS dataset'''
    import matplotlib.pyplot as plt
    sentences = readCSV(STSS_131_DATA)
    sim_values, STSS_values = [], []
    n = 0
    
    for s in sentences:

        sim_values.append(wordNetSimilarity(s.first_sentence, s.second_sentence))
        STSS_values.append(s.human_SS)

        '''
        if n < 10:
            sim_values.append(wordNetSimilarity(s.first_sentence, s.second_sentence))
            STSS_values.append(s.human_SS)
            n = n+1
            #print('Sentence "%s" similarity to "%s", score: %s, STSS-131 value: %s' \
            #%(s.first_sentence, s.second_sentence, \
            #wordNetSimilarity(s.first_sentence, s.second_sentence), \
            #s.human_SS))
        '''
    print("******************************************************")
    print("Using path_similarity")
    print(stats.pearsonr(sim_values,STSS_values))
    print(stats.pearsonr(STSS_values, STSS_values))

    '''
    plt.plot(sim_values)
    plt.plot(STSS_values)
    plt.show()
    '''

    sim_values, STSS_values = [], []
    n = 0
    for s in sentences:
        sim_values.append(wordNetSimilarity(s.first_sentence, s.second_sentence, use_wup = True))
        STSS_values.append(s.human_SS)
   
    print("******************************************************")
    print("Using wup")
    p = stats.pearsonr(sim_values,STSS_values)
    print(p)

    sim_values, STSS_values = [], []
    n = 0
    for s in sentences:
        sim_values.append(wordNetSimilarity(s.first_sentence, s.second_sentence, perform_stemming = True))
        STSS_values.append(s.human_SS)

    print("******************************************************")
    print("Preprocessing: Stemming")
    p = stats.pearsonr(sim_values,STSS_values)
    print(p)  


    sim_values, STSS_values = [], []
    for s in sentences:
        sim_values.append(wordNetSimilarity(s.first_sentence, s.second_sentence, perform_lemmatization = True))
        STSS_values.append(s.human_SS)

    print("******************************************************")
    print("Preprocessing: lemmatization")
    p = stats.pearsonr(sim_values,STSS_values)
    print(p)


    sim_values, STSS_values = [], []
    n = 0
    for s in sentences:
        sim_values.append(wordNetSimilarity(s.first_sentence, s.second_sentence, perform_stemming = True, use_idf=True))
        STSS_values.append(s.human_SS)

    print("******************************************************")
    print("Preprocessing: Stemming, use tf_idf")
    p = stats.pearsonr(sim_values,STSS_values)
    print(p)      
    
if __name__ == "__main__":
    STSS_tests()