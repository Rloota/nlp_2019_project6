import nltk
import numpy
import csv
from nltk.corpus import wordnet as wn


from dataclasses import dataclass
from typing import List
from copy import copy


@dataclass
class SentencePair:
    SP_id: int
    first_sentence: str
    second_sentence: str
    human_SS: float
    standard_deviation: float


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')


STSS_131_DATA = "data/STSS-131.csv"

"""
# Synset

Synset is a special kind of a simple interface that is present in NLTK to look up words in WordNet.
Synset instances are the groupings of synonymous words that express the same concept. 
Some of the words have only one Synset and some have several. 

Hypernyms: More abstract terms
Hyponyms: More specific terms. 

Source: https://www.geeksforgeeks.org/nlp-synsets-for-a-word-in-wordnet/

"""

def readCSV(filename) -> List[SentencePair]:
    """
    Read sample data from CSV file and generate dataobject for each sentence pair.

    :return: List of SentencePairs
    """
    sentences = []
    with open(filename, newline='') as csvfile:
        sample_data = csv.reader(csvfile, delimiter=';', quotechar='"')
        for i, row in enumerate(sample_data):
            # Skip the first row
            if i == 0:
                continue
            # Check that values are in correct range:
            # According to source:
            # Semanticsimilarity ratings for STSS-131 (on a scale from 0.00 to 4.00)
            try:
                assert float(row[3]) >= 0 and float(row[3]) <= 4
                assert float(row[4]) >= 0 and float(row[4]) <= 4
                sentence_pair = SentencePair(int(row[0]), row[1], row[2], float(row[3]), float(row[4]))
            except ValueError as e:
                print(e)
                print(f"Values were: {row[0]} {row[1]} {row[2]} {row[3]} {row[4]}")
                exit(1)

            sentences.append(copy(sentence_pair))
    return sentences



def similarity(s1, s2):
    """ An attempt to measure similarity of sentences using Wordnet. """

    #step 1 tokenize sentences
    tokens1 = nltk.word_tokenize(s1)
    tokens2 = nltk.word_tokenize(s2)

    #print(tokens1)
    #print(tokens2)

    #step 2 tag words

    tag1 = nltk.pos_tag(tokens1)
    tag2 = nltk.pos_tag(tokens2)

    #print(tag1)
    #print(tag2)
    w1=[]
    w2=[]

    for word in tag1:
        w1 += wn.synsets(word[0])
    for word in tag2:
        w2 += wn.synsets(word[0])

    score, count = 0.0, 0
    n_score = []
    for w1synset in w1:
        for w2synset in w2:
            path_sim = w1synset.path_similarity(w2synset)
            if path_sim != None:
                n_score.append(path_sim)

        score += max(n_score)
        count += 1

    score /= count
    return score


s1 = "Would you like to go out to drink with me tonight?"
s2 = "I really don't know what to eat tonight so I might go out somewhere."

s3 = "I advise you to treat this matter very seriously as it is vital."
s4 = "You must take this most seriously, it will affect you."

sentences = readCSV(STSS_131_DATA)
for s in sentences:
    print(s.SP_id)

# print("Same sentence: %s."%(similarity(s1,s1)))
# print("s1 similarity to s2: %s. As per STSS-131 should be 0.77"%(similarity(s1,s2)))
# print("s3 similarity to s4: %s. As per STSS-131 should be 0.69"%(similarity(s3,s4)))