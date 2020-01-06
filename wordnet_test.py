import nltk
import numpy
import csv
from nltk.corpus import wordnet as wn


from dataclasses import dataclass
from typing import List
from copy import copy


@dataclass
class SentencePair:
    """
    Data object for sentence pair, particulary crafter for dataset
    STSS-131 which is presented in here https://semanticsimilarity.files.wordpress.com/2013/11/trmmucca20131_12.pdf
    """

    SP_id: int
    first_sentence: str
    second_sentence: str
    human_SS: float
    standard_deviation: float


# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('wordnet')


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

    :rtype: List of SentencePairs
    """
    sentences = []
    with open(filename, newline="") as csvfile:
        sample_data = csv.reader(csvfile, delimiter=";", quotechar='"')
        for i, row in enumerate(sample_data):
            # Skip the first row
            if i == 0:
                continue
            """
            Check that values are in correct range:
            According to source:
            Semanticsimilarity ratings for STSS-131 (on a scale from 0.00 to 4.00)
            """
            try:
                assert float(row[3]) >= 0 and float(row[3]) <= 4
                assert float(row[4]) >= 0 and float(row[4]) <= 4
                assert len(row) == 5
                sentence_pair = SentencePair(
                    int(row[0]), row[1], row[2], float(row[3]), float(row[4])
                )
            except ValueError as e:
                # "Dataset had invalid format"
                print(e)
                print(f"Values were: {row[0]} {row[1]} {row[2]} {row[3]} {row[4]}\n")
                print("Invalid format.")
                exit(1)

            sentences.append(copy(sentence_pair))
    return sentences


def wordNetSimilarity(s1, s2):
    """ 
    An attempt to measure similarity of sentences using Wordnet for single sentence pair. 
    
    Heavily based on the method presented by Mihalcea et al. https://www.aaai.org/Papers/AAAI/2006/AAAI06-123.pdf

    :param s1: First sentence
    :param s2: Second sentence
    :rtype: Similarity score of sentences
    """

    # Step 1, preprosessing: tokenize sentences
    # Sentences are broken into words, symbols and other potential meaningful elements
    tokens1 = nltk.word_tokenize(s1)
    tokens2 = nltk.word_tokenize(s2)
    print(tokens1)

    # Step 2 tag words. NLTK method 'pos_tag' is pretrained.
    # It was trained with Treebank corpus, and supports Treebank tags

    tag1 = nltk.pos_tag(tokens1)
    tag2 = nltk.pos_tag(tokens2)

    print(tag1)
    print(tag2)
    s1_synsets = []
    s2_synsets = []

    # Get synsets of each word (looping list of tokens)
    for word in tokens1:
        s1_synsets += wn.synsets(word)
    for word in tokens2:
        s2_synsets += wn.synsets(word)

    score, count = 0.0, 0
    n_score = []
    for w1synset in s1_synsets:
        for w2synset in s2_synsets:
            path_sim = w1synset.path_similarity(w2synset)
            if path_sim != None:
                n_score.append(path_sim)

        score += max(n_score)
        count += 1

    score /= count
    return score



# print("s3 similarity to s4: %s. As per STSS-131 should be 0.69"%(similarity(s3,s4)))


if __name__ == "__main__":
    s1 = "Would you like to go out to drink with me tonight?"
    s2 = "I really don't know what to eat tonight so I might go out somewhere."

    s3 = "I advise you to treat this matter very seriously as it is vital."
    s4 = "You must take this most seriously, it will affect you."

    sentences = readCSV(STSS_131_DATA)
    for s in sentences:
        print(s.SP_id)

    # print("Same sentence: %s."%(similarity(s1,s1)))
    print(
        "s1 similarity to s2: %s. As per STSS-131 should be 0.77"
        % (wordNetSimilarity(s1, s2))
    )
