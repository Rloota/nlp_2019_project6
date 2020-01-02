import nltk
import numpy
from nltk.corpus import wordnet as wn
'''
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')
'''

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



print("Same sentence: %s."%(similarity(s1,s1)))
print("s1 similarity to s2: %s. As per STSS-131 should be 0.77"%(similarity(s1,s2)))
print("s3 similarity to s4: %s. As per STSS-131 should be 0.69"%(similarity(s3,s4)))