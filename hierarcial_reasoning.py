import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus.reader.wordnet import Synset
from nltk import pos_tag
import logging
from utils import readCSV
from typing import List
from copy import copy
from dataclasses import dataclass, field


nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download("wordnet")
nltk.download("stopwords")

# Filter, enable for removal of stopwords and punctuation
FILTER = True

STSS_131_DATA = "data/STSS-131.csv"

# Logging

LOGGING_LEVEL = logging.INFO
logger = logging.getLogger()
logger.setLevel(LOGGING_LEVEL)
# Formatter of log
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler = logging.StreamHandler()
handler.setLevel(LOGGING_LEVEL)
handler.setFormatter(formatter)
logger.addHandler(handler)


def default_field(obj):
    return field(default_factory=lambda: obj)


@dataclass
class WordToken:

    word: str
    tag: str
    lemma: str
    synsets: List[Synset]
    hypernyms: List[Synset] = default_field([])
    hyponyms: List[Synset] = default_field([])


def isPunctuation(token):
    for char in token:
        if char.isalpha() or char.isdigit():
            return False
    return True


def isStopword(token):
    if token.lower() in stopwords.words("english"):
        return True
    else:
        return False


def wordnetPosTag(tag):
    """
    Convert POS codes to Wordnet tags:

    ADJ, ADJ_SAT, ADV, NOUN, VERB = "a", "s", "r", "n", "v"


    # WordNet POS tags are: NOUN = 'n', ADJ = 's', VERB = 'v', ADV = 'r', ADJ_SAT = 'a'
    # Descriptions (c) https://web.stanford.edu/~jurafsky/slp3/10.pdf

    Source for tag_map: https://stackoverflow.com/questions/5364493/lemmatizing-pos-tagged-words-with-nltk
    """
    tag_map = {
        "CC": None,  # coordin. conjunction (and, but, or)
        "CD": wn.NOUN,  # cardinal number (one, two)
        "DT": None,  # determiner (a, the)
        "EX": wn.ADV,  # existential ‘there’ (there)
        "FW": None,  # foreign word (mea culpa)
        "IN": wn.ADV,  # preposition/sub-conj (of, in, by)
        # "JJ": [wn.ADJ, wn.ADJ_SAT],  # adjective (yellow)
        "JJ": wn.ADJ, 
        # "JJR": [wn.ADJ, wn.ADJ_SAT],  # adj., comparative (bigger)
        "JJR": wn.ADJ, 
        # "JJS": [wn.ADJ, wn.ADJ_SAT],  # adj., superlative (wildest)
        "JJS": wn.ADJ, 
        "LS": None,  # list item marker (1, 2, One)
        "MD": None,  # modal (can, should)
        "NN": wn.NOUN,  # noun, sing. or mass (llama)
        "NNS": wn.NOUN,  # noun, plural (llamas)
        "NNP": wn.NOUN,  # proper noun, sing. (IBM)
        "NNPS": wn.NOUN,  # proper noun, plural (Carolinas)
        "PDT": [wn.ADJ, wn.ADJ_SAT],  # predeterminer (all, both)
        "POS": None,  # possessive ending (’s )
        "PRP": None,  # personal pronoun (I, you, he)
        "PRP$": None,  # possessive pronoun (your, one’s)
        "RB": wn.ADV,  # adverb (quickly, never)
        "RBR": wn.ADV,  # adverb, comparative (faster)
        "RBS": wn.ADV,  # adverb, superlative (fastest)
        "RP": [wn.ADJ, wn.ADJ_SAT],  # particle (up, off)
        "SYM": None,  # symbol (+,%, &)
        "TO": None,  # “to” (to)
        "UH": None,  # interjection (ah, oops)
        "VB": wn.VERB,  # verb base form (eat)
        "VBD": wn.VERB,  # verb past tense (ate)
        "VBG": wn.VERB,  # verb gerund (eating)
        "VBN": wn.VERB,  # verb past participle (eaten)
        "VBP": wn.VERB,  # verb non-3sg pres (eat)
        "VBZ": wn.VERB,  # verb 3sg pres (eats)
        "WDT": None,  # wh-determiner (which, that)
        "WP": None,  # wh-pronoun (what, who)
        "WP$": None,  # possessive (wh- whose)
        "WRB": None,  # wh-adverb (how, where)
        "$": None,  #  dollar sign ($)
        "#": None,  # pound sign (#)
        "“": None,  # left quote (‘ or “)
        "”": None,  # right quote (’ or ”)
        "(": None,  # left parenthesis ([, (, {, <)
        ")": None,  # right parenthesis (], ), }, >)
        ",": None,  # comma (,)
        ".": None,  # sentence-final punc (. ! ?)
        ":": None,  # mid-sentence punc (: ; ... – -)
    }
    return tag_map.get(tag)


def genMappedWords(tokens):
    """
    Convert each token into object with details (lemma, tag and synset included). Also maps Treebank tags into WordNet tags.


    :param tokens: List of tokens with tag data (tuple)
    """

    wordlist_enriched = []
    lemmatizer = WordNetLemmatizer()
    for token in tokens:
        if not token[0]:
            logger.warning("Empty token in sentences.")
            continue
        # Check for punctuation or valid english word in WordNet database
        if FILTER:
            if isPunctuation(token[0]) or isStopword(token[0]):
                logger.debug(f"Punctuation or stopword detected for word '{token[0]}'.")
                continue
        wordNetTag = wordnetPosTag(token[1])
        if not wordNetTag:
            logger.warning(
                f"No matching tag in WordNet for given Treebank tag '{token[1]}'."
            )
            continue
        logger.debug(F"token: {token} tag: {wordNetTag}")
        lemma = lemmatizer.lemmatize(token[0], wordNetTag)
        wordlist_enriched.append(
            WordToken(token[0], wordNetTag, lemma, wn.synsets(token[0]))
        )
    return wordlist_enriched


def addHypernymsHyponyms(wordlist: List[WordToken]):
    """
    Method for adding list of Hypernyms and Hyponyms for each word in sentence.
    
    In this case, we are only interested about nouns and verbs.

    Sentece has been tagged and tokenized already. It is in form of WordToken list object.
    """

    for i, word in enumerate(wordlist):
        word.hypernyms = (
            []
        )  # For some reason, need to reset this that old content is not carried for next word
        word.hyponyms = []
        if word.tag == wn.NOUN or word.tag == wn.VERB:
            for synset in word.synsets:
                word.hypernyms.extend(synset.hypernyms())
                word.hyponyms.extend(synset.hyponyms())

            logger.debug(f"Added hypernyms and hyponyms for word {word.word}")

    # if word.tag == wn.VERB:
    #     for synset in word.synsets:
    #         word.hypernyms.append(synset.hypernyms())

    #     logger.debug(f"Added hypernyms for VERB {word.word}")
    #     for test in word.hypernyms:
    #         print(test)


def measureSimilarity(sentence1, sentence2):
    """
    Method implementing the equation presented in assignment.
    """

    s1_details, s2_details = preprocess(sentence1, sentence2)
    s1_verbs = []
    s2_verbs = []
    s1_nouns = []
    s2_nouns = []

    # First, get only tokens tagged as VERB or NOUN

    for word in s1_details:
        if word.tag == wn.VERB:
            s1_verbs.append(word)
        if word.tag == wn.NOUN:
            s1_nouns.append(word)

    for word in s2_details:
        if word.tag == wn.VERB:
            s2_verbs.append(word)
        if word.tag == wn.NOUN:
            s2_nouns.append(word)

    # print(s1_verbs)
    # print(s1_nouns)
    # Get all hypernyms of each VERB in sentence
    s1_verb_hypernyms = [hypernym for i in s1_verbs for hypernym in i.hypernyms]
    s2_verb_hypernyms = [hypernym for i in s2_verbs for hypernym in i.hypernyms]
    # Get all hyponyms of each VERB in sentence
    s1_verb_hyponyms = [hyponyms for i in s1_verbs for hyponyms in i.hyponyms]
    s2_verb_hyponyms = [hyponyms for i in s2_verbs for hyponyms in i.hyponyms]

    # Intersection of VERB hypernyms between two sentences
    verbs_hypernyms_intersection = list(set(s1_verb_hypernyms) & set(s2_verb_hypernyms))
    logger.debug(
        f"Length of list of verb hypernyms intersection {len(verbs_hypernyms_intersection)}"
    )
    # Intersection of VERB hyponyms between two sentences
    verbs_hyponyms_intersection = list(set(s1_verb_hyponyms) & set(s2_verb_hyponyms))
    logger.debug(
        f"Length of list of verb hyponyms intersection {len(verbs_hyponyms_intersection)}"
    )
    # Get all hypernyms of each NOUN in sentence
    s1_noun_hypernyms = [hypernym for i in s1_nouns for hypernym in i.hypernyms]
    s2_noun_hypernyms = [hypernym for i in s2_nouns for hypernym in i.hypernyms]
    nouns_hypernyms_intersection = list(set(s1_noun_hypernyms) & set(s2_noun_hypernyms))
    logger.debug(
        f"Length of list of noun hypernyms intersection {len(nouns_hypernyms_intersection)}"
    )
    # Get all hyponyms of each NOUN in sentence
    s1_noun_hyponyms = [hyponyms for i in s1_nouns for hyponyms in i.hyponyms]
    s2_noun_hyponyms = [hyponyms for i in s2_nouns for hyponyms in i.hyponyms]
    nouns_hyponyms_intersection = list(set(s1_noun_hyponyms) & set(s2_noun_hyponyms))
    logger.debug(
        f"Length of list of noun hyponyms intersection {len(nouns_hyponyms_intersection)}"
    )
    # Make union for all noun hypernyms and all verb hyponyms
    union_noun_hypernyms = list(set(s1_noun_hypernyms) | set(s2_noun_hypernyms))
    union_verb_hyponyms = list(set(s1_verb_hyponyms) | set(s2_verb_hyponyms))
    print("noun ypernym intersection")
    print(nouns_hypernyms_intersection)
    print("\n\n")
    print("union_noun_hypernyms")
    print(union_noun_hypernyms)
    #print(nouns_hypernyms_intersection / union_noun_hypernyms)

    res1 = len(set(nouns_hypernyms_intersection) & set(union_noun_hypernyms)) / float(
        len(set(nouns_hypernyms_intersection) | set(union_noun_hypernyms))
    )
    res2 = len(set(verbs_hyponyms_intersection) & set(union_verb_hyponyms)) / float(
        len(set(verbs_hyponyms_intersection) | set(union_verb_hyponyms))
    )
    print( res1, res2)


    nouns_synsets1 = []
    nouns_synsets2 = []
    verbs_synsets1 = []
    verbs_synsets2 = []

    # Combine synsets of each noun in both sentence
    for noun in s1_nouns:
        nouns_synsets1 +=  list(set(noun.synsets) - set(nouns_synsets1))
    for noun in s2_nouns:
        nouns_synsets2 += list(set(noun.synsets) - set(nouns_synsets2))
     # Combine synsets of each verb in both sentence
    for verb in s1_verbs:
        verbs_synsets1 +=  list(set(verb.synsets) - set(verbs_synsets1))
    for verb in s2_verbs:
        verbs_synsets2 +=  list(set(verb.synsets) - set(verbs_synsets2))


    score, count, best_score = 0.0, 0, 0
    for synset in nouns_synsets1:
        # print(synset)
        # Get the similarity value of the most similar word in the other sentence
        synset_scores = [synset.path_similarity(ss) for ss in nouns_synsets2 if synset.path_similarity(ss)]
        if synset_scores:
            best_score = max(synset_scores)
 
        # Check that the similarity could have been computed
        if best_score is not None:
            score += best_score
            count += 1
    
    noun_score = score / count

    score, count, best_score = 0.0, 0, 0
    for synset in verbs_synsets1:
        # print(synset)
        # Get the similarity value of the most similar word in the other sentence
        synset_scores = [synset.path_similarity(ss) for ss in verbs_synsets2 if synset.path_similarity(ss)]
        if synset_scores:
            best_score = max(synset_scores)
 
        # Check that the similarity could have been computed
        if best_score is not None:
            score += best_score
            count += 1

    verb_score = score / count

    final_score = (noun_score * res1 + verb_score * res2 ) / 2
    print(f"Final score is {final_score * 4}") 
    return final_score * 4

def preprocess(sentence1, sentence2):
    """
    Method for tokenizing and tagging sentence pairs
    """

    # Tokenize sentences
    s1_tokens = nltk.word_tokenize(sentence1)
    s2_tokens = nltk.word_tokenize(sentence2)
    logger.debug(f"Tokens of the first sentence: {s1_tokens}")
    logger.debug(f"Tokens of the second sentence: {s2_tokens}")

    # Tag each token
    s1_tagged = pos_tag(s1_tokens)
    s2_tagged = pos_tag(s2_tokens)
    logger.debug(f"Tagged tokens of the first sentence: {s1_tagged}")
    logger.debug(f"Tagged tokens of the second sentence: {s2_tagged}")

    s1_details = genMappedWords(s1_tagged)
    s2_details = genMappedWords(s2_tagged)

    addHypernymsHyponyms(s1_details)
    addHypernymsHyponyms(s2_details)
    logger.info("Preprocessing done.")

    return s1_details, s2_details


def main():

    # List of sentence objects
    sentences = readCSV(STSS_131_DATA)
    scores = []
    for i in range (0,5):
        scores.append(measureSimilarity(sentences[i].first_sentence, sentences[i].second_sentence))

    print(scores)


if __name__ == "__main__":
    main()
    # pass
