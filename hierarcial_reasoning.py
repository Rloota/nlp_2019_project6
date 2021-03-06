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
from scipy import stats
import sys
import csv

nltk.download("averaged_perceptron_tagger")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")

# Filter, enable for removal of stopwords and punctuation

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
        "PDT": wn.ADJ,  # predeterminer (all, both)
        "POS": None,  # possessive ending (’s )
        "PRP": None,  # personal pronoun (I, you, he)
        "PRP$": None,  # possessive pronoun (your, one’s)
        "RB": wn.ADV,  # adverb (quickly, never)
        "RBR": wn.ADV,  # adverb, comparative (faster)
        "RBS": wn.ADV,  # adverb, superlative (fastest)
        "RP": wn.ADJ,  # particle (up, off)
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


def genMappedWords(tokens, filtering=True):
    """
    Convert each token into object with details (lemma, tag and synset included). Also maps Treebank tags into WordNet tags.


    :param tokens: List of tokens with tag data (tuple)
    """
    FILTER = filtering

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
        logger.debug(f"token: {token} tag: {wordNetTag}")
        lemma = lemmatizer.lemmatize(token[0], wordNetTag)
        wordlist_enriched.append(
            WordToken(token[0], wordNetTag, lemma, wn.synsets(token[0]))
        )
    return wordlist_enriched


def addHypernymsHyponyms(wordlist: List[WordToken]):
    """
    Method for adding list of Hypernyms and Hyponyms for each word in sentence.
    
    In this case, we are only interested about nouns and verbs.

    Sentence has been tagged and tokenized already. It is in form of WordToken list object.
    """

    for i, word in enumerate(wordlist):
        word.hypernyms = set()
        # For some reason, need to reset this that old content is not carried for next word
        word.hyponyms = set()
        if word.tag == wn.NOUN or word.tag == wn.VERB:
            for synset in word.synsets:
                word.hypernyms.update(set(synset.hypernyms()))
                word.hyponyms.update(set(synset.hyponyms()))

            logger.debug(f"Added hypernyms and hyponyms for word {word.word}")

    # if word.tag == wn.VERB:
    #     for synset in word.synsets:
    #         word.hypernyms.append(synset.hypernyms())

    #     logger.debug(f"Added hypernyms for VERB {word.word}")
    #     for test in word.hypernyms:
    #         print(test)


def measureSimilarity(sentence1, sentence2, filtering=True):
    """
    Method implementing the equation presented in assignment.
    """

    s1_details, s2_details = preprocess(sentence1, sentence2, filtering)
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

    if not s1_verbs:
        logger.warning(f"First sentence has no single verb. Sentence is: {sentence1}")
    if not s2_verbs:
        logger.warning(f"Second sentence has no single verb. Sentence is {sentence2}")
    if not s1_nouns:
        logger.warning(f"First sentence has no single noun. Sentence is {sentence1}")
    if not s2_nouns:
        logger.warning(f"Second sentence has no single noun. Sentence is {sentence2}")

    # print(s1_nouns)
    s1_verb_hypernyms = set()
    s2_verb_hypernyms = set()
    s1_verb_hyponyms = set()
    s2_verb_hyponyms = set()
    # Get all hyponyms of each VERB in sentence
    # Get all hypernyms of each VERB in sentence
    for i in s1_verbs:
        s1_verb_hypernyms.update(i.hypernyms)
        s1_verb_hyponyms.update(i.hyponyms)

    for i in s2_verbs:
        s2_verb_hypernyms.update(i.hypernyms)
        s2_verb_hyponyms.update(i.hyponyms)

    # Intersection of VERB hypernyms between two sentences
    verbs_hypernyms_intersection = s1_verb_hypernyms.intersection(s2_verb_hypernyms)
    logger.debug(
        f"Length of list of verb hypernyms intersection {len(verbs_hypernyms_intersection)}"
    )
    # Intersection of VERB hyponyms between two sentences
    verbs_hyponyms_intersection = s1_verb_hyponyms.intersection(s2_verb_hyponyms)
    logger.debug(
        f"Length of list of verb hyponyms intersection {len(verbs_hyponyms_intersection)}"
    )

    s1_noun_hypernyms = set()
    s2_noun_hypernyms = set()
    s1_noun_hyponyms = set()
    s2_noun_hyponyms = set()

    # Get all hypernyms of each NOUN in sentence
    # Get all hyponyms of each NOUN in sentence

    for i in s1_nouns:
        s1_noun_hypernyms.update(i.hypernyms)
        s1_noun_hyponyms.update(i.hyponyms)

    for i in s2_nouns:
        s2_noun_hypernyms.update(i.hypernyms)
        s2_noun_hyponyms.update(i.hyponyms)

    nouns_hypernyms_intersection = s1_noun_hypernyms.intersection(s2_noun_hypernyms)
    logger.debug(
        f"Length of list of noun hypernyms intersection {len(nouns_hypernyms_intersection)}"
    )

    nouns_hyponyms_intersection = s1_noun_hyponyms.intersection(s2_noun_hyponyms)
    logger.debug(
        f"Length of list of noun hyponyms intersection {len(nouns_hyponyms_intersection)}"
    )
    # Make union for all noun hypernyms and all verb hyponyms
    union_noun_hypernyms = s1_noun_hypernyms.union(s2_noun_hypernyms)
    union_verb_hyponyms = s1_verb_hyponyms.union(s2_verb_hyponyms)

    try:
        res1 = len(nouns_hypernyms_intersection) / len(union_noun_hypernyms)
        res2 = len(verbs_hyponyms_intersection) / len(union_verb_hyponyms)
    except ZeroDivisionError:
        logger.warning("Division by ZERO")
        return "DIVISION_BY_ZERO_NO_HYPONYMS"

    nouns_synsets1 = set()
    nouns_synsets2 = set()
    verbs_synsets1 = set()
    verbs_synsets2 = set()

    # Combine synsets of each noun in both sentence
    for noun in s1_nouns:
        nouns_synsets1.update(set(noun.synsets))
    for noun in s2_nouns:
        nouns_synsets2.update(set(noun.synsets))
    # Combine synsets of each verb in both sentence
    for verb in s1_verbs:
        verbs_synsets1.update(set(verb.synsets))
    for verb in s2_verbs:
        verbs_synsets2.update(set(verb.synsets))

    score, count, best_score = 0.0, 0, 0
    for synset in nouns_synsets1:
        # print(synset)
        # Get the similarity value of the most similar word in the other sentence
        synset_scores = [
            synset.path_similarity(ss)
            for ss in nouns_synsets2
            if synset.path_similarity(ss)
        ]
        if synset_scores:
            best_score = max(synset_scores)
            score += best_score
            count += 1

    if count > 0:
        noun_score = score / count
        noun_score *= 4
    else:
        noun_score = 0
        # sys.exit()
        # return "NOT_KNOWN"

    score, count, best_score = 0.0, 0, 0
    for synset in verbs_synsets1:
        # print(synset)
        # Get the similarity value of the most similar word in the other sentence
        synset_scores = [
            synset.path_similarity(ss)
            for ss in verbs_synsets2
            if synset.path_similarity(ss)
        ]
        if synset_scores:
            best_score = max(synset_scores)
            score += best_score
            count += 1

    if count > 0:
        verb_score = score / count
        verb_score *= 4
    else:
        verb_score = 0

    final_score = (noun_score * res1 + verb_score * res2) / 2

    return final_score


def preprocess(sentence1, sentence2, filtering=True):
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

    s1_details = genMappedWords(s1_tagged, filtering)
    s2_details = genMappedWords(s2_tagged, filtering)

    addHypernymsHyponyms(s1_details)
    addHypernymsHyponyms(s2_details)
    logger.info("Preprocessing done.")

    return s1_details, s2_details


def main():

    # List of sentence objects
    sentences = readCSV(STSS_131_DATA)
    STSS_values = []
    scores = []
    final_data = []
    init = 66
    for i, j in enumerate(sentences):
        # for i in range(0,3):
        result = measureSimilarity(
            sentences[i].first_sentence, sentences[i].second_sentence
        )
        # scores.append((init, result))
        print(f" CASE {init}")
        # if init == 127:
        #     break
        if isinstance(result, str):
            pass
        else:
            scores.append(result)
            final_data.append((init, result))
            STSS_values.append(j.human_SS)
        init += 1

    # print(final_data)
    p = stats.pearsonr(scores, STSS_values)
    print(f"Pearsons  Correla-tion  Coefficient  against  Human  Judgement: {p}")
    with open('new_method_no_punc_stopword.csv', 'w') as f:
        writer = csv.writer(f)
        for row in final_data:
            writer.writerow(row)

    # STSS_values = []
    # scores = []
    # final_data = []
    # init = 66
    # for i, j in enumerate(sentences):
    #     # for i in range(0,3):
    #     result = measureSimilarity(
    #         sentences[i].first_sentence, sentences[i].second_sentence, filtering=False
    #     )
    #     # scores.append((init, result))
    #     print(f" CASE {init}")
    #     # if init == 127:
    #     #     break
    #     if isinstance(result, str):
    #         pass
    #     else:
    #         scores.append(result)
    #         final_data.append((init, result))
    #         STSS_values.append(j.human_SS)
    #     init += 1

    # p = stats.pearsonr(scores, STSS_values)
    # print(f"{p}")
    # with open('new_method_no_preprocessing.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     for row in final_data:
    #         writer.writerow(row)

    # print(measureSimilarity(
    #         "We tried to bargain with him but it made no difference, he still didn’t change his mind.", "I tried bargaining with him, but he just wouldn’t listen."
    #     ))

"""
Pearson with preprocessing: (0.4277631388197102, 0.0005244450772423713)
Pearson without preprocessing: (0.3336452222256247, 0.0061874049904368554)
"""

if __name__ == "__main__":
    main()
    # pass
