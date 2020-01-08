import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus.reader.wordnet import Synset
from nltk import pos_tag
import logging
from utils import readCSV

from copy import copy
from dataclasses import dataclass

nltk.download("wordnet")
nltk.download("stopwords")

# Filter, enable for removal of stopwords and punctuation
FILTER = True

STSS_131_DATA = "data/STSS-131.csv"

# Logging

LOGGING_LEVEL = logging.DEBUG
logger = logging.getLogger()
logger.setLevel(LOGGING_LEVEL)
# Formatter of log
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler = logging.StreamHandler()
handler.setLevel(LOGGING_LEVEL)
handler.setFormatter(formatter)
logger.addHandler(handler)


@dataclass
class WordToken:

    word: str
    tag: str
    lemma: str
    synset: Synset


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
            'CC':None, # coordin. conjunction (and, but, or)  
            'CD':wn.NOUN, # cardinal number (one, two)             
            'DT':None, # determiner (a, the)                    
            'EX':wn.ADV, # existential ‘there’ (there)           
            'FW':None, # foreign word (mea culpa)             
            'IN':wn.ADV, # preposition/sub-conj (of, in, by)   
            'JJ':[wn.ADJ, wn.ADJ_SAT], # adjective (yellow)                  
            'JJR':[wn.ADJ, wn.ADJ_SAT], # adj., comparative (bigger)          
            'JJS':[wn.ADJ, wn.ADJ_SAT], # adj., superlative (wildest)           
            'LS':None, # list item marker (1, 2, One)          
            'MD':None, # modal (can, should)                    
            'NN':wn.NOUN, # noun, sing. or mass (llama)          
            'NNS':wn.NOUN, # noun, plural (llamas)                  
            'NNP':wn.NOUN, # proper noun, sing. (IBM)              
            'NNPS':wn.NOUN, # proper noun, plural (Carolinas)
            'PDT':[wn.ADJ, wn.ADJ_SAT], # predeterminer (all, both)            
            'POS':None, # possessive ending (’s )               
            'PRP':None, # personal pronoun (I, you, he)     
            'PRP$':None, # possessive pronoun (your, one’s)    
            'RB':wn.ADV, # adverb (quickly, never)            
            'RBR':wn.ADV, # adverb, comparative (faster)        
            'RBS':wn.ADV, # adverb, superlative (fastest)     
            'RP':[wn.ADJ, wn.ADJ_SAT], # particle (up, off)
            'SYM':None, # symbol (+,%, &)
            'TO':None, # “to” (to)
            'UH':None, # interjection (ah, oops)
            'VB':wn.VERB, # verb base form (eat)
            'VBD':wn.VERB, # verb past tense (ate)
            'VBG':wn.VERB, # verb gerund (eating)
            'VBN':wn.VERB, # verb past participle (eaten)
            'VBP':wn.VERB, # verb non-3sg pres (eat)
            'VBZ':wn.VERB, # verb 3sg pres (eats)
            'WDT':None, # wh-determiner (which, that)
            'WP':None, # wh-pronoun (what, who)
            'WP$':None, # possessive (wh- whose)
            'WRB':None, # wh-adverb (how, where)
            '$':None, #  dollar sign ($)
            '#':None, # pound sign (#)
            '“':None, # left quote (‘ or “)
            '”':None, # right quote (’ or ”)
            '(':None, # left parenthesis ([, (, {, <)
            ')':None, # right parenthesis (], ), }, >)
            ',':None, # comma (,)
            '.':None, # sentence-final punc (. ! ?)
            ':':None # mid-sentence punc (: ; ... – -)
        }
    return tag_map.get(tag)



def genMappedWords(tokens):
    """
    Convert each token into object with details. Also maps Treebank tags into WordNet tags.

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
            logger.warning(f"No matching tag in WordNet for given Treebank tag '{token[1]}'.")
            continue
        lemma = lemmatizer.lemmatize(token[0], wordNetTag)
        wordlist_enriched.append(
            WordToken(
                token[0],
                wordNetTag,
                lemma,
                wn.synsets(token[0]),
            )
        )
    return wordlist_enriched

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
    for i in s1_details:
        print(i.lemma)
        print(i.tag)
        print(i.synset)


def main():

    # List of sentence objects
    sentences = readCSV(STSS_131_DATA)

    preprocess(sentences[0].first_sentence, sentences[0].second_sentence)


if __name__ == "__main__":
    main()
    # pass
