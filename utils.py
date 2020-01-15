from dataclasses import dataclass
from typing import List
import csv
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


def readCSV(filename) -> List[SentencePair]:
    """
    Read sample data from CSV file and generate dataobject for each sentence pair.

    Supports data format in STSS-131 sample data.

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