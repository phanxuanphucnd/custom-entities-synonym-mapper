import re
from typing import List, Text

from fuzzywuzzy import fuzz

PATTERNS = {
    "[àáảãạăắằẵặẳâầấậẫẩ]": "a",
    "[đ]": "d",
    "[èéẻẽẹêềếểễệ]": "e",
    "[ìíỉĩị]": "i",
    "[òóỏõọôồốổỗộơờớởỡợ]": "o",
    "[ùúủũụưừứửữự]": "u",
    "[ỳýỷỹỵ]": "y",
}


def remove_tone_notation(string):
    """ Remove Vietnamese tone notation.

    :param string: input string to be converted.

    returns: converted string which is removed ton notation.
    """
    output = string.lower()
    for regex, replace in PATTERNS.items():
        output = re.sub(regex, replace, output)
        output = re.sub("_", " ", output)
    return output


def fuzz_partial_ratio(seq1, seq2):
    """ Calculate partial_ratio between 2 string. """
    score = fuzz.partial_ratio(seq1, seq2) / 100
    return score

def fuzz_ratio(seq1, seq2):
    """ Calculate ratio between 2 string. """
    score = fuzz.ratio(seq1, seq2) / 100
    return score

def get_max_fuzzy_score(string, list_string):
    """ Get maximum the simimarlity score in list using Fuzzy Matching algorithm

    :param string: The string input to compare
    :param list_string: List of string correspondings a given object type

    returns: list object type with score equal or greater than threshold
    """
    if not string or not list_string:
        return []
        
    string = remove_tone_notation(string)
    list_string = [remove_tone_notation(s) for s in list_string]
    scores = []

    for _, value in enumerate(list_string):
        value = remove_tone_notation(value)
        score = fuzz_ratio(string, value)
        scores.append(score)

    return max(scores)
