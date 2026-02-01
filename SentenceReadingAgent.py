"""
A semantic frame-based question answering agent that extracts thematic roles
from sentences and answers WH-questions by querying the appropriate frame slots.

References:

    Jurafsky, D., & Martin, J. H. (2009). Speech and Language Processing (2nd ed.).
    Pearson Prentice Hall.

    Key chapters:
    - Ch. 3: Words and Transducers, pp. 57-90 (tokenization)
    - Ch. 5: Part-of-Speech Tagging, pp. 123-164
    - Ch. 19: Lexical Semantics, pp. 617-656 (thematic roles)
    - Ch. 20: Computational Lexical Semantics, pp. 657-700 (semantic role labeling)
    - Ch. 23: Question Answering and Summarization, pp. 767-812 (answer type detection)

    Jurafsky, D., & Martin, J. H. (2026). Speech and Language Processing (3rd ed.).
    Online manuscript. https://web.stanford.edu/~jurafsky/slp3/

    Key chapters:
    - Ch. 2: Words and Tokens
      https://web.stanford.edu/~jurafsky/slp3/2.pdf
    - Ch. 17: Sequence Labeling for POS and Named Entities
      https://web.stanford.edu/~jurafsky/slp3/17.pdf
    - Ch. 21: Semantic Role Labeling and Argument Structure
      https://web.stanford.edu/~jurafsky/slp3/21.pdf
"""

import re


class SentenceReadingAgent:
    """An agent that reads sentences and answers questions.

    Semantic role labeling pipeline:
    1. Tokenization - Jurafsky & Martin (2009), Ch. 3; (2026), Ch. 2
    2. POS tagging - Jurafsky & Martin (2009), Ch. 5; (2026), Ch. 17
    3. Frame extraction - Jurafsky & Martin (2009), Ch. 19-20; (2026), Ch. 21
    4. Question classification - Jurafsky & Martin (2009), Ch. 23
    5. Frame querying - Jurafsky & Martin (2009), Ch. 23
    """

    def __init__(self):
        # POS (Part of Speech): tells you if a word is a NOUN, VERB, ADJ, etc.
        # Lemma: base form of the word ("brought" → "bring")
        self.WORD_DATA = {
            "Serena": {"pos": "PROPN", "lemma": "serena"},
            "Andrew": {"pos": "PROPN", "lemma": "andrew"},
            "Bobbie": {"pos": "PROPN", "lemma": "bobbie"},
            "Cason": {"pos": "PROPN", "lemma": "cason"},
            "David": {"pos": "PROPN", "lemma": "david"},
            "Farzana": {"pos": "PROPN", "lemma": "farzana"},
            "Frank": {"pos": "PROPN", "lemma": "frank"},
            "Hannah": {"pos": "PROPN", "lemma": "hannah"},
            "Ida": {"pos": "PROPN", "lemma": "ida"},
            "Irene": {"pos": "PROPN", "lemma": "irene"},
            "Jim": {"pos": "PROPN", "lemma": "jim"},
            "Jose": {"pos": "PROPN", "lemma": "jose"},
            "Keith": {"pos": "PROPN", "lemma": "keith"},
            "Laura": {"pos": "PROPN", "lemma": "laura"},
            "Lucy": {"pos": "PROPN", "lemma": "lucy"},
            "Meredith": {"pos": "PROPN", "lemma": "meredith"},
            "Nick": {"pos": "PROPN", "lemma": "nick"},
            "Ada": {"pos": "PROPN", "lemma": "ada"},
            "Yeeling": {"pos": "PROPN", "lemma": "yeeling"},
            "Yan": {"pos": "PROPN", "lemma": "yan"},
            "the": {"pos": "PRON", "lemma": "the"},
            "of": {"pos": "ADP", "lemma": "of"},
            "to": {"pos": "PART", "lemma": "to"},
            "and": {"pos": "CCONJ", "lemma": "and"},
            "a": {"pos": "PRON", "lemma": "a"},
            "in": {"pos": "ADP", "lemma": "in"},
            "is": {"pos": "AUX", "lemma": "be"},
            "it": {"pos": "PRON", "lemma": "it"},
            "you": {"pos": "PRON", "lemma": "you"},
            "that": {"pos": "SCONJ", "lemma": "that"},
            "he": {"pos": "PRON", "lemma": "he"},
            "was": {"pos": "AUX", "lemma": "be"},
            "for": {"pos": "ADP", "lemma": "for"},
            "on": {"pos": "ADP", "lemma": "on"},
            "are": {"pos": "AUX", "lemma": "be"},
            "with": {"pos": "ADP", "lemma": "with"},
            "as": {"pos": "ADP", "lemma": "as"},
            "I": {"pos": "PRON", "lemma": "I"},
            "his": {"pos": "PRON", "lemma": "his"},
            "they": {"pos": "PRON", "lemma": "they"},
            "be": {"pos": "AUX", "lemma": "be"},
            "at": {"pos": "ADP", "lemma": "at"},
            "one": {"pos": "NUM", "lemma": "one"},
            "have": {"pos": "VERB", "lemma": "have"},
            "this": {"pos": "PRON", "lemma": "this"},
            "from": {"pos": "ADP", "lemma": "from"},
            "or": {"pos": "CCONJ", "lemma": "or"},
            "had": {"pos": "VERB", "lemma": "have"},
            "by": {"pos": "ADP", "lemma": "by"},
            "hot": {"pos": "ADJ", "lemma": "hot"},
            "but": {"pos": "CCONJ", "lemma": "but"},
            "some": {"pos": "PRON", "lemma": "some"},
            "what": {"pos": "PRON", "lemma": "what"},
            "there": {"pos": "PRON", "lemma": "there"},
            "we": {"pos": "PRON", "lemma": "we"},
            "can": {"pos": "AUX", "lemma": "can"},
            "out": {"pos": "ADV", "lemma": "out"},
            "other": {"pos": "ADJ", "lemma": "other"},
            "were": {"pos": "AUX", "lemma": "be"},
            "all": {"pos": "PRON", "lemma": "all"},
            "your": {"pos": "PRON", "lemma": "your"},
            "when": {"pos": "SCONJ", "lemma": "when"},
            "up": {"pos": "ADV", "lemma": "up"},
            "use": {"pos": "NOUN", "lemma": "use"},
            "word": {"pos": "NOUN", "lemma": "word"},
            "how": {"pos": "SCONJ", "lemma": "how"},
            "said": {"pos": "VERB", "lemma": "say"},
            "an": {"pos": "PRON", "lemma": "an"},
            "each": {"pos": "PRON", "lemma": "each"},
            "she": {"pos": "PRON", "lemma": "she"},
            "which": {"pos": "PRON", "lemma": "which"},
            "do": {"pos": "VERB", "lemma": "do"},
            "their": {"pos": "PRON", "lemma": "their"},
            "time": {"pos": "NOUN", "lemma": "time"},
            "if": {"pos": "SCONJ", "lemma": "if"},
            "will": {"pos": "AUX", "lemma": "will"},
            "way": {"pos": "NOUN", "lemma": "way"},
            "about": {"pos": "ADV", "lemma": "about"},
            "many": {"pos": "ADJ", "lemma": "many"},
            "then": {"pos": "ADV", "lemma": "then"},
            "them": {"pos": "PRON", "lemma": "they"},
            "would": {"pos": "AUX", "lemma": "would"},
            "write": {"pos": "VERB", "lemma": "write"},
            "wrote": {"pos": "VERB", "lemma": "write"},
            "like": {"pos": "INTJ", "lemma": "like"},
            "so": {"pos": "ADV", "lemma": "so"},
            "these": {"pos": "PRON", "lemma": "these"},
            "her": {"pos": "PRON", "lemma": "she"},
            "long": {"pos": "ADJ", "lemma": "long"},
            "make": {"pos": "VERB", "lemma": "make"},
            "thing": {"pos": "NOUN", "lemma": "thing"},
            "see": {"pos": "VERB", "lemma": "see"},
            "him": {"pos": "PRON", "lemma": "he"},
            "two": {"pos": "NUM", "lemma": "two"},
            "has": {"pos": "VERB", "lemma": "have"},
            "look": {"pos": "VERB", "lemma": "look"},
            "more": {"pos": "ADV", "lemma": "more"},
            "day": {"pos": "NOUN", "lemma": "day"},
            "could": {"pos": "AUX", "lemma": "could"},
            "go": {"pos": "VERB", "lemma": "go"},
            "come": {"pos": "VERB", "lemma": "come"},
            "did": {"pos": "VERB", "lemma": "do"},
            "my": {"pos": "PRON", "lemma": "my"},
            "sound": {"pos": "VERB", "lemma": "sound"},
            "no": {"pos": "INTJ", "lemma": "no"},
            "most": {"pos": "ADV", "lemma": "most"},
            "number": {"pos": "NOUN", "lemma": "number"},
            "who": {"pos": "PRON", "lemma": "who"},
            "over": {"pos": "ADV", "lemma": "over"},
            "know": {"pos": "VERB", "lemma": "know"},
            "water": {"pos": "NOUN", "lemma": "water"},
            "than": {"pos": "ADP", "lemma": "than"},
            "call": {"pos": "VERB", "lemma": "call"},
            "first": {"pos": "ADV", "lemma": "first"},
            "people": {"pos": "NOUN", "lemma": "people"},
            "may": {"pos": "AUX", "lemma": "may"},
            "down": {"pos": "ADV", "lemma": "down"},
            "side": {"pos": "NOUN", "lemma": "side"},
            "been": {"pos": "AUX", "lemma": "be"},
            "now": {"pos": "ADV", "lemma": "now"},
            "find": {"pos": "VERB", "lemma": "find"},
            "any": {"pos": "PRON", "lemma": "any"},
            "new": {"pos": "ADJ", "lemma": "new"},
            "work": {"pos": "VERB", "lemma": "work"},
            "part": {"pos": "NOUN", "lemma": "part"},
            "take": {"pos": "VERB", "lemma": "take"},
            "get": {"pos": "VERB", "lemma": "get"},
            "place": {"pos": "NOUN", "lemma": "place"},
            "made": {"pos": "VERB", "lemma": "make"},
            "live": {"pos": "ADJ", "lemma": "live"},
            "where": {"pos": "SCONJ", "lemma": "where"},
            "after": {"pos": "ADP", "lemma": "after"},
            "back": {"pos": "ADV", "lemma": "back"},
            "little": {"pos": "ADJ", "lemma": "little"},
            "only": {"pos": "ADV", "lemma": "only"},
            "round": {"pos": "ADJ", "lemma": "round"},
            "man": {"pos": "NOUN", "lemma": "man"},
            "year": {"pos": "NOUN", "lemma": "year"},
            "came": {"pos": "VERB", "lemma": "come"},
            "show": {"pos": "VERB", "lemma": "show"},
            "every": {"pos": "DET", "lemma": "every"},
            "good": {"pos": "ADJ", "lemma": "good"},
            "me": {"pos": "PRON", "lemma": "I"},
            "give": {"pos": "VERB", "lemma": "give"},
            "our": {"pos": "PRON", "lemma": "our"},
            "under": {"pos": "ADP", "lemma": "under"},
            "name": {"pos": "NOUN", "lemma": "name"},
            "very": {"pos": "ADV", "lemma": "very"},
            "through": {"pos": "ADP", "lemma": "through"},
            "just": {"pos": "ADV", "lemma": "just"},
            "form": {"pos": "NOUN", "lemma": "form"},
            "much": {"pos": "ADJ", "lemma": "much"},
            "great": {"pos": "ADJ", "lemma": "great"},
            "think": {"pos": "VERB", "lemma": "think"},
            "say": {"pos": "VERB", "lemma": "say"},
            "help": {"pos": "VERB", "lemma": "help"},
            "low": {"pos": "ADJ", "lemma": "low"},
            "line": {"pos": "NOUN", "lemma": "line"},
            "before": {"pos": "ADP", "lemma": "before"},
            "turn": {"pos": "VERB", "lemma": "turn"},
            "cause": {"pos": "VERB", "lemma": "cause"},
            "same": {"pos": "ADJ", "lemma": "same"},
            "mean": {"pos": "VERB", "lemma": "mean"},
            "differ": {"pos": "VERB", "lemma": "differ"},
            "move": {"pos": "VERB", "lemma": "move"},
            "right": {"pos": "INTJ", "lemma": "right"},
            "boy": {"pos": "NOUN", "lemma": "boy"},
            "old": {"pos": "ADJ", "lemma": "old"},
            "too": {"pos": "ADV", "lemma": "too"},
            "does": {"pos": "VERB", "lemma": "do"},
            "tell": {"pos": "VERB", "lemma": "tell"},
            "sentence": {"pos": "NOUN", "lemma": "sentence"},
            "set": {"pos": "VERB", "lemma": "set"},
            "three": {"pos": "NUM", "lemma": "three"},
            "want": {"pos": "VERB", "lemma": "want"},
            "air": {"pos": "NOUN", "lemma": "air"},
            "well": {"pos": "ADV", "lemma": "well"},
            "also": {"pos": "ADV", "lemma": "also"},
            "play": {"pos": "VERB", "lemma": "play"},
            "small": {"pos": "ADJ", "lemma": "small"},
            "end": {"pos": "NOUN", "lemma": "end"},
            "put": {"pos": "VERB", "lemma": "put"},
            "home": {"pos": "NOUN", "lemma": "home"},
            "read": {"pos": "VERB", "lemma": "read"},
            "hand": {"pos": "NOUN", "lemma": "hand"},
            "port": {"pos": "NOUN", "lemma": "port"},
            "large": {"pos": "ADJ", "lemma": "large"},
            "spell": {"pos": "VERB", "lemma": "spell"},
            "add": {"pos": "VERB", "lemma": "add"},
            "even": {"pos": "ADV", "lemma": "even"},
            "land": {"pos": "NOUN", "lemma": "land"},
            "here": {"pos": "ADV", "lemma": "here"},
            "must": {"pos": "AUX", "lemma": "must"},
            "big": {"pos": "ADJ", "lemma": "big"},
            "high": {"pos": "ADJ", "lemma": "high"},
            "such": {"pos": "ADJ", "lemma": "such"},
            "follow": {"pos": "VERB", "lemma": "follow"},
            "act": {"pos": "NOUN", "lemma": "act"},
            "why": {"pos": "SCONJ", "lemma": "why"},
            "ask": {"pos": "VERB", "lemma": "ask"},
            "men": {"pos": "NOUN", "lemma": "man"},
            "change": {"pos": "VERB", "lemma": "change"},
            "went": {"pos": "VERB", "lemma": "go"},
            "light": {"pos": "NOUN", "lemma": "light"},
            "kind": {"pos": "ADV", "lemma": "kind"},
            "off": {"pos": "ADP", "lemma": "off"},
            "need": {"pos": "VERB", "lemma": "need"},
            "house": {"pos": "PROPN", "lemma": "house"},
            "picture": {"pos": "NOUN", "lemma": "picture"},
            "try": {"pos": "VERB", "lemma": "try"},
            "us": {"pos": "PRON", "lemma": "we"},
            "again": {"pos": "ADV", "lemma": "again"},
            "animal": {"pos": "NOUN", "lemma": "animal"},
            "point": {"pos": "NOUN", "lemma": "point"},
            "mother": {"pos": "NOUN", "lemma": "mother"},
            "world": {"pos": "NOUN", "lemma": "world"},
            "near": {"pos": "ADP", "lemma": "near"},
            "build": {"pos": "VERB", "lemma": "build"},
            "self": {"pos": "NOUN", "lemma": "self"},
            "earth": {"pos": "NOUN", "lemma": "earth"},
            "father": {"pos": "PROPN", "lemma": "father"},
            "head": {"pos": "NOUN", "lemma": "head"},
            "stand": {"pos": "VERB", "lemma": "stand"},
            "own": {"pos": "ADJ", "lemma": "own"},
            "page": {"pos": "NOUN", "lemma": "page"},
            "should": {"pos": "AUX", "lemma": "should"},
            "country": {"pos": "NOUN", "lemma": "country"},
            "found": {"pos": "VERB", "lemma": "find"},
            "answer": {"pos": "VERB", "lemma": "answer"},
            "school": {"pos": "NOUN", "lemma": "school"},
            "grow": {"pos": "VERB", "lemma": "grow"},
            "study": {"pos": "NOUN", "lemma": "study"},
            "still": {"pos": "ADV", "lemma": "still"},
            "learn": {"pos": "VERB", "lemma": "learn"},
            "plant": {"pos": "NOUN", "lemma": "plant"},
            "cover": {"pos": "VERB", "lemma": "cover"},
            "food": {"pos": "NOUN", "lemma": "food"},
            "sun": {"pos": "PROPN", "lemma": "sun"},
            "four": {"pos": "NUM", "lemma": "four"},
            "thought": {"pos": "VERB", "lemma": "think"},
            "let": {"pos": "VERB", "lemma": "let"},
            "keep": {"pos": "VERB", "lemma": "keep"},
            "eye": {"pos": "NOUN", "lemma": "eye"},
            "never": {"pos": "ADV", "lemma": "never"},
            "last": {"pos": "ADJ", "lemma": "last"},
            "door": {"pos": "NOUN", "lemma": "door"},
            "between": {"pos": "ADP", "lemma": "between"},
            "city": {"pos": "NOUN", "lemma": "city"},
            "tree": {"pos": "NOUN", "lemma": "tree"},
            "cross": {"pos": "VERB", "lemma": "cross"},
            "since": {"pos": "SCONJ", "lemma": "since"},
            "hard": {"pos": "ADJ", "lemma": "hard"},
            "start": {"pos": "VERB", "lemma": "start"},
            "might": {"pos": "AUX", "lemma": "might"},
            "story": {"pos": "NOUN", "lemma": "story"},
            "saw": {"pos": "VERB", "lemma": "see"},
            "far": {"pos": "ADV", "lemma": "far"},
            "sea": {"pos": "NOUN", "lemma": "sea"},
            "draw": {"pos": "VERB", "lemma": "draw"},
            "left": {"pos": "VERB", "lemma": "leave"},
            "late": {"pos": "ADV", "lemma": "late"},
            "run": {"pos": "VERB", "lemma": "run"},
            "don't": {"pos": "PART", "lemma": "not"},
            "while": {"pos": "SCONJ", "lemma": "while"},
            "press": {"pos": "NOUN", "lemma": "press"},
            "close": {"pos": "ADV", "lemma": "close"},
            "night": {"pos": "NOUN", "lemma": "night"},
            "real": {"pos": "ADJ", "lemma": "real"},
            "life": {"pos": "NOUN", "lemma": "life"},
            "few": {"pos": "ADJ", "lemma": "few"},
            "stop": {"pos": "VERB", "lemma": "stop"},
            "open": {"pos": "ADJ", "lemma": "open"},
            "seem": {"pos": "VERB", "lemma": "seem"},
            "together": {"pos": "ADV", "lemma": "together"},
            "next": {"pos": "ADJ", "lemma": "next"},
            "white": {"pos": "ADJ", "lemma": "white"},
            "children": {"pos": "NOUN", "lemma": "child"},
            "begin": {"pos": "VERB", "lemma": "begin"},
            "got": {"pos": "VERB", "lemma": "get"},
            "walk": {"pos": "VERB", "lemma": "walk"},
            "example": {"pos": "NOUN", "lemma": "example"},
            "ease": {"pos": "NOUN", "lemma": "ease"},
            "paper": {"pos": "NOUN", "lemma": "paper"},
            "often": {"pos": "ADV", "lemma": "often"},
            "always": {"pos": "ADV", "lemma": "always"},
            "music": {"pos": "NOUN", "lemma": "music"},
            "those": {"pos": "PRON", "lemma": "those"},
            "both": {"pos": "PRON", "lemma": "both"},
            "mark": {"pos": "PROPN", "lemma": "mark"},
            "book": {"pos": "PROPN", "lemma": "book"},
            "letter": {"pos": "NOUN", "lemma": "letter"},
            "until": {"pos": "ADP", "lemma": "until"},
            "mile": {"pos": "NOUN", "lemma": "mile"},
            "river": {"pos": "NOUN", "lemma": "river"},
            "car": {"pos": "NOUN", "lemma": "car"},
            "feet": {"pos": "NOUN", "lemma": "foot"},
            "care": {"pos": "VERB", "lemma": "care"},
            "second": {"pos": "ADJ", "lemma": "second"},
            "group": {"pos": "NOUN", "lemma": "group"},
            "carry": {"pos": "VERB", "lemma": "carry"},
            "took": {"pos": "VERB", "lemma": "take"},
            "rain": {"pos": "NOUN", "lemma": "rain"},
            "eat": {"pos": "VERB", "lemma": "eat"},
            "room": {"pos": "NOUN", "lemma": "room"},
            "friend": {"pos": "NOUN", "lemma": "friend"},
            "began": {"pos": "VERB", "lemma": "begin"},
            "idea": {"pos": "NOUN", "lemma": "idea"},
            "fish": {"pos": "NOUN", "lemma": "fish"},
            "mountain": {"pos": "NOUN", "lemma": "mountain"},
            "north": {"pos": "NOUN", "lemma": "north"},
            "once": {"pos": "ADV", "lemma": "once"},
            "base": {"pos": "NOUN", "lemma": "base"},
            "hear": {"pos": "VERB", "lemma": "hear"},
            "horse": {"pos": "NOUN", "lemma": "horse"},
            "cut": {"pos": "VERB", "lemma": "cut"},
            "sure": {"pos": "ADJ", "lemma": "sure"},
            "watch": {"pos": "VERB", "lemma": "watch"},
            "color": {"pos": "NOUN", "lemma": "color"},
            "face": {"pos": "VERB", "lemma": "face"},
            "wood": {"pos": "NOUN", "lemma": "wood"},
            "main": {"pos": "ADJ", "lemma": "main"},
            "enough": {"pos": "ADV", "lemma": "enough"},
            "plain": {"pos": "ADV", "lemma": "plain"},
            "girl": {"pos": "NOUN", "lemma": "girl"},
            "usual": {"pos": "ADJ", "lemma": "usual"},
            "young": {"pos": "ADJ", "lemma": "young"},
            "ready": {"pos": "ADJ", "lemma": "ready"},
            "above": {"pos": "ADV", "lemma": "above"},
            "ever": {"pos": "ADV", "lemma": "ever"},
            "red": {"pos": "ADJ", "lemma": "red"},
            "Red": {"pos": "ADJ", "lemma": "red"},
            "list": {"pos": "NOUN", "lemma": "list"},
            "though": {"pos": "SCONJ", "lemma": "though"},
            "feel": {"pos": "VERB", "lemma": "feel"},
            "talk": {"pos": "VERB", "lemma": "talk"},
            "bird": {"pos": "NOUN", "lemma": "bird"},
            "soon": {"pos": "ADV", "lemma": "soon"},
            "body": {"pos": "NOUN", "lemma": "body"},
            "dog": {"pos": "NOUN", "lemma": "dog"},
            "dogs": {"pos": "NOUN", "lemma": "dog"},
            "dog's": {"pos": "PART", "lemma": "'s"},
            "family": {"pos": "NOUN", "lemma": "family"},
            "direct": {"pos": "ADJ", "lemma": "direct"},
            "pose": {"pos": "VERB", "lemma": "pose"},
            "leave": {"pos": "VERB", "lemma": "leave"},
            "song": {"pos": "NOUN", "lemma": "song"},
            "measure": {"pos": "NOUN", "lemma": "measure"},
            "state": {"pos": "NOUN", "lemma": "state"},
            "product": {"pos": "NOUN", "lemma": "product"},
            "black": {"pos": "ADJ", "lemma": "black"},
            "short": {"pos": "ADJ", "lemma": "short"},
            "numeral": {"pos": "ADJ", "lemma": "numeral"},
            "class": {"pos": "NOUN", "lemma": "class"},
            "wind": {"pos": "NOUN", "lemma": "wind"},
            "question": {"pos": "NOUN", "lemma": "question"},
            "happen": {"pos": "VERB", "lemma": "happen"},
            "complete": {"pos": "ADJ", "lemma": "complete"},
            "ship": {"pos": "NOUN", "lemma": "ship"},
            "area": {"pos": "NOUN", "lemma": "area"},
            "half": {"pos": "ADJ", "lemma": "half"},
            "rock": {"pos": "NOUN", "lemma": "rock"},
            "order": {"pos": "NOUN", "lemma": "order"},
            "fire": {"pos": "NOUN", "lemma": "fire"},
            "south": {"pos": "ADJ", "lemma": "south"},
            "problem": {"pos": "NOUN", "lemma": "problem"},
            "piece": {"pos": "NOUN", "lemma": "piece"},
            "told": {"pos": "VERB", "lemma": "tell"},
            "knew": {"pos": "VERB", "lemma": "know"},
            "pass": {"pos": "VERB", "lemma": "pass"},
            "farm": {"pos": "NOUN", "lemma": "farm"},
            "top": {"pos": "ADJ", "lemma": "top"},
            "whole": {"pos": "ADJ", "lemma": "whole"},
            "king": {"pos": "NOUN", "lemma": "king"},
            "size": {"pos": "NOUN", "lemma": "size"},
            "heard": {"pos": "VERB", "lemma": "hear"},
            "best": {"pos": "ADJ", "lemma": "well"},
            "hour": {"pos": "NOUN", "lemma": "hour"},
            "better": {"pos": "ADV", "lemma": "well"},
            "true": {"pos": "ADJ", "lemma": "true"},
            "during": {"pos": "ADP", "lemma": "during"},
            "hundred": {"pos": "NUM", "lemma": "hundred"},
            "am": {"pos": "AUX", "lemma": "be"},
            "remember": {"pos": "VERB", "lemma": "remember"},
            "step": {"pos": "VERB", "lemma": "step"},
            "early": {"pos": "ADV", "lemma": "early"},
            "hold": {"pos": "VERB", "lemma": "hold"},
            "west": {"pos": "PROPN", "lemma": "west"},
            "ground": {"pos": "NOUN", "lemma": "ground"},
            "interest": {"pos": "NOUN", "lemma": "interest"},
            "reach": {"pos": "VERB", "lemma": "reach"},
            "fast": {"pos": "ADJ", "lemma": "fast"},
            "five": {"pos": "NUM", "lemma": "five"},
            "sing": {"pos": "VERB", "lemma": "sing"},
            "sings": {"pos": "VERB", "lemma": "sing"},
            "listen": {"pos": "VERB", "lemma": "listen"},
            "six": {"pos": "NUM", "lemma": "six"},
            "table": {"pos": "NOUN", "lemma": "table"},
            "travel": {"pos": "NOUN", "lemma": "travel"},
            "less": {"pos": "ADV", "lemma": "less"},
            "morning": {"pos": "NOUN", "lemma": "morning"},
            "ten": {"pos": "NUM", "lemma": "ten"},
            "simple": {"pos": "ADJ", "lemma": "simple"},
            "several": {"pos": "ADJ", "lemma": "several"},
            "vowel": {"pos": "NOUN", "lemma": "vowel"},
            "toward": {"pos": "ADP", "lemma": "toward"},
            "war": {"pos": "NOUN", "lemma": "war"},
            "lay": {"pos": "VERB", "lemma": "lie"},
            "against": {"pos": "ADP", "lemma": "against"},
            "pattern": {"pos": "NOUN", "lemma": "pattern"},
            "slow": {"pos": "ADJ", "lemma": "slow"},
            "center": {"pos": "PROPN", "lemma": "center"},
            "love": {"pos": "NOUN", "lemma": "love"},
            "person": {"pos": "NOUN", "lemma": "person"},
            "money": {"pos": "NOUN", "lemma": "money"},
            "serve": {"pos": "VERB", "lemma": "serve"},
            "appear": {"pos": "VERB", "lemma": "appear"},
            "road": {"pos": "NOUN", "lemma": "road"},
            "map": {"pos": "NOUN", "lemma": "map"},
            "science": {"pos": "NOUN", "lemma": "science"},
            "rule": {"pos": "NOUN", "lemma": "rule"},
            "govern": {"pos": "VERB", "lemma": "govern"},
            "pull": {"pos": "VERB", "lemma": "pull"},
            "cold": {"pos": "ADJ", "lemma": "cold"},
            "notice": {"pos": "VERB", "lemma": "notice"},
            "voice": {"pos": "NOUN", "lemma": "voice"},
            "fall": {"pos": "NOUN", "lemma": "fall"},
            "power": {"pos": "NOUN", "lemma": "power"},
            "town": {"pos": "NOUN", "lemma": "town"},
            "fine": {"pos": "ADJ", "lemma": "fine"},
            "certain": {"pos": "ADJ", "lemma": "certain"},
            "fly": {"pos": "VERB", "lemma": "fly"},
            "unit": {"pos": "NOUN", "lemma": "unit"},
            "lead": {"pos": "VERB", "lemma": "lead"},
            "cry": {"pos": "VERB", "lemma": "cry"},
            "dark": {"pos": "ADJ", "lemma": "dark"},
            "machine": {"pos": "NOUN", "lemma": "machine"},
            "note": {"pos": "NOUN", "lemma": "note"},
            "wait": {"pos": "VERB", "lemma": "wait"},
            "plan": {"pos": "NOUN", "lemma": "plan"},
            "figure": {"pos": "NOUN", "lemma": "figure"},
            "star": {"pos": "PROPN", "lemma": "star"},
            "box": {"pos": "PROPN", "lemma": "box"},
            "noun": {"pos": "PROPN", "lemma": "noun"},
            "field": {"pos": "NOUN", "lemma": "field"},
            "rest": {"pos": "VERB", "lemma": "rest"},
            "correct": {"pos": "ADJ", "lemma": "correct"},
            "able": {"pos": "ADJ", "lemma": "able"},
            "pound": {"pos": "NOUN", "lemma": "pound"},
            "done": {"pos": "VERB", "lemma": "do"},
            "beauty": {"pos": "NOUN", "lemma": "beauty"},
            "drive": {"pos": "VERB", "lemma": "drive"},
            "stood": {"pos": "VERB", "lemma": "stand"},
            "contain": {"pos": "VERB", "lemma": "contain"},
            "front": {"pos": "NOUN", "lemma": "front"},
            "teach": {"pos": "VERB", "lemma": "teach"},
            "week": {"pos": "NOUN", "lemma": "week"},
            "final": {"pos": "ADJ", "lemma": "final"},
            "gave": {"pos": "VERB", "lemma": "give"},
            "green": {"pos": "ADJ", "lemma": "green"},
            "oh": {"pos": "INTJ", "lemma": "oh"},
            "quick": {"pos": "ADJ", "lemma": "quick"},
            "develop": {"pos": "VERB", "lemma": "develop"},
            "sleep": {"pos": "NOUN", "lemma": "sleep"},
            "warm": {"pos": "ADJ", "lemma": "warm"},
            "free": {"pos": "ADJ", "lemma": "free"},
            "minute": {"pos": "NOUN", "lemma": "minute"},
            "strong": {"pos": "ADJ", "lemma": "strong"},
            "special": {"pos": "ADJ", "lemma": "special"},
            "mind": {"pos": "VERB", "lemma": "mind"},
            "behind": {"pos": "ADV", "lemma": "behind"},
            "clear": {"pos": "ADV", "lemma": "clear"},
            "tail": {"pos": "NOUN", "lemma": "tail"},
            "produce": {"pos": "VERB", "lemma": "produce"},
            "fact": {"pos": "NOUN", "lemma": "fact"},
            "street": {"pos": "NOUN", "lemma": "street"},
            "inch": {"pos": "NOUN", "lemma": "inch"},
            "lot": {"pos": "NOUN", "lemma": "lot"},
            "nothing": {"pos": "PRON", "lemma": "nothing"},
            "course": {"pos": "NOUN", "lemma": "course"},
            "stay": {"pos": "VERB", "lemma": "stay"},
            "wheel": {"pos": "NOUN", "lemma": "wheel"},
            "full": {"pos": "ADJ", "lemma": "full"},
            "force": {"pos": "NOUN", "lemma": "force"},
            "blue": {"pos": "ADJ", "lemma": "blue"},
            "object": {"pos": "VERB", "lemma": "object"},
            "decide": {"pos": "VERB", "lemma": "decide"},
            "surface": {"pos": "NOUN", "lemma": "surface"},
            "deep": {"pos": "ADJ", "lemma": "deep"},
            "moon": {"pos": "NOUN", "lemma": "moon"},
            "island": {"pos": "NOUN", "lemma": "island"},
            "foot": {"pos": "NOUN", "lemma": "foot"},
            "yet": {"pos": "ADV", "lemma": "yet"},
            "busy": {"pos": "ADJ", "lemma": "busy"},
            "test": {"pos": "NOUN", "lemma": "test"},
            "record": {"pos": "PROPN", "lemma": "record"},
            "boat": {"pos": "NOUN", "lemma": "boat"},
            "common": {"pos": "ADJ", "lemma": "common"},
            "gold": {"pos": "ADJ", "lemma": "gold"},
            "possible": {"pos": "ADJ", "lemma": "possible"},
            "plane": {"pos": "NOUN", "lemma": "plane"},
            "age": {"pos": "NOUN", "lemma": "age"},
            "dry": {"pos": "ADJ", "lemma": "dry"},
            "wonder": {"pos": "NOUN", "lemma": "wonder"},
            "laugh": {"pos": "VERB", "lemma": "laugh"},
            "thousand": {"pos": "NUM", "lemma": "thousand"},
            "ago": {"pos": "ADV", "lemma": "ago"},
            "ran": {"pos": "VERB", "lemma": "run"},
            "check": {"pos": "VERB", "lemma": "check"},
            "game": {"pos": "NOUN", "lemma": "game"},
            "shape": {"pos": "NOUN", "lemma": "shape"},
            "yes": {"pos": "INTJ", "lemma": "yes"},
            "cool": {"pos": "ADJ", "lemma": "cool"},
            "miss": {"pos": "VERB", "lemma": "miss"},
            "brought": {"pos": "VERB", "lemma": "bring"},
            "heat": {"pos": "NOUN", "lemma": "heat"},
            "snow": {"pos": "NOUN", "lemma": "snow"},
            "bed": {"pos": "NOUN", "lemma": "bed"},
            "bring": {"pos": "VERB", "lemma": "bring"},
            "sit": {"pos": "VERB", "lemma": "sit"},
            "perhaps": {"pos": "ADV", "lemma": "perhaps"},
            "fill": {"pos": "VERB", "lemma": "fill"},
            "east": {"pos": "NOUN", "lemma": "east"},
            "weight": {"pos": "NOUN", "lemma": "weight"},
            "language": {"pos": "NOUN", "lemma": "language"},
            "among": {"pos": "ADP", "lemma": "among"},
            "adult": {"pos": "NOUN", "lemma": "adult"},
            "adults": {"pos": "NOUN", "lemma": "adult"},
            "yellow": {"pos": "ADJ", "lemma": "yellow"},
            "orange": {"pos": "ADJ", "lemma": "orange"},
            "purple": {"pos": "ADJ", "lemma": "purple"},
            "brown": {"pos": "ADJ", "lemma": "brown"},
            "pink": {"pos": "ADJ", "lemma": "pink"},
            "gray": {"pos": "ADJ", "lemma": "gray"},
            "grey": {"pos": "ADJ", "lemma": "grey"},
            "silver": {"pos": "ADJ", "lemma": "silver"},
            "tiny": {"pos": "ADJ", "lemma": "tiny"},
            "huge": {"pos": "ADJ", "lemma": "huge"},
            "tall": {"pos": "ADJ", "lemma": "tall"},
            "wide": {"pos": "ADJ", "lemma": "wide"},
            "narrow": {"pos": "ADJ", "lemma": "narrow"},
            "thick": {"pos": "ADJ", "lemma": "thick"},
            "thin": {"pos": "ADJ", "lemma": "thin"},
            "ancient": {"pos": "ADJ", "lemma": "ancient"},
            "modern": {"pos": "ADJ", "lemma": "modern"},
            "fresh": {"pos": "ADJ", "lemma": "fresh"},
        }
        self.NAMES = {
            "ada",
            "andrew",
            "bobbie",
            "cason",
            "david",
            "farzana",
            "frank",
            "hannah",
            "ida",
            "irene",
            "jim",
            "jose",
            "keith",
            "laura",
            "lucy",
            "meredith",
            "nick",
            "serena",
            "yan",
            "yeeling",
        }
        self.DIST = {"mile", "foot", "feet", "meter", "kilometer", "inch", "yard"}
        self.TIME_PATTERN = re.compile(r"\d{1,2}:\d{2}(AM|PM)?", re.IGNORECASE)
        self.TIME_MARKERS = {"this", "last", "next", "every", "on"}
        self.TIME_WORDS = {
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
            "today",
            "tomorrow",
            "yesterday",
            "now",
            "soon",
            "later",
            "recently",
            "morning",
            "afternoon",
            "evening",
            "night",
            "noon",
            "midnight",
            "spring",
            "summer",
            "fall",
            "autumn",
            "winter",
            "week",
            "month",
            "year",
            "day",
            "hour",
            "minute",
            "second",
        }
        self.CLAUSE_MARKERS = {"when", "while", "if", "because", "although", "unless"}
        self.W_WORDS = {"who", "whom", "what", "where", "when", "why", "how"}
        self.W_MODIFIERS = {
            "time": "WHEN",
            "far": "HOW_FAR",
            "long": "HOW_LONG",
            "many": "HOW_QUANTITY",
            "much": "HOW_QUANTITY",
            "color": "WHAT_COLOR",
            "often": "HOW_FREQUENCY",
        }
        self.W_MOVEMENT = {"get", "go", "travel", "arrive", "walk", "drive", "come"}
        self.W_BASE = {
            "who": "WHO_AGENT",
            "whom": "WHO_AGENT",
            "what": "WHAT_OBJECT",
            "where": "WHERE",
            "when": "WHEN",
            "why": "WHY",
            "how": "HOW",
            "color": "WHAT_COLOR",
            "long": "HOW_LONG",
            "far": "HOW_FAR",
            "old": "HOW_OLD",
        }
        self.DIRECTIONS = {"east", "west", "north", "south"}
        self.NUM_WORDS = {
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
            "twenty",
            "thirty",
            "forty",
            "fifty",
            "sixty",
            "seventy",
            "eighty",
            "ninety",
            "hundred",
            "thousand",
            "million",
            "billion",
        }

    def _is_hyphenated_number(self, word: str) -> bool:
        """Check if word is a hyphenated number like twenty-one or thirty-two."""
        parts = word.lower().split("-")
        return len(parts) == 2 and all(p in self.NUM_WORDS for p in parts)

    def tokenize(self, text: str) -> list:
        """Tokenize the sentence returning a list of the tokens.

        Args:
            text: The text to be tokenized.

        References:
            - Ch.2 Words and Tokens:
                https://web.stanford.edu/~jurafsky/slp3/2.pdf
            - Tokenization in NLP:
                https://www.datacamp.com/blog/what-is-tokenization
        """
        text = text.rstrip(".?!")
        tokens = text.split()
        return tokens

    def get_pos(self, word: str, prev_word: str = None, next_word: str = None) -> str:
        """Get part-of-speech tag for a word.

        Args:
            word: The given word that needs to be tagged.
            prev_word: The previous word in the sentence.
            next_word: The next word in the sentence.

        References:
            - Part of Speech Tagging:
                https://campus.datacamp.com/courses/feature-engineering-for-nlp-in-python/text-preprocessing-pos-tagging-and-ner?ex=8
            - Penn Treebank POS Tags:
                https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
            - Ch.17 Sequence Labeling for POS and Named Entities:
                https://web.stanford.edu/~jurafsky/slp3/17.pdf
        """
        word_lower = word.lower()
        # Check known words first
        if word_lower in self.NAMES:
            return "PROPN"

        # handle ambigour 'to' position.
        elif word_lower == "to":
            if next_word:
                next_data = self.WORD_DATA.get(next_word, None)
                if "pos" in next_data and next_data.get("pos") == "VERB":
                    return "PART"
                else:
                    return "ADP"
            return "ADP"

        elif self.TIME_PATTERN.match(word) or word_lower in self.TIME_WORDS:
            return "TIME"

        # Infer from context: article/adjective usually precedes noun
        elif prev_word and prev_word.lower() in {
            "a",
            "an",
            "the",
            "this",
            "that",
            "some",
        }:
            if word_lower in self.WORD_DATA:
                pos = self.WORD_DATA[word_lower]["pos"]
                if pos == "VERB":
                    return "NOUN"  # "a play" → NOUN
                return pos  # "the best" → stays ADJ
            return "NOUN"

        elif self._is_hyphenated_number(word_lower):
            return "NUM"

        elif word_lower in self.WORD_DATA:
            return self.WORD_DATA[word_lower]["pos"]

        # Infer from morphology (suffix patterns)
        elif word_lower.endswith(("tion", "ness", "ment", "ity", "er", "or")):
            return "NOUN"
        elif word_lower.endswith(("ly",)):
            return "ADV"
        elif word_lower.endswith(("ed", "ing")) and len(word_lower) > 4:
            return "VERB"

        # Capitalized mid-sentence = likely proper noun
        elif word[0].isupper():
            return "PROPN"
        else:
            return "NOUN"  # Default guess: noun (most common for unknowns)

    def tag_tokens(self, tokens: list[str]) -> list[tuple[str, str]]:
        """Tag all tokens with POS and return the list of tuples.
        Args:
            tokens: The list of words that make up the sentence.
        """
        tagged_tokens = []
        if len(tokens) > 1:
            for i, token in enumerate(tokens):
                prev_token = tokens[i - 1] if i > 0 else None
                next_token = tokens[i + 1] if i < len(tokens) - 1 else None
                tagged_tokens.append(
                    (
                        token,
                        self.get_pos(
                            word=token, prev_word=prev_token, next_word=next_token
                        ),
                    )
                )
        return tagged_tokens

    def get_frame_from_tagged_tokens(
        self, tagged_tokens: list[tuple[str, str]]
    ) -> dict:
        """Extract a sentence frame from tagged tokens.

        Args:
            tagged_tokens: The list of tuples containing tokens and their part of speach (POS) tags.

        References:
            - Frame extraction
                - Ch.20 Computational Lexical Semantics (Jurafsky & Martin, 2009)
                - Ch.21 Semantic Role Labeling and Argument Structure (Jurafsky & Martin, 2026)
            - Semantic roles
                - Ch.19 Lexical Semantics (Jurafsky & Martin, 2009)
            - Semantic Roles in NLP
                https://www.geeksforgeeks.org/nlp/semantic-roles-in-nlp/
        """
        frame = {
            "agents": [],  # Fillmore's AGENT case
            "action": None,  # The predicate/verb
            "objects": [],  # PATIENT/THEME case
            "recipients": [],  # RECIPIENT/GOAL case
            "locations": [],  # LOCATION case
            "times": [],  # TEMPORAL adjunct
            "instruments": [],  # INSTRUMENT case
            "companions": [],  # COMITATIVE case
            "modifiers": {},  # Adjective-noun links
            "distances": [],  # Measure phrases
            "quantities": [],
        }
        verb_idx = None
        for i, (word, pos) in enumerate(tagged_tokens):
            if pos == "VERB":
                verb_idx = i
                frame["action"] = word
                break
        # fallback to AUX
        if verb_idx is None:
            for i, (word, pos) in enumerate(tagged_tokens):
                if pos == "AUX":
                    verb_idx = i
                    frame["action"] = word
                    break

        if verb_idx is None:
            return frame

        # After verb → other roles based on prepositions
        current_prep = None
        current_adj = None
        current_num = None

        # Before verb → agents with modifiers
        prev_word = None
        for word, pos in tagged_tokens[:verb_idx]:
            if pos == "ADJ":
                current_adj = word
            elif pos == "TIME":
                if prev_word and prev_word.lower() in self.TIME_MARKERS:
                    frame["times"].append(f"{prev_word} {word}")
                else:
                    frame["times"].append(word)
            elif pos in ["PROPN", "NOUN"]:
                frame["agents"].append(word)
                if current_adj:
                    frame["modifiers"][word] = current_adj
                    current_adj = None
            prev_word = word

        prev_word = None
        for word, pos in tagged_tokens[verb_idx + 1 :]:
            # stop if the word is a clause marker.
            if word.lower() in self.CLAUSE_MARKERS:
                break

            if pos == "NUM":
                if prev_word and prev_word.lower() in {"a", "an"}:
                    current_num = f"{prev_word} {word}"
                # handle for non hyphenated compound numbers.
                elif current_num:
                    current_num = f"{current_num} {word}"
                else:
                    current_num = word
            elif pos == "ADP":
                current_prep = word.lower()
            elif pos == "ADJ":
                current_adj = word
            elif pos == "DET":
                prev_word = word
                continue

            elif pos == "TIME":
                if prev_word and prev_word.lower() in self.TIME_MARKERS:
                    frame["times"].append(f"{prev_word} {word}")
                else:
                    frame["times"].append(word)

            elif pos in ["NOUN", "PROPN"]:
                # Handle directions as locations
                if word.lower() in self.DIRECTIONS and current_prep is None:
                    frame["locations"].append(word)
                    prev_word = word
                    continue

                # Handle measure/quantity phrases like "3 feet" or "2 miles" or "a thousand dogs"
                if current_num and word.lower() in self.DIST:
                    frame["distances"].append(f"{current_num} {word}")
                    current_num = None
                    current_prep = None
                    continue
                elif current_num:
                    frame["quantities"].append(f"{current_num} {word}")

                # Assign role based on preposition
                if current_prep is None:
                    frame["objects"].append(word)
                elif current_prep == "to":
                    if pos == "PROPN":
                        frame["recipients"].append(word)
                    else:
                        frame["locations"].append(word)
                elif current_prep == "of":
                    frame["objects"].append(word)
                elif current_prep in ["at", "in", "on", "from"]:
                    if current_prep == "on" and word.lower() in self.TIME_WORDS:
                        frame["times"].append(word)
                    else:
                        frame["locations"].append(word)
                elif current_prep == "with":
                    if pos == "PROPN":
                        frame["companions"].append(word)
                    else:
                        frame["instruments"].append(word)

                # Track adjective modifiers
                if current_adj:
                    frame["modifiers"][word] = current_adj
                    current_adj = None

                current_prep = None

            prev_word = word

            # Handle predicate adjectives: "The water is blue"
            if current_adj and frame["agents"]:
                frame["modifiers"][frame["agents"][-1]] = current_adj
        return frame

    def classify_question(self, question: str) -> str:
        """Classify the type of question.

        Args:
            question: The given question to classify.

        References:
            - Question classification and answer type taxonomy
                - Ch.23 Question Answering (Jurafsky & Martin, 2009)
            - Answering Questions from text
                https://campus.datacamp.com/courses/natural-language-processing-nlp-in-python/token-classification-and-text-generation?ex=5
        """
        qst_lower = question.lower()
        tokens = self.tokenize(qst_lower)

        wh_idx = None
        wh_word = None

        # Step 1: Tokenize the question.
        for i, tok in enumerate(tokens):
            if tok in self.W_WORDS:
                wh_idx = i
                wh_word = tok
                break

        if wh_word is None:
            return "UNKNOWN"

        # Step 2: Get surrounding tokens
        prev_token = tokens[wh_idx - 1] if wh_idx > 0 else None
        next_token = tokens[wh_idx + 1] if wh_idx + 1 < len(tokens) else None
        last_token = tokens[-1]

        # Step 3: classify the question based upon the rules.

        # Rule 1: PREP + WH (e.g., "with whom", "at what time", "who is blue")
        if prev_token == "with":
            if wh_word in {"who", "whom"}:
                return "WHO_WITH"
            else:
                return "WITH_WHAT"
        if prev_token == "to":
            return "WHO_RECIPIENT"

        # Rule 2: WH + MODIFIER (e.g., "how far", "what time")
        if next_token in self.W_MODIFIERS:
            return self.W_MODIFIERS[next_token]

        # Rule 3: WHO + WITH anywhere (e.g., "Who does Lucy go with?")
        if wh_word in {"who", "whom"} and "with" in tokens:
            return "WHO_WITH"

        # Rule 4: WHO + trailing TO (e.g., "Who did Ada bring the note to?")
        if wh_word in {"who", "whom"} and last_token == "to":
            return "WHO_RECIPIENT"

        # Rule 5: WHAT + is subject
        if wh_word == "what" and "is" in tokens:
            # Check if asking about subject
            return "WHAT_SUBJECT"

        # Rule 6: HOW + movement verb
        if wh_word == "how":
            if self.W_MOVEMENT & set(tokens):
                return "HOW_METHOD"

        # Rule 7: Base WH-word (fallback)
        return self.W_BASE.get(wh_word, "UNKNOWN")

    def solve(self, sentence: str, question: str) -> str:
        """Answer the question based on the given sentence.
        Args:
            sentence: The sentence used to answer the question.
            question: The question that pplies to the sentence.
        """
        tokens = self.tokenize(sentence)
        tagged_tokens = self.tag_tokens(tokens)
        print(f"tagged_tokens: {tagged_tokens}")
        frame = self.get_frame_from_tagged_tokens(tagged_tokens)
        print(f"frame: {frame}")
        q_type = self.classify_question(question)
        print(f"q_type: {q_type}")

        q_type_parts = q_type.split("_")
        # do the 5w's in question types and then get subtypes from that.

        # WHO
        if q_type_parts[0] == "WHO":
            if len(q_type_parts) > 1:
                if q_type_parts[1] == "AGENT":
                    if frame["agents"]:
                        return frame["agents"][-1]
                elif q_type_parts[1] == "RECIPIENT":
                    if frame["recipients"]:
                        return frame["recipients"][-1]
                elif q_type_parts[1] == "WITH":
                    for name in frame["agents"]:
                        if name.lower() not in question.lower():
                            return name
                    if frame["agents"]:
                        return frame["agents"][0]
                # Fallback for existential sentences ("There are men")
                if frame["objects"]:
                    return frame["objects"][0]
                # Fall back to AGENT
                if frame["agents"]:
                    return frame["agents"][0]
        # WHAT
        elif q_type_parts[0] == "WHAT":
            if len(q_type_parts) > 1:
                if q_type_parts[1] == "OBJECT":
                    if frame["objects"]:
                        return frame["objects"][-1]
                    if frame["agents"]:
                        return frame["agents"][-1]
                elif q_type_parts[1] == "SUBJECT":
                    return frame["agents"][-1]
                elif q_type_parts[1] in ["COLOR"]:
                    q_tokens = self.tokenize(question)
                    for token in q_tokens:
                        if token in frame["modifiers"]:
                            return frame["modifiers"][token]
                elif q_type_parts[1] == "MODIFIER":
                    if token in frame["modifiers"]:
                        return token

        # WHEN
        elif q_type_parts[0] == "WHEN":
            if frame["times"]:
                # Prefer numeric time (8:00AM) over word time
                for t in frame["times"]:
                    if self.TIME_PATTERN.match(t):
                        return t
                return frame["times"][-1]
        # WHERE
        elif q_type_parts[0] == "WHERE":
            if frame["locations"]:
                return frame["locations"][-1]
        # WHY
        elif q_type_parts[0] == "WHY":
            pass

        # HOW
        elif q_type_parts[0] == "HOW":
            if len(q_type_parts) > 1:
                if q_type_parts[1] == "METHOD":
                    if frame["action"]:
                        return frame["action"]
                elif q_type_parts[1] == "FAR":
                    if frame["distances"]:
                        return frame["distances"][-1]
                elif q_type_parts[1] in ["LONG", "OLD"]:
                    q_tokens = self.tokenize(question)
                    for token in q_tokens:
                        if token in frame["modifiers"]:
                            return frame["modifiers"][token]
                elif q_type_parts[1] == "QUANTITY":
                    if frame["quantities"]:
                        return frame["quantities"][0]

        return ""
