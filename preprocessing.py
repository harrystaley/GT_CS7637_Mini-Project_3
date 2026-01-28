import spacy
from pathlib import Path

KNOWN_NAMES = {
    "ada", "andrew", "bobbie", "cason", "david", "farzana", "frank",
    "hannah", "ida", "irene", "jim", "jose", "keith", "laura",
    "lucy", "meredith", "nick", "serena", "yan", "yeeling"
}

nlp = spacy.load("en_core_web_sm")

def preprocess_word_list(word_lst: str = "mostcommon.txt"):
    """Preprocess the wordlist and return the attributes of the words."""
    common = Path(word_lst)
    if not common.exists():
        raise FileNotFoundError("mostcommong.txt does not exist.")

    with common.open(mode='r') as f:
        words = [line.strip() for line in f if line.strip()]

    word_info = {}
    for word in words:
        word_lower = word.lower()
        if word_lower in KNOWN_NAMES:
            word_info[word] = {
                "pos": "PROPN",      # POS (Part of Speech): tells you if a word is a NOUN, VERB, ADJ, etc.
                "lemma": word_lower  # Lemma: base form of the word ("brought" → "bring")
            }
            continue

        doc = nlp(word)
        for token in doc:
            word_info[word] = {
                "pos": token.pos_,      # POS (Part of Speech): tells you if a word is a NOUN, VERB, ADJ, etc.
                "lemma": token.lemma_   # Lemma: base form of the word ("brought" → "bring")
            }
    print(word_info)

if __name__ == "__main__":
    preprocess_word_list()