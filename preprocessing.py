import spacy
from pathlib import Path

nlp = spacy.load("en_core_web_sm")

def preprocess_word_list(word_lst: str = "mostcommon.txt"):
    """Preprocess the wordlist and return the attributes of the words."""
    common = Path(word_lst)
    if not common.exists():
        raise FileNotFoundError("mostcommong.txt does not exist.")

    with common.open(mode='r') as f:
        words = [line.strip().lower() for line in f if line.strip()]

    word_info = {}
    for word in words:
        doc = nlp(word)
        for token in doc:
            word_info[word] = {
                "pos": token.pos_,
                "lemma": token.lemma_
            }
    print(word_info)

if __name__ == "__main__":
    preprocess_word_list()