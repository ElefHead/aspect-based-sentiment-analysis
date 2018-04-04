from spacy.vectors import Vectors
from spacy.tokenizer import Tokenizer
from spacy import load

number_batch = './data/numberbatch-en.txt'

nlp = load("en")
tokenizer = Tokenizer(nlp.vocab)


# Remove Stopwords
def remove_stopwords(sentence) :
    return " ".join([str(token) for token in tokenizer(sentence.replace('[comma]', '').replace(".","").lower()) if not token.is_stop and not token.is_punct and not token.is_digit and token.is_alpha])


text = "I was disappointed when I realized that the keyboard doesn't light up on this model."

print(remove_stopwords(text))
