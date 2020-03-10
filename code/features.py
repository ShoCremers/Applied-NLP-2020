import spacy
from pycorenlp import StanfordCoreNLP

# clickbaits tend to have longer headlines
def number_of_words(sentence):
    return len(sentence)

# Clickbaits tend to have smaller words; simpler words (short words more common in everyday english),
# and more common for word shortening (e.g. "Don't" instead of "do not")
def avg_length_of_word(sentence):
    total_len = 0
    for token in sentence:
        total_len += len(token)
    return total_len / len(sentence)

# non-clickbaits use more descriptive words
def stopword_percentage(sentence):
    count_sw = 0
    for token in sentence:
        count_sw += token.is_stop
    return count_sw / len(sentence)

# clickbait title uses number more often to list. non-clickbaits also include numbers but much rare and
# more meaningful (e.g. year, number of people dead)
# not a feature in the original paper
def has_number(sentence):
    for token in sentence:
        if token.pos_ == 'NUM':
            return True
    return False

def has_determiner(sentence):
    for token in sentence:
        if token.pos_ == 'DET':
            return True
    return False

def has_pronoun(sentence):
    for token in sentence:
        if token.pos_ == 'PRON':
            return True
    return False

def has_comparative(sentence):
    for token in sentence:
        if token.tag_ == 'JJR' or token.tag_ == 'RBR':
            return True
    return False

def has_superlative(sentence):
    for token in sentence:
        if token.tag_ == 'JJS' or token.tag_ == 'RBS':
            return True
    return False

def sentiment_score(sentence):
    snlp = StanfordCoreNLP('http://localhost:9000')
    result = snlp.annotate(sentence,
        properties={
            'annotators': 'sentiment',
            'outputFormat': 'json',
            'timeout': 10000,
        })
    return result["sentences"][0]["sentimentValue"]
