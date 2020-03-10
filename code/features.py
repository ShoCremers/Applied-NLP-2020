import nltk
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
            'timeout': 1000,
        })
    return result["sentences"][0]["sentimentValue"]


if __name__ == '__main__':
    #nlp = spacy.load("en_core_web_lg")
    nlp = spacy.load("en_core_web_sm")
    sentencex = "A Couple Did A Stunning Photo Shoot With Their Baby After Learning She Had An Inoperable Brain Tumor. People Don't Know Who David Bowie Is And We're Here To Clear That Up. 24 Times Ruby Rose And Phoebe Dahl Defined Relationship Goals In 2015. G.M. Lowers 2009 Outlook for All U.S. Auto Sales. Your dog is not listening to me"
    sentence0 = "A Couple Did A Stunning Photo Shoot With Their Baby After Learning She Had An Inoperable Brain Tumor"
    sentence1 = "People Don't Know Who David Bowie Is And We're Here To Clear That Up"
    sentence2 = "24 Times Ruby Rose And Phoebe Dahl Defined Relationship Goals In 2015"
    sentence3 = "G.M. Lowers 2009 Outlook for All U.S. Auto Sales"
    sentence4 = "Your dog is not listening to me"

    doc = nlp(sentence4)
    # results = sentiment_score(sentence0)

    print(sentiment_score(sentence0))
    # for sentence in results["sentences"]:
    #     print("{}: {}".format(
    #         " ".join([t["word"] for t in sentence["tokens"]]),
    #         sentence["sentimentValue"]
    #         )
    #     )
    #print(doc)
    #print(number_of_words(doc))
    #print(avg_length_of_word(doc))
    #print(stopword_percentage(doc))
    #print(has_number(doc))
    #print(has_determiner(doc))
    for token in doc:
        print(token.text, token.tag_)

    print(has_superlative(doc))

    sentence2 = sentence2.lower()
    doc = nlp(sentence2)