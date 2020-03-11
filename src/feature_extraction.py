import pandas as pd
import features
import spacy


if __name__ == '__main__':

    nlp = spacy.load("en_core_web_lg")

    df_cb = pd.read_csv("dataset/random_clickbait.csv")

    df_ncb = pd.read_csv("dataset/random_non_clickbait.csv")

    df = pd.concat([df_cb, df_ncb], ignore_index=True)
    tot_headlines = df.shape[0]

    columns = ['Headlines', 'class', '#words','word length', 'stopword%', 'has number', 'has determiner', 'has pronoun',
               'has comparative', 'has superlative', 'has sup or comp', 'sentiment value']

    # statistics for clickbait data
    df_features = pd.DataFrame(index=range(tot_headlines), columns=columns)

    for index, row in df.iterrows():
        sentence = row["Headlines"].lower()
        doc = nlp(sentence)

        df_features["Headlines"][index] = sentence
        df_features["class"][index] = row["label"]
        df_features["#words"][index] = features.number_of_words(doc)
        df_features["word length"][index] = features.avg_length_of_word(doc)
        df_features["stopword%"][index] = features.stopword_percentage(doc)
        df_features["has number"][index] = int(features.has_number(doc))
        df_features["has determiner"][index] = int(features.has_determiner(doc))
        df_features["has pronoun"][index] = int(features.has_pronoun(doc))
        df_features["has comparative"][index] = int(features.has_comparative(doc))
        df_features["has superlative"][index] = int(features.has_superlative(doc))
        df_features["has sup or comp"][index] = int(features.has_comparative(doc) or features.has_superlative(doc))
        df_features["sentiment value"][index] = features.sentiment_score(sentence)

        print(index / tot_headlines, end="\r", flush=True)


    print(df_features)
    df_features.to_csv('dataset/features.csv', index=False)