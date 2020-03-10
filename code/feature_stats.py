import pandas as pd
import features
import spacy


if __name__ == '__main__':

    nlp = spacy.load("en_core_web_lg")

    # clickbait variables
    df_cb = pd.read_csv("dataset/random_clickbait.csv")
    tot_headlines_cb = df_cb.shape[0]
    tot_words_cb = 0
    tot_word_length_cb = 0
    stop_word_cb = 0
    has_number_cb = 0
    has_determiner_cb = 0
    has_pronoun_cb = 0
    comparative_cb = 0
    superlative_cb = 0
    both_cb = 0

    # non clickbait variables
    df_ncb = pd.read_csv("dataset/random_non_clickbait.csv")
    tot_headlines_ncb = df_ncb.shape[0]
    tot_words_ncb = 0
    tot_word_length_ncb = 0
    stop_word_ncb = 0
    has_number_ncb = 0
    has_determiner_ncb = 0
    has_pronoun_ncb = 0
    comparative_ncb = 0
    superlative_ncb = 0
    both_ncb = 0


    columns = ['Headlines', '#words','word length', 'stopword%', 'has number', 'has determiner', 'has pronoun',
               'has comparative', 'has superlative', 'has sup or comp']

    # statistics for clickbait data
    df_cb_features = pd.DataFrame(index=range(tot_headlines_cb), columns=columns)

    for index, row in df_cb.iterrows():
        sentence = row["Headlines"].lower()
        doc = nlp(sentence)

        # tot_words_cb += features.number_of_words(doc)
        # tot_word_length_cb += features.avg_length_of_word(doc)
        # stop_word_cb += features.stopword_percentage(doc)
        # has_number_cb += features.has_number(doc)
        # has_determiner_cb += features.has_determiner(doc)
        # has_pronoun_cb += features.has_pronoun(doc)

        df_cb_features["Headlines"][index] = sentence
        df_cb_features["#words"][index] = features.number_of_words(doc)
        df_cb_features["word length"][index] = features.avg_length_of_word(doc)
        df_cb_features["stopword%"][index] = features.stopword_percentage(doc)
        df_cb_features["has number"][index] = int(features.has_number(doc))
        df_cb_features["has determiner"][index] = int(features.has_determiner(doc))
        df_cb_features["has pronoun"][index] = int(features.has_pronoun(doc))
        df_cb_features["has comparative"][index] = int(features.has_comparative(doc))
        df_cb_features["has superlative"][index] = int(features.has_superlative(doc))
        df_cb_features["has sup or comp"][index] = int(features.has_comparative(doc) or features.has_superlative(doc))

        superlative_cb += int(features.has_superlative(doc))
        comparative_cb += int(features.has_comparative(doc))
        both_cb += int(features.has_comparative(doc) or features.has_superlative(doc))

    print(df_cb_features)

    df_cb_features.to_csv('dataset/cb_features.csv', index=False)

    # avg_words_cb = tot_words_cb / tot_headlines_cb
    # avg_word_length_cb = tot_word_length_cb / tot_headlines_cb
    # avg_stop_word_percentage_cb = stop_word_cb / tot_headlines_cb
    # avg_number_cb = has_number_cb / tot_headlines_cb
    # avg_determiner_cb = has_determiner_cb / tot_headlines_cb
    # avg_pronoun_cb = has_pronoun_cb / tot_headlines_cb



    # statistics for non clickbait data

    # for index, row in df_ncb.iterrows():
    #     doc = nlp(row["Headlines"])
    #     tot_words_ncb += features.number_of_words(doc)
    #     tot_word_length_ncb += features.avg_length_of_word(doc)
    #     stop_word_ncb += features.stopword_percentage(doc)
    #     has_number_ncb += features.has_number(doc)
    #     has_determiner_ncb += features.has_determiner(doc)
    #     has_pronoun_ncb += features.has_pronoun(doc)
    #
    # avg_words_ncb = tot_words_ncb / tot_headlines_ncb
    # avg_word_length_ncb = tot_word_length_ncb / tot_headlines_ncb
    # avg_stop_word_percentage_ncb = stop_word_ncb / tot_headlines_ncb
    # avg_number_ncb = has_number_ncb / tot_headlines_ncb
    # avg_determiner_ncb = has_determiner_ncb / tot_headlines_ncb
    # avg_pronoun_ncb = has_pronoun_ncb / tot_headlines_ncb
    #
    # print("number of word ->", "cb:", avg_words_cb, "ncb:", avg_words_ncb)
    # print("length of words ->", "cb:", avg_word_length_cb, "ncb:", avg_word_length_ncb)
    # print("stop word percentage ->", "cb:", avg_stop_word_percentage_cb, "ncb:", avg_stop_word_percentage_ncb)
    # print("has number percentage ->", "cb:", avg_number_cb, "ncb:", avg_number_ncb)
    # print("has determiner percentage ->", "cb:", avg_determiner_cb, "ncb:", avg_determiner_ncb)
    # print("has pronoun percentage ->", "cb:", avg_pronoun_cb, "ncb:", avg_pronoun_ncb)