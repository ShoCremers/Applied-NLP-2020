import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    df = pd.read_csv("dataset/features.csv")
    df_cb = df.loc[df['class'] == 1]
    df_ncb = df.loc[df['class'] == -1]

    cb = df_cb.describe()
    ncb = df_ncb.describe()

    ### plot for number of words
    # words_cb = cb['#words']['mean']
    # words_ncb = ncb['#words']['mean']
    #
    # Class = ('Clickbait', 'Non Clickbait')
    # y_pos = np.arange(len(Class))
    # words = [words_cb, words_ncb]
    #
    # plt.bar(y_pos, words, align='center', width=0.6, alpha=1, color=['b', 'orange'])
    # plt.xticks(y_pos, Class)
    # plt.ylabel('Average Number of Words')

    plt.show()

    ### plot for word length
    # word_length_cb = cb['word length']['mean']
    # word_length_ncb = ncb['word length']['mean']
    #
    # Class = ('Clickbait', 'Non Clickbait')
    # y_pos = np.arange(len(Class))
    # word_length = [word_length_cb, word_length_ncb]
    #
    # plt.bar(y_pos, word_length, align='center', width=0.6, alpha=1, color=['b', 'orange'])
    # plt.xticks(y_pos, Class)
    # plt.ylabel('Average Word Length')
    #
    # plt.show()

    ### plot for stop word %
    stopword_cb = cb['stopword%']['mean']
    stopword_ncb = ncb['stopword%']['mean']

    Class = ('Clickbait', 'Non Clickbait')
    y_pos = np.arange(len(Class))
    stopword = [stopword_cb, stopword_ncb]

    plt.bar(y_pos, stopword, align='center', width=0.6, alpha=1, color=['b', 'orange'])
    plt.xticks(y_pos, Class)
    plt.ylabel('Percentage')

    plt.show()


    ### plot for POS %

    # n_groups = 6
    # means_cb = (cb['has number']['mean'], cb['has determiner']['mean'], cb['has pronoun']['mean'],
    #          cb['has comparative']['mean'], cb['has superlative']['mean'], cb['has sup or comp']['mean'])
    # means_ncb = (ncb['has number']['mean'], ncb['has determiner']['mean'], ncb['has pronoun']['mean'],
    #          ncb['has comparative']['mean'], ncb['has superlative']['mean'], ncb['has sup or comp']['mean'])
    #
    # # create plot
    # fig, ax = plt.subplots()
    # index = np.arange(n_groups)
    # bar_width = 0.3
    # opacity = 0.8
    #
    # rects1 = plt.bar(index, means_cb, bar_width, alpha=opacity, color='b', label='Clickbait')
    #
    # rects2 = plt.bar(index + bar_width, means_ncb, bar_width, alpha=opacity, color='orange', label='Non Clickbait')
    #
    # plt.ylabel('Percentage')
    # plt.xticks(index + bar_width/2, ('Number', 'Determiner', 'Pronoun', 'Comparative', 'Superlative', 'Comparative \nor Superlative'))
    # plt.legend()
    #
    # plt.tight_layout()
    # plt.show()


    ### plot for sentiment score

    # sentiment_cb = cb['sentiment value']['mean']
    # sentiment_ncb = ncb['sentiment value']['mean']
    #
    # Class = ('Clickbait', 'Non Clickbait')
    # y_pos = np.arange(len(Class))
    # sentiment = [sentiment_cb, sentiment_ncb]
    #
    # plt.bar(y_pos, sentiment, align='center', width = 0.6, alpha=1, color = ['b','orange'])
    # plt.xticks(y_pos, Class)
    # plt.ylabel('Average Sentiment Score')
    #
    # plt.show()


