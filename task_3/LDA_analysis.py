import numpy as np
import seaborn as sns
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim
from gensim import models
from gensim import corpora
import matplotlib.pyplot as plt
from collections import Counter
from gensim.models import Phrases
from gensim.models.coherencemodel import CoherenceModel
from wordcloud import WordCloud
import matplotlib.colors as mcolors

# Plotly imports - HTML plots
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls


# ======================================================================================================================
# LDA
# ======================================================================================================================

def LDA_analysis(data, pyLDAvis_show=False, plot_graphs=False):
    """
    :param data: the data to pass to LDA model
    :param pyLDAvis_show: Set if the pyLDAvis webpage will be loaded [ONLY FOR THE WHOLE DATASET]
    :param plot_graphs: whether to plot graphs or not
    :return: dictionary with 20 words for each topic and their frequency (count)
    """
    # Prepare bi-grams and tri-grams
    list_of_list_of_tokens = data['clean_text'].tolist()

    print("list_of_list_of_tokens: ", list_of_list_of_tokens)

    # threshold: Represent a score threshold for forming the phrases -- bi-grams/tri-grams (higher means fewer phrases, default: 10.0)
    bigram_model = Phrases(list_of_list_of_tokens, threshold=5.0)  # get ready bi-gram identifier
    # min_count: Ignore all words and bigrams with total collected count lower than this value (default=5)
    trigram_model = Phrases(bigram_model[list_of_list_of_tokens], min_count=1, threshold=5.0)  # get ready tri-gram identifier
    # get possible bi-grams and tri-grams, depending on the number of times that bi-grams and tri-grams appear together
    list_of_list_of_tokens = list(trigram_model[bigram_model[list_of_list_of_tokens]])

    print("tokens with uni-grams, bi-grams and tri-grams combined: ", list_of_list_of_tokens)

    # Prepare objects for LDA gensim implementation
    dictionary_LDA = corpora.Dictionary(list_of_list_of_tokens)
    print("dictionary_LDA", dictionary_LDA)

    # no_below: Filter words that appear in less than 2 posts
    # no_above: more than 0.8 documents (fraction of total corpus size, not absolute number)
    # keep_n: keep only the first 100000 most frequent tokens
    dictionary_LDA.filter_extremes(no_below=2, no_above=0.8, keep_n=100000)
    print("FILTERED dictionary_LDA", dictionary_LDA)
    corpus = [dictionary_LDA.doc2bow(list_of_tokens) for list_of_tokens in list_of_list_of_tokens]
    print("corpus", corpus)

    # Run LDA
    np.random.seed(123456)
    #num_topics = 5
    num_topics = int(3 + 0.001 * len(data))  # select number of topics based on data size [1 + 0.1% of the number of data]
    print("num_topics: ", num_topics)
    # num_topics: the number of topics
    # eta: the [distribution of the] number of words per topic
    # alpha: the [distribution of the] number of topics per document
    lda_model = models.LdaModel(corpus, num_topics=num_topics,
                                id2word=dictionary_LDA,
                                passes=20,
                                alpha=[0.01] * num_topics,
                                eta=[0.01] * len(dictionary_LDA.keys()))

    # print detected topics
    for i, topic in lda_model.show_topics(formatted=True, num_topics=num_topics, num_words=5):
        print(str(i) + ": " + topic)
        print()

    # Allocating topics to documents

    # print the document
    # print("document: ", data.clean_text.loc[0][:500])
    # print the % of topics a document is about
    print("% of topics document is about: ", lda_model[corpus[0]])  # corpus[0] means the first document.


    # Compute Coherence Score - use as indicator for evaluation
    coherence_model_lda = CoherenceModel(model=lda_model, texts=list_of_list_of_tokens, dictionary=dictionary_LDA, coherence='c_v', processes=1)
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

    # ======================================================================================================================
    # Data Exploration and Plots
    # ======================================================================================================================

    if plot_graphs:
        topics = [lda_model[corpus[i]] for i in range(len(data))]

        def topics_document_to_dataframe(topics_document, num_topics):
            res = pd.DataFrame(columns=range(num_topics))
            for topic_weight in topics_document:
                res.loc[0, topic_weight[0]] = topic_weight[1]
            return res

        # Create a matrix of topic weighting (similar to TF-IDF), with documents as rows and topics as columns
        document_topic = \
            pd.concat([topics_document_to_dataframe(topics_document, num_topics=num_topics) for topics_document in topics]) \
                .reset_index(drop=True).fillna(0)

        print("document_topic", document_topic)

        # Which documents are about topic 1
        print(document_topic.sort_values(1, ascending=False)[1])

        # print(data.clean_text.loc[91][:1000])

    # ======================================================================================================================

        # Looking at the distribution of topics in all documents

        sns.set(rc={'figure.figsize': (10, 20)})
        sns.heatmap(document_topic.loc[document_topic.idxmax(axis=1).sort_values().index])
        plt.show()

    # ======================================================================================================================

        # The distribution of posts per topic

        sns.set(rc={'figure.figsize': (10, 5)})
        document_topic.idxmax(axis=1).value_counts().plot.bar(color='lightblue')
        plt.show()



        data = [go.Bar(
            x=document_topic.columns.unique(),
            y=document_topic.idxmax(axis=1).value_counts().values,
            marker=dict(colorscale='Jet',
                        color=document_topic.idxmax(axis=1).value_counts().values
                        ),
            text='Text posts attributed to Topic'
        )]

        layout = go.Layout(
            title='Topics distribution'
        )

        fig = go.Figure(data=data, layout=layout)

        py.plot(fig, filename='topics-distribution.html')

    # ======================================================================================================================
        # Wordcloud of Top N words in each topic

        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

        cloud = WordCloud(width=1600,
                          height=800,
                          colormap='tab10',
                          color_func=lambda *args, **kwargs: cols[i],
                          prefer_horizontal=1.0)

        topics = lda_model.show_topics(formatted=False)

        fig, axes = plt.subplots(1, 3, figsize=(20, 10), facecolor='k', sharex=True, sharey=True)  # set the number of plots

        for i, ax in enumerate(axes.flatten()):
            fig.add_subplot(ax)
            topic_words = dict(topics[i][1])
            cloud.generate_from_frequencies(topic_words, max_font_size=300)
            plt.gca().imshow(cloud)
            plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16), color='white')
            plt.gca().axis('off')

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        plt.show()

    # ======================================================================================================================
        # Word Count and Importance of Topic Keywords

        data_flat = [w for w_list in list_of_list_of_tokens for w in w_list]
        counter = Counter(data_flat)

        out = []
        for i, topic in topics:
            for word, weight in topic:
                out.append([word, i, weight, counter[word]])

        df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])

        # Plot Word Count and Weights of Topic Keywords
        fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharey=True, dpi=100)
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
        for i, ax in enumerate(axes.flatten()):
            ax.bar(x='word', height="word_count", data=df.loc[df.topic_id == i, :], color=cols[i], width=0.5, alpha=0.3,
                   label='Word Count')
            ax_twin = ax.twinx()
            ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id == i, :], color=cols[i], width=0.2,
                        label='Weights')
            ax.set_ylabel('Word Count', color=cols[i])
            ax_twin.set_ylim(0, 0.030)
            ax.set_ylim(0, 3500)
            ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
            ax.tick_params(axis='y', left=False)
            ax.set_xticklabels(df.loc[df.topic_id == i, 'word'], rotation=30, horizontalalignment='right')
            ax.legend(loc='upper left')
            ax_twin.legend(loc='upper right')

        fig.tight_layout(w_pad=2)
        fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)
        plt.show()


    # ======================================================================================================================

    if pyLDAvis_show:  # Set if the pyLDAvis webpage will be loaded [ONLY FOR THE WHOLE DATASET]
        # Visualizing topics

        # https://cran.r-project.org/web/packages/LDAvis/vignettes/details.pdf

        # size of bubble: proportional to the proportions of the topics across the N total tokens in the corpus
        # red bars: estimated number of times a given term was generated by a given topic
        # blue bars: overall frequency of each term in the corpus

        # -- Relevance of words is computed with a parameter lambda
        # -- Lambda optimal value ~0.6 (https://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf)
        vis = pyLDAvis.gensim.prepare(topic_model=lda_model, corpus=corpus, dictionary=dictionary_LDA)
        pyLDAvis.show(vis)

    # ======================================================================================================================

    data_flat = [w for w_list in list_of_list_of_tokens for w in w_list]
    counter = Counter(data_flat)

    topics = lda_model.show_topics(formatted=False, num_words=20)
    word_freq = {}
    for i, topic in topics:
        for word, weight in topic:
            word_freq[word] = counter[word]


    return word_freq
