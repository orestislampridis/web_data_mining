import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud


def convert(string):
    string = (string.replace('(', ''))
    string = (string.replace(')', ''))
    string = (string.replace("'", ''))
    string = (string.replace(',', ''))

    return string


# Initialize dataframes and read csv files
bigrams_collocations_df = pd.read_csv('filtered_bigramFreqTable.csv')
trigrams_collocations_df = pd.read_csv('filtered_trigramFreqTable.csv')

bigrams_collocations_df["bigram"] = bigrams_collocations_df.bigram.apply(lambda x: convert(x))
trigrams_collocations_df["trigram"] = trigrams_collocations_df.trigram.apply(lambda x: convert(x))

print(bigrams_collocations_df)
print(trigrams_collocations_df)

data_1 = dict(zip(bigrams_collocations_df['bigram'].tolist(), bigrams_collocations_df['freq'].tolist()))
data_2 = dict(zip(trigrams_collocations_df['trigram'].tolist(), trigrams_collocations_df['freq'].tolist()))

wc = WordCloud(width=800, height=400, max_words=200).generate_from_frequencies(data_2)

f1 = plt.figure(figsize=(10, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()
f1.savefig("collocations_trigrams_wordcloud.png", bbox_inches='tight')
