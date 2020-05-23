import random
import matplotlib.pyplot as plt
from wordcloud import (WordCloud, get_single_color_func)


class GroupedColorFunc(object):
    """Create a color function object which assigns DIFFERENT SHADES of
       specified colors to certain words based on the color to words mapping.

       Uses wordcloud.get_single_color_func

       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.

       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.color_func_to_words = [
            (get_single_color_func(color), set(words))
            for (color, words) in color_to_words.items()]

        self.default_color_func = get_single_color_func(default_color)

    def get_color_func(self, word):
        """Returns a single_color_func associated with the word"""
        try:
            color_func = next(
                color_func for (color_func, words) in self.color_func_to_words
                if word in words)
        except StopIteration:
            color_func = self.default_color_func

        return color_func

    def __call__(self, word, **kwargs):
        return self.get_color_func(word)(word, **kwargs)


def emerg_topics_wordcloud(words_per_emergin_topic, words_freq):
    # Since the text is small collocations are turned off and text is lower-cased
    wc = WordCloud(width=1600, height=800, collocations=False).generate_from_frequencies(words_freq)

    numb_emerg_topics = len(words_per_emergin_topic)
    color_to_words = dict()  # pic random distinct colors to the human eye for each different emerging topics
    for topic in words_per_emergin_topic:
        r = int(random.random() * 256)
        g = int(random.random() * 256)
        b = int(random.random() * 256)
        step = 256 / numb_emerg_topics

        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        color = ['#%02x%02x%02x' % (r, g, b)]  # convert RGB to HEX value

        #color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])]  # generate random colors

        color_to_words[color[0]] = topic

    # Words that are not in any of the color_to_words values
    # will be colored with a grey single color function
    default_color = 'grey'

    # Create a color function with multiple tones
    grouped_color_func = GroupedColorFunc(color_to_words, default_color)

    # Apply our color function
    wc.recolor(color_func=grouped_color_func)

    # Plot
    plt.figure(figsize=(20, 10), facecolor='k')
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()

    #plt.savefig('wordcloud.png', facecolor='k', bbox_inches='tight')  # save the wordcloud
