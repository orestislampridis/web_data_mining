import os
from flask import Flask, request, jsonify, render_template



# http://localhost:5000


# app = Flask(__name__, template_folder='template') # to rename the templates folder that have the index.html file
app = Flask(__name__)

app.debug = False
app.secret_key = "gsjdgsgsdgdsgsdgshdfgsgsdfsrtjkueev"


# ======================================================================================================================
# ADD ROUTES TO CREATE API
# ======================================================================================================================
@app.route('/')
def index():
    subtitle_html = [('Instagram - Posts per date', 'https://rawcdn.githack.com/orestislampridis/web_data_mining/31761ac0fd4a6cf2424f639b2e5601d4df7dc1e7/task_7/static/results/html_post_per_day/insta_posts_per_day.html'),
                     ('Twitter - Posts per date', 'https://rawcdn.githack.com/orestislampridis/web_data_mining/31761ac0fd4a6cf2424f639b2e5601d4df7dc1e7/task_7/static/results/html_post_per_day/twitter_posts_per_day.html')
                     ]

    return render_template('index.html', subtask_title='This is the Main Page. Welcome', dynamic_plots=subtitle_html)


@app.route('/home', methods=['POST'])
def home():
    subtitle_html = [('Instagram - Posts per date', 'https://rawcdn.githack.com/orestislampridis/web_data_mining/31761ac0fd4a6cf2424f639b2e5601d4df7dc1e7/task_7/static/results/html_post_per_day/insta_posts_per_day.html'),
                     ('Twitter - Posts per date', 'https://rawcdn.githack.com/orestislampridis/web_data_mining/31761ac0fd4a6cf2424f639b2e5601d4df7dc1e7/task_7/static/results/html_post_per_day/twitter_posts_per_day.html')
                     ]

    return render_template('index.html', subtask_title='This is the Main Page. Welcome', dynamic_plots=subtitle_html)


# ======================================================================================================================
# TASK 3
# ======================================================================================================================

@app.route('/ner', methods=['POST'])
def ner():

    path = "static/results/named entities"

    # ======================================================================================================================

    subtitle_images = []
    for image_name in os.listdir(path):
        # create the full input path and read the file
        input_path = os.path.join("static/results/named entities", image_name)
        subtitle_images.append((image_name.replace('.png', ''), input_path))
    print(subtitle_images)

    return render_template('index.html', subtask_title='Named Entity Recognition', static_plots=subtitle_images)



@app.route('/collocations', methods=['POST'])
def collocations():

    path = "static/results/Collocations"

    for x in os.listdir("static"):
        print(x)

    for x in os.listdir("static/results/Collocations"):
        print(x)

    # ======================================================================================================================

    subtitle_images = []
    for image_name in os.listdir(path):
        # create the full input path and read the file
        input_path = os.path.join("static/results/Collocations", image_name)
        subtitle_images.append((image_name.replace('.png', ''), input_path))
    print(subtitle_images)

    return render_template('index.html', subtask_title='Collocations', static_plots=subtitle_images)



@app.route('/emerg_topics', methods=['POST'])
def emerg_topics():

    twitter_path = "static/results/emerging topics/twitter/imgs/all_topics"
    twitter_path2 = "static/results/emerging topics/twitter/imgs/emerg_topics"

    insta_path = "static/results/emerging topics/insta/imgs/all_topics"
    insta_path2 = "static/results/emerging topics/insta/imgs/emerg_topics"

    # ======================================================================================================================

    subtitle_images = []

    # ======================================================================================================================

    for image_name in os.listdir(twitter_path):
        # create the full input path and read the file
        input_path = os.path.join("static/results/emerging topics/twitter/imgs/all_topics", image_name)
        subtitle_images.append((image_name.replace('.png', ''), input_path))

    subtitle_images.sort(key=lambda tup: tup[0])

    temp_subtitle_images = []
    for image_name in os.listdir(twitter_path2):
        input_path2 = os.path.join("static/results/emerging topics/twitter/imgs/emerg_topics", image_name)
        temp_subtitle_images.append((image_name.replace('.png', ''), input_path2))

    temp_subtitle_images.sort(key=lambda tup: tup[0])

    subtitle_images += temp_subtitle_images

    # ======================================================================================================================

    temp_subtitle_images1 = []
    for image_name in os.listdir(insta_path):
        # create the full input path and read the file
        input_path = os.path.join("static/results/emerging topics/insta/imgs/all_topics", image_name)
        temp_subtitle_images1.append((image_name.replace('.png', ''), input_path))

    temp_subtitle_images1.sort(key=lambda tup: tup[0])

    subtitle_images += temp_subtitle_images1

    temp_subtitle_images2 = []
    for image_name in os.listdir(insta_path2):
        input_path2 = os.path.join("static/results/emerging topics/insta/imgs/emerg_topics", image_name)
        temp_subtitle_images2.append((image_name.replace('.png', ''), input_path2))

    temp_subtitle_images2.sort(key=lambda tup: tup[0])

    subtitle_images += temp_subtitle_images2

    # ======================================================================================================================

    print(subtitle_images)

    # ======================================================================================================================

    subtitle_html = [('Instagram - Topics Distribution', 'https://rawcdn.githack.com/orestislampridis/web_data_mining/31761ac0fd4a6cf2424f639b2e5601d4df7dc1e7/task_7/static/results/emerging%20topics/insta/html_files/topics-distribution.html'),
                     ('Twitter - Topics Distribution', 'https://rawcdn.githack.com/orestislampridis/web_data_mining/31761ac0fd4a6cf2424f639b2e5601d4df7dc1e7/task_7/static/results/emerging%20topics/twitter/html_files/topics-distribution.html'),
                     ('Instagram - LDA Analysis on whole dataset', 'https://rawcdn.githack.com/orestislampridis/web_data_mining/31761ac0fd4a6cf2424f639b2e5601d4df7dc1e7/task_7/static/results/emerging%20topics/insta/html_files/insta_lda.html'),
                     ('Twitter - LDA Analysis on whole dataset', 'https://rawcdn.githack.com/orestislampridis/web_data_mining/31761ac0fd4a6cf2424f639b2e5601d4df7dc1e7/task_7/static/results/emerging%20topics/twitter/html_files/twitter_lda.html')
                     ]

    return render_template('index.html', subtask_title='Emerging Topics', static_plots=subtitle_images, dynamic_plots=subtitle_html)



@app.route('/affect_analysis', methods=['POST'])
def affect_analysis():

    path = "static/results/affective analysis"

    # ======================================================================================================================

    subtitle_images = []
    for image_name in os.listdir(path):
        # create the full input path and read the file
        input_path = os.path.join("static/results/affective analysis", image_name)
        subtitle_images.append((image_name.replace('.png', ''), input_path))
    print(subtitle_images)

    # ======================================================================================================================

    subtitle_html = [('Instagram - Emotion per day (One-vs-Rest Linear SVC)', 'https://rawcdn.githack.com/orestislampridis/web_data_mining/dfdbf76953b23ee62b618f52c6e7050d0bc6e13d/task_7/static/results/affective_analysis_html/insta_emotion_per_day.html'),
                     ('Twitter - Emotion per day (One-vs-Rest Linear SVC)', 'https://rawcdn.githack.com/orestislampridis/web_data_mining/dfdbf76953b23ee62b618f52c6e7050d0bc6e13d/task_7/static/results/affective_analysis_html/twitter_emotion_per_day.html')
                     ]

    return render_template('index.html', subtask_title='Affective Analysis', static_plots=subtitle_images, dynamic_plots=subtitle_html)



@app.route('/sentim_analysis', methods=['POST'])
def sentim_analysis():

    path = "static/results/sentiment detection"

    # ======================================================================================================================

    subtitle_images = []
    for image_name in os.listdir(path):
        # create the full input path and read the file
        input_path = os.path.join("static/results/sentiment detection", image_name)
        subtitle_images.append((image_name.replace('.png', ''), input_path))
    print(subtitle_images)

    # ======================================================================================================================

    subtitle_html = [('Sentiment Distribution - VADER vs TextBlob', 'https://rawcdn.githack.com/orestislampridis/web_data_mining/31761ac0fd4a6cf2424f639b2e5601d4df7dc1e7/task_7/static/results/sentiment/sentiment-distribution.html')]

    return render_template('index.html', subtask_title='Sentiment Analysis', static_plots=subtitle_images, dynamic_plots=subtitle_html)


# ======================================================================================================================
# TASK 5
# ======================================================================================================================

@app.route('/age_detect', methods=['POST'])
def age_detect():

    path = "static/results/age prediction"

    # ======================================================================================================================

    subtitle_images = []
    for image_name in os.listdir(path):
        # create the full input path and read the file
        input_path = os.path.join("static/results/age prediction", image_name)
        subtitle_images.append((image_name.replace('.png', ''), input_path))
    print(subtitle_images)

    return render_template('index.html', subtask_title='Age Detection', static_plots=subtitle_images)



@app.route('/gend_detect', methods=['POST'])
def gend_detect():

    path = "static/results/gender detection/imgs"

    # ======================================================================================================================

    subtitle_images = []
    for image_name in os.listdir(path):
        # create the full input path and read the file
        input_path = os.path.join("static/results/gender detection/imgs", image_name)
        subtitle_images.append((image_name.replace('.png', ''), input_path))
    print(subtitle_images)


    subtitle_html = [('', 'https://rawcdn.githack.com/orestislampridis/web_data_mining/31761ac0fd4a6cf2424f639b2e5601d4df7dc1e7/task_7/static/results/gender%20detection/html_files/gender-distribution.html')]

    return render_template('index.html', subtask_title='Sentiment Analysis', static_plots=subtitle_images, dynamic_plots=subtitle_html)


@app.route('/person_detect', methods=['POST'])
def person_detect():

    path = "static/results/Personality Detection"

    # ======================================================================================================================

    subtitle_images = []
    for image_name in os.listdir(path):
        # create the full input path and read the file
        input_path = os.path.join("static/results/Personality Detection", image_name)
        subtitle_images.append((image_name.replace('.png', ''), input_path))
    print(subtitle_images)

    return render_template('index.html', subtask_title='Sentiment Analysis', static_plots=subtitle_images)


# ======================================================================================================================
# TASK 6
# ======================================================================================================================

@app.route('/base_like', methods=['POST'])
def base_like():

    twitter_path = "static/results/emerging topics/twitter/imgs/all_topics"
    twitter_path2 = "static/results/emerging topics/twitter/imgs/emerg_topics"

    insta_path = "static/results/emerging topics/insta/imgs/all_topics"
    insta_path2 = "static/results/emerging topics/insta/imgs/emerg_topics"

    # ======================================================================================================================

    subtitle_images = []

    # ======================================================================================================================

    for image_name in os.listdir(twitter_path):
        # create the full input path and read the file
        input_path = os.path.join("static/results/emerging topics/twitter/imgs/all_topics", image_name)
        subtitle_images.append((image_name.replace('.png', ''), input_path))

    for image_name in os.listdir(twitter_path2):
        input_path2 = os.path.join("static/results/emerging topics/twitter/imgs/emerg_topics", image_name)
        subtitle_images.append((image_name.replace('.png', ''), input_path2))

    # ======================================================================================================================

    for image_name in os.listdir(insta_path):
        # create the full input path and read the file
        input_path = os.path.join("static/results/emerging topics/insta/imgs/all_topics", image_name)
        subtitle_images.append((image_name.replace('.png', ''), input_path))

    for image_name in os.listdir(insta_path2):
        input_path2 = os.path.join("static/results/emerging topics/insta/imgs/emerg_topics", image_name)
        subtitle_images.append((image_name.replace('.png', ''), input_path2))

    # ======================================================================================================================

    print(subtitle_images)

    # ======================================================================================================================

    subtitle_html = [('', 'https://rawcdn.githack.com/orestislampridis/web_data_mining/31761ac0fd4a6cf2424f639b2e5601d4df7dc1e7/task_7/static/results/like%20prediction/insta/html_files/base/insta_base_model_data_distrib.html'),
                     ('', 'https://rawcdn.githack.com/orestislampridis/web_data_mining/31761ac0fd4a6cf2424f639b2e5601d4df7dc1e7/task_7/static/results/like%20prediction/twitter/html_files/base/twitter_base_model_pred_best_model.html'),

                     ('', 'https://rawcdn.githack.com/orestislampridis/web_data_mining/31761ac0fd4a6cf2424f639b2e5601d4df7dc1e7/task_7/static/results/like%20prediction/insta/html_files/base/insta_base_model_feature_imp.html'),
                     ('', 'https://rawcdn.githack.com/orestislampridis/web_data_mining/31761ac0fd4a6cf2424f639b2e5601d4df7dc1e7/task_7/static/results/like%20prediction/twitter/html_files/base/twitter_base_model_feature_imp.html'),

                     ('', 'https://rawcdn.githack.com/orestislampridis/web_data_mining/31761ac0fd4a6cf2424f639b2e5601d4df7dc1e7/task_7/static/results/like%20prediction/insta/html_files/base/insta_base_model_perform.html'),
                     ('', 'https://rawcdn.githack.com/orestislampridis/web_data_mining/31761ac0fd4a6cf2424f639b2e5601d4df7dc1e7/task_7/static/results/like%20prediction/twitter/html_files/base/twitter_base_model_perform.html')
                     ]

    return render_template('index.html', subtask_title='Emerging Topics', static_plots=subtitle_images, dynamic_plots=subtitle_html)




@app.route('/nlp_like', methods=['POST'])
def nlp_like():

    twitter_path = "static/results/emerging topics/twitter/imgs/all_topics"
    twitter_path2 = "static/results/emerging topics/twitter/imgs/emerg_topics"

    insta_path = "static/results/emerging topics/insta/imgs/all_topics"
    insta_path2 = "static/results/emerging topics/insta/imgs/emerg_topics"

    # ======================================================================================================================

    subtitle_images = []

    # ======================================================================================================================

    for image_name in os.listdir(twitter_path):
        # create the full input path and read the file
        input_path = os.path.join("static/results/emerging topics/twitter/imgs/all_topics", image_name)
        subtitle_images.append((image_name.replace('.png', ''), input_path))

    for image_name in os.listdir(twitter_path2):
        input_path2 = os.path.join("static/results/emerging topics/twitter/imgs/emerg_topics", image_name)
        subtitle_images.append((image_name.replace('.png', ''), input_path2))

    # ======================================================================================================================

    for image_name in os.listdir(insta_path):
        # create the full input path and read the file
        input_path = os.path.join("static/results/emerging topics/insta/imgs/all_topics", image_name)
        subtitle_images.append((image_name.replace('.png', ''), input_path))

    for image_name in os.listdir(insta_path2):
        input_path2 = os.path.join("static/results/emerging topics/insta/imgs/emerg_topics", image_name)
        subtitle_images.append((image_name.replace('.png', ''), input_path2))

    # ======================================================================================================================

    print(subtitle_images)

    # ======================================================================================================================

    subtitle_html = [('', 'https://rawcdn.githack.com/orestislampridis/web_data_mining/31761ac0fd4a6cf2424f639b2e5601d4df7dc1e7/task_7/static/results/like%20prediction/insta/html_files/nlp/insta_nlp_perform.html'),
                     ('', 'https://rawcdn.githack.com/orestislampridis/web_data_mining/61b6e960eedcd516fccf465c4e3440b901787d55/task_7/static/results/like prediction/twitter/nlp/twitter_nlp_perform.html'),

                     ('', 'https://rawcdn.githack.com/orestislampridis/web_data_mining/31761ac0fd4a6cf2424f639b2e5601d4df7dc1e7/task_7/static/results/like%20prediction/insta/html_files/nlp/insta_nlp_pred_best_model.html'),
                     ('', 'https://rawcdn.githack.com/orestislampridis/web_data_mining/61b6e960eedcd516fccf465c4e3440b901787d55/task_7/static/results/like prediction/twitter/nlp/twitter_nlp_pred_best_model.html')
                     ]

    return render_template('index.html', subtask_title='Emerging Topics', static_plots=subtitle_images, dynamic_plots=subtitle_html)



# ======================================================================================================================
# MAP
# ======================================================================================================================

@app.route('/super_map', methods=['POST'])
def super_map():
    subtitle_html = ['https://rawcdn.githack.com/orestislampridis/web_data_mining/31761ac0fd4a6cf2424f639b2e5601d4df7dc1e7/task_4/map_1.html']

    return render_template('index.html', subtask_title='Analysis Visualization - Super Map', map_plots=subtitle_html)


@app.route('/cluster_map', methods=['POST'])
def cluster_map():
    subtitle_html = ['https://rawcdn.githack.com/orestislampridis/web_data_mining/31761ac0fd4a6cf2424f639b2e5601d4df7dc1e7/task_4/map_2.html']

    return render_template('index.html', subtask_title='Analysis Visualization - Super Cluster Map', map_plots=subtitle_html)


# ======================================================================================================================
# Main
# ======================================================================================================================

if __name__ == "__main__":
    app.run(debug=True)
    # app.run("localhost", "9999", debug=True)