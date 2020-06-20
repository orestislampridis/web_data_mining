import numpy as np
import os
from flask import Flask, request, jsonify, render_template



# http://localhost:5000


# app = Flask(__name__, template_folder='template') # to rename the templates folder that have the index.html file
app = Flask(__name__)



# ======================================================================================================================
# ADD ROUTES TO CREATE API
# ======================================================================================================================
@app.route('/')
def home():
    return render_template('index.html', subtask_title='This is the Main Page. Welcome')


@app.route('/ner', methods=['POST'])
def ner():

    path = os.getcwd() + "\\static\\results\\named entities"

    # ======================================================================================================================

    subtitle_images = []
    for image_name in os.listdir(path):
        # create the full input path and read the file
        input_path = os.path.join("\\static\\results\\named entities", image_name)
        subtitle_images.append((image_name.replace('.png', ''), input_path))
    print(subtitle_images)

    # subtitle_images = [('This is the distribution of gender', '\\static\\apps.png')]
    #print(os.getcwd() + '\\static\\apps.png')

    subtitle_html = [('HTML HERE', '\\templates\\gender-distribution.html')]

    return render_template('index.html', subtask_title='Named Entity Recognition', static_plots=subtitle_images, dynamic_plots=subtitle_html)



@app.route('/collocations', methods=['POST'])
def collocations():

    path = os.getcwd() + "\\static\\results\\named entities"

    # ======================================================================================================================

    subtitle_images = []
    for image_name in os.listdir(path):
        # create the full input path and read the file
        input_path = os.path.join("\\static\\results\\named entities", image_name)
        subtitle_images.append((image_name.replace('.png', ''), input_path))
    print(subtitle_images)

    # subtitle_images = [('This is the distribution of gender', '\\static\\apps.png')]
    #print(os.getcwd() + '\\static\\apps.png')

    subtitle_html = [('HTML HERE', '\\templates\\gender-distribution.html')]

    return render_template('index.html', subtask_title='Named Entity Recognition', static_plots=subtitle_images, dynamic_plots=subtitle_html)



@app.route('/emerg_topics', methods=['POST'])
def emerg_topics():

    twitter_path = os.getcwd() + "\\static\\results\\emerging topics\\twitter\\imgs\\all_topics"
    twitter_path2 = os.getcwd() + "\\static\\results\\emerging topics\\twitter\\imgs\\emerg_topics"

    insta_path = os.getcwd() + "\\static\\results\\emerging topics\\insta\\imgs\\all_topics"
    insta_path2 = os.getcwd() + "\\static\\results\\emerging topics\\insta\\imgs\\emerg_topics"

    # ======================================================================================================================

    subtitle_images = []

    # ======================================================================================================================

    for image_name in os.listdir(twitter_path):
        # create the full input path and read the file
        input_path = os.path.join("\\static\\results\\emerging topics\\twitter\\imgs\\all_topics", image_name)
        subtitle_images.append((image_name.replace('.png', ''), input_path))

    for image_name in os.listdir(twitter_path2):
        input_path2 = os.path.join("\\static\\results\\emerging topics\\twitter\\imgs\\emerg_topics", image_name)
        subtitle_images.append((image_name.replace('.png', ''), input_path2))

    # ======================================================================================================================

    for image_name in os.listdir(insta_path):
        # create the full input path and read the file
        input_path = os.path.join("\\static\\results\\emerging topics\\insta\\imgs\\all_topics", image_name)
        subtitle_images.append((image_name.replace('.png', ''), input_path))

    for image_name in os.listdir(insta_path2):
        input_path2 = os.path.join("\\static\\results\\emerging topics\\insta\\imgs\\emerg_topics", image_name)
        subtitle_images.append((image_name.replace('.png', ''), input_path2))

    # ======================================================================================================================

    print(subtitle_images)

    # subtitle_images = [('This is the distribution of gender', '\\static\\apps.png')]
    #print(os.getcwd() + '\\static\\apps.png')

    subtitle_html = [('HTML HERE', '\\templates\\gender-distribution.html')]

    return render_template('index.html', subtask_title='Emerging Topics', static_plots=subtitle_images, dynamic_plots=subtitle_html)



@app.route('/dtm', methods=['POST'])
def dtm():
    '''
    For rendering results on HTML GUI
    '''
    print("IM HERE")
    #form_text = [request.form.get('hate_speech_text_field')]
    # str(bytes_string, 'utf-8') # convert byte-string variable into a regular string

    return render_template('index.html')



@app.route('/affect_analysis', methods=['POST'])
def affect_analysis():

    path = os.getcwd() + "\\static\\results\\named entities"

    # ======================================================================================================================

    subtitle_images = []
    for image_name in os.listdir(path):
        # create the full input path and read the file
        input_path = os.path.join("\\static\\results\\named entities", image_name)
        subtitle_images.append((image_name.replace('.png', ''), input_path))
    print(subtitle_images)

    # subtitle_images = [('This is the distribution of gender', '\\static\\apps.png')]
    #print(os.getcwd() + '\\static\\apps.png')

    subtitle_html = [('HTML HERE', '\\templates\\gender-distribution.html')]

    return render_template('index.html', subtask_title='Affective Analysis', static_plots=subtitle_images, dynamic_plots=subtitle_html)



# EXAMPLE 2
@app.route('/sentim_analysis', methods=['POST'])
def sentim_analysis():

    path = os.getcwd() + "\\static\\results\\sentiment\\imgs"

    # ======================================================================================================================

    subtitle_images = []
    for image_name in os.listdir(path):
        # create the full input path and read the file
        input_path = os.path.join("\\static\\results\\sentiment\\imgs", image_name)
        subtitle_images.append((image_name.replace('.png', ''), input_path))
    print(subtitle_images)

    # subtitle_images = [('This is the distribution of gender', '\\static\\apps.png')]
    #print(os.getcwd() + '\\static\\apps.png')

    subtitle_html = [('HTML HERE', '\\templates\\gender-distribution.html')]

    return render_template('index.html', subtask_title='Sentiment Analysis', static_plots=subtitle_images, dynamic_plots=subtitle_html)



if __name__ == "__main__":
    app.run(debug=True)
    # app.run("localhost", "9999", debug=True)