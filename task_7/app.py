from flask import Flask, request, render_template



# http://localhost:5000


# app = Flask(__name__, template_folder='template') # to rename the templates folder that have the index.html file
app = Flask(__name__)


# ======================================================================================================================
# ADD ROUTES TO CREATE API
# ======================================================================================================================
@app.route('/')
def home():
    return render_template('index.html', subtask_title='This is the Main Page. Welcome')


# EXAMPLE 1
@app.route('/emerg_topics', methods=['POST'])
def emerg_topics():
    '''
    For rendering results on HTML GUI
    '''
    form_text = [request.form.get('hate_speech_text_field')]
    # str(bytes_string, 'utf-8') # convert byte-string variable into a regular string

    return render_template('index.html')


@app.route('/dtm', methods=['POST'])
def dtm():
    '''
    For rendering results on HTML GUI
    '''
    print("IM HERE")
    # form_text = [request.form.get('hate_speech_text_field')]
    # str(bytes_string, 'utf-8') # convert byte-string variable into a regular string

    return render_template('index.html')


@app.route('/ner', methods=['POST'])
def ner():
    '''
    For rendering results on HTML GUI
    '''
    form_text = [request.form.get('hate_speech_text_field')]
    # str(bytes_string, 'utf-8') # convert byte-string variable into a regular string

    return render_template('index.html')


@app.route('/affect_analysis', methods=['POST'])
def affect_analysis():
    '''
    For rendering results on HTML GUI
    '''
    # str(bytes_string, 'utf-8') # convert byte-string variable into a regular string

    import os
    print(os.getcwd() + '\\static\\apps.png')
    subtitle_images = [('This is the distribution of gender', '\\static\\apps.png')]
    subtitle_html = [('HTML HERE', '\\templates\\gender-distribution.html')]

    return render_template('index.html', subtask_title='Affective Analysis', static_plots=subtitle_images,
                           dynamic_plots=subtitle_html)


# EXAMPLE 2
@app.route('/sentim_analysis', methods=['POST'])
def sentim_analysis():
    form = request.form
    if request.method == 'POST':
        year = request.form['year']
        return render_template('index.html', prediction_text='Employee Salary should be {}')



if __name__ == "__main__":
    app.run(debug=True)
    # app.run("localhost", "9999", debug=True)