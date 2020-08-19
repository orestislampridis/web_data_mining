# web_data_mining
Web Data Mining project for Data and Web Science Msc

This project is an analysis performed on Twitter and Instagram data on the topic 'quarantine'. It features 7 main tasks in total, which are the following:
##### Project tasks
- [x] (task_1) Data collecting
- [x] (task_2) Pre-processing
- [x] (task_3) Emerging topics extraction
- [x] (task_3) Sentiment and emotion analysis
- [x] (all tasks) Visualization of data analysis
- [x] (task_5) User Profiling
- [x] (task_6) Like Prediction
- [x] (task_7) Web Application with Flask to showcase the visualization analysis

Website with analysis results hosted on HEROKU at: https://web-data-mining.herokuapp.com

Made by: Chrysovalantis Kontoulis, Orestis Lampridis, Petros Tzallas

### How to read big json file into database: twitter_db and collection: twitter_collection of mongoDB (file: tweets_full.json)
mongoimport --db twitter_db --collection twitter_collection --file tweets_full.json --batchSize 1
