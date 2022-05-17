# web_data_mining

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

Website with analysis results hosted on HEROKU at: https://web-data-mining.herokuapp.com (up to 30sec. loading time)

Created by: [Chrysovalantis Kontoulis](https://github.com/valantiskon), [Orestis Lampridis](https://github.com/orestislampridis) and [Petros Tzallas](https://github.com/ptzallas)

### How to read big json file into database: twitter_db and collection: twitter_collection of mongoDB (file: tweets_full.json)
mongoimport --db twitter_db --collection twitter_collection --file tweets_full.json --batchSize 1
