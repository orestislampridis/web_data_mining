# web_data_mining
Web Data Mining project for Data and Web Science Msc

Website with analysis results hosted on HEROKU at: https://web-data-mining.herokuapp.com

Made by: Chrysovalantis Kontoulis, Orestis Lampridis, Petros Tzallas

#### How to read big json file into database: twitter_db and collection: twitter_collection of mongoDB (file: tweets_full.json)
mongoimport --db twitter_db --collection twitter_collection --file tweets_full.json --batchSize 1
