import json
import pymongo
import datetime
import instaloader

import time


def on_data(data):
    try:
        datajson = json.loads(data)

        created_at = datajson['created_at']
        dt = datetime.datetime.strptime(created_at, '%a %b %d %H: %M: %S +0000 %Y')
        datajson['created_at'] = dt
        coll.insert('datajson')
        print('tweet inserted')
    except Exception as e:
        print(e)



def start_stream(coll):
    for i in range(0, 10):
        try:
            # code for instaloader
            L = instaloader.Instaloader()

            hashtag = 'quarantine'

            print("===================================================================================")
            posts = L.get_hashtag_posts(hashtag)
            print("GOT POSTS")
            print("===================================================================================")

            # get just one post from each user
            users = set()
            for post in posts:
                if not post.owner_profile in users:
                    print("===================================================================================")
                    print("start")
                    print("===================================================================================")
                    L.download_post(post, '#' + hashtag)
                    users.add(post.owner_profile)

                    print()

                    print("===================================================================================")
                    print("get_comments")
                    print(post.get_comments())
                    print("===================================================================================")
                    print("get_likes")
                    print(post.get_likes())
                    print("===================================================================================")
                    print("comments")
                    print(post.comments)
                    print("===================================================================================")
                    print("likes")
                    print(post.likes)
                    print("===================================================================================")
                    print("caption")
                    print(post.caption)
                    print("===================================================================================")
                    print("caption_mentions")
                    print(post.caption_mentions)
                    print("===================================================================================")
                    print("caption_hashtags")
                    print(post.caption_hashtags)
                    print("===================================================================================")
                    print("location")
                    print(post.location)
                    print("===================================================================================")

                    print("date")
                    print(post.date)
                    print("===================================================================================")
                    print("date_local")
                    print(post.date_local)
                    print("===================================================================================")
                    print("date_utc")
                    print(post.date_utc)
                    print("===================================================================================")
                    print("tagged_users")
                    print(post.tagged_users)
                    print("===================================================================================")
                    print("video_view_count")
                    print(post.video_view_count)
                    print("===================================================================================")

                    print("owner_id")
                    # The ID of the Post’s owner.
                    print(post.owner_id)
                    print("===================================================================================")
                    print("owner_profile")
                    # Profile instance of the Post’s owner.
                    print(post.owner_profile)
                    print("===================================================================================")
                    print("owner_username")
                    # The Post’s lowercase owner name.
                    print(post.owner_username)

                    print()

                    print("===================================================================================")
                    print("END")
                    print("===================================================================================")

    # ======================================================================================================================
    # Insert data to mongoDB
    # ======================================================================================================================

                    # get the like the posts of the owner take on average, e.g. last 10 posts
                    post.owner_profile.get_posts()

                    # download the last 5 pictures with given hashtag
                    #           L.download_hashtag(hashtag, max_count=5)


                    # put post date and the date of mining to compare in how many days the post collected these likes


                    # put the MongoDB document body together
                    doc_body = {
#                        "owner_profile": post.owner_profile,
                        "owner_followers": post.owner_profile.followers,
                        "followees": post.owner_profile.followees,
                        "comments": post.comments,
                        "likes": post.likes,
                        "caption": post.caption,
                        "caption_mentions": post.caption_mentions,
                        "caption_hashtags": post.caption_hashtags,
                        "tagged_users": post.tagged_users,
                        "location": post.location,
                        "video_view_count": post.video_view_count,
                        "owner_private": post.owner_profile.is_private,
                        "owner_viewable_story": post.owner_profile.has_viewable_story,
                        "owner_verified": post.owner_profile.is_verified,

                        "owner_igtvcount": post.owner_profile.igtvcount,
                        "date": post.date,
                        "date_local": post.date_local,
                        "date_utc": post.date_utc,
                        "owner_id": post.owner_id,
                        "owner_username": post.owner_username
                    }

                    # Insert data to mongoDB
                    print("ndocument body:", doc_body)
                    result = coll.insert_one(doc_body)

                    # just print the result if using 2.x or older of PyMongo
                    print("nresult _id:", result.inserted_id)

    # ======================================================================================================================

                else:
                    print("{} from {} skipped.".format(post, post.owner_profile))

            # get number of likes
            likes = set()
            for post in posts:
                print(post)
                likes = likes | set(post.get_likes())

            # sort post depending on their popularity and human involvement by taking into account number of likes and comments
            posts_sorted_by_likes = sorted(posts,
                                           key=lambda p: p.likes + p.comments,
                                           reverse=True)

            filename = 'instadata.json'

            # Saves a Post, Profile or StoryItem to a ‘.json’ or ‘.json.xz’ file such that it can later be loaded by load_structure_from_file().
            # If the specified filename ends in ‘.xz’, the file will be LZMA compressed. Otherwise, a pretty-printed JSON file will be created.
            # structure (Union[Post, Profile, StoryItem]) – Post, Profile or StoryItem
            # filename (str) – Filename, ends in ‘.json’ or ‘.json.xz’
        #            L.save_structure_to_file(structure, filename)

        # Loads a Post, Profile or StoryItem from a ‘.json’ or ‘.json.xz’ file that has been saved by save_structure_to_file().
        # context (InstaloaderContext) – Instaloader.context linked to the new object, used for additional queries if neccessary.
        # filename (str) – Filename, ends in ‘.json’ or ‘.json.xz’
        #           L.load_structure_from_file(L.context, filename)

        except Exception as e:
            print(e)


# ======================================================================================================================
# Stating Point
# ======================================================================================================================

conn = pymongo.MongoClient('mongodb://localhost') # ('mongodb://localhost:27017')   ('localhost', 27017)
db = conn['Instagram_Data'] # client['countries_db']
coll = db['post_data']

start_stream(coll)




'''
with open('currencies.json') as f:
    file_data = json.load(f)

# if pymongo >= 3.0 use insert_one() for inserting one document
coll.insert_one(file_data)

# if pymongo >= 3.0 use insert_many() for inserting many documents
coll.insert_many(file_data)


# ======================================================================================================================


# Insert many with custom id, if no id is given the mongoDB assigns one
mylist = [
  { "_id": 1, "name": "John", "address": "Highway 37"},
  { "_id": 2, "name": "Peter", "address": "Lowstreet 27"},
  { "_id": 3, "name": "Amy", "address": "Apple st 652"},
  { "_id": 4, "name": "Hannah", "address": "Mountain 21"},
  { "_id": 5, "name": "Michael", "address": "Valley 345"},
  { "_id": 6, "name": "Sandy", "address": "Ocean blvd 2"},
  { "_id": 7, "name": "Betty", "address": "Green Grass 1"},
  { "_id": 8, "name": "Richard", "address": "Sky st 331"},
  { "_id": 9, "name": "Susan", "address": "One way 98"},
  { "_id": 10, "name": "Vicky", "address": "Yellow Garden 2"},
  { "_id": 11, "name": "Ben", "address": "Park Lane 38"},
  { "_id": 12, "name": "William", "address": "Central st 954"},
  { "_id": 13, "name": "Chuck", "address": "Main Road 989"},
  { "_id": 14, "name": "Viola", "address": "Sideway 1633"}
]

coll.insert_many(mylist)


# ======================================================================================================================


# data to pass into the Python dictionary object
all_ages = [23, 45, 10, 56, 32]
datetime_now = datetime.datetime # pass this to a MongoDB doc
print("datetime_now:", datetime_now)

# put the MongoDB document body together
doc_body = {
"timestamp" : time.time(), # returns float of epoch time
"median age" : sum(all_ages)/len(all_ages),
"date" : datetime_now
}

# update dictionary keys to add new fields for the MongoDB document body
doc_body.update({"field int": 1234})
doc_body.update({"field str": "I am a string, yo!"})

# Insert data to mongoDB
print("ndocument body:", doc_body)
result = coll.insert_one(doc_body)

# just print the result if using 2.x or older of PyMongo
print("nresult _id:", result.inserted_id)
'''

# ======================================================================================================================

conn.close()

# ======================================================================================================================