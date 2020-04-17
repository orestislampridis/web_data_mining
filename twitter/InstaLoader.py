import json
import pymongo
import datetime
from math import ceil
from itertools import islice

import time
import numpy as np

import instaloader
from instaloader import Profile



def start_stream(coll):
    for i in range(0, 1000):
        print("REP %d FROM 1000" % i)
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
#                    L.download_post(post, '#' + hashtag)
                    users.add(post.owner_profile)

                    print()

                    print("===================================================================================")
                    print("TEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEST: _location")
                    print(post._location)
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
                    print("post.is_video")
                    print(post.is_video)
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

                    # ======================================================================================================================
                    # used for avg likes-comments-video_views calculation
                    profile = Profile.from_username(L.context, post.owner_username)  # get the profile of the user of the current post

                    # get the profile posts of the owner of the post
                    owner_profile_post = profile.get_posts()
                    # ======================================================================================================================

                    start_time = time.time()

                    print("===================================================================================")

                    # get statistics for all posts
                    photos_likes_all = []
                    photos_comments_all = []

                    videos_likes_all = []
                    videos_comments_all = []
                    videos_views_all = []
                    for profile_post in owner_profile_post:
                        if profile_post.is_video:  # if the post is video
                            # videos_likes_all
                            if profile_post.likes is not None:
                                videos_likes_all.append(profile_post.likes)
                            else:  # if there are no likes on the post
                                videos_likes_all.append(0)

                            # videos_comments_all
                            if profile_post.comments is not None:
                                videos_comments_all.append(profile_post.comments)
                            else:  # if there are no comments on the post
                                videos_comments_all.append(0)

                            # videos_views_all
                            if profile_post.video_view_count is not None:
                                videos_views_all.append(profile_post.video_view_count)
                            else:  # if there are no views on the post
                                videos_views_all.append(0)
                        else:  # if the post is photo
                            # photos_likes_all
                            if profile_post.likes is not None:
                                photos_likes_all.append(profile_post.likes)
                            else:  # if there are no likes on the post
                                photos_likes_all.append(0)

                            # photos_comments_all
                            if profile_post.comments is not None:
                                photos_comments_all.append(profile_post.comments)
                            else:  # if there are no comments on the post
                                photos_comments_all.append(0)

                    print("likes_all_photos ", photos_likes_all)
                    print("comments_all_photos ", photos_comments_all)
                    print("likes_all_videos ", videos_likes_all)
                    print("comments_all_videos ", videos_comments_all)

                    print("===================================================================================")
                    # count the number of posts that are videos
                    videos_count_all = len(videos_likes_all)
                    print("videos_count_all ", videos_count_all)

                    # count the number of posts that are photos
                    photos_count_all = len(photos_likes_all)
                    print("photos_count_all ", photos_count_all)

                    if not photos_count_all:  # in case the user does not have uploaded a photo
                        # ===================================================================================
                        all_photos_avg_likes = 0
                        all_photos_stdev_likes = 0
                        # ===================================================================================
                        all_photos_avg_comments = 0
                        all_photos_stdev_comments = 0
                    elif photos_count_all == 1:  # in case the user has uploaded just one photo
                        # ===================================================================================
                        all_photos_avg_likes = photos_likes_all[0]
                        all_photos_stdev_likes = 0
                        # ===================================================================================
                        all_photos_avg_comments = photos_comments_all[0]
                        all_photos_stdev_comments = 0
                    else:  # in case the user has uploaded more than one photo
                        print("===================================================================================")
                        print("all_photos_avg_likes")
                        # dtype=np.float64 -> more accurate results
                        all_photos_avg_likes = np.mean(photos_likes_all, dtype=np.float64)
                        print(all_photos_avg_likes)

                        print("all_photos_stdev_likes")
                        # dtype=np.float64 -> more accurate results
                        # ddof=0 -> default, interprete data as population, ddof=1 -> interprete data as samples, i.e. estimate true variance
                        all_photos_stdev_likes = np.std(photos_likes_all, dtype=np.float64, ddof=0)
                        print(all_photos_stdev_likes)

                        print("===================================================================================")
                        print("all_photos_avg_comments")
                        # dtype=np.float64 -> more accurate results
                        all_photos_avg_comments = np.mean(photos_comments_all, dtype=np.float64)
                        print(all_photos_avg_comments)

                        print("all_photos_stdev_comments")
                        # dtype=np.float64 -> more accurate results
                        # ddof=0 -> default, interprete data as population, ddof=1 -> interprete data as samples, i.e. estimate true variance
                        all_photos_stdev_comments = np.std(photos_comments_all, dtype=np.float64, ddof=0)
                        print(all_photos_stdev_comments)

                    if not videos_count_all:  # in case the user does not have uploaded a video
                        # ===================================================================================
                        all_videos_avg_likes = all_videos_stdev_likes = 0
                        # ===================================================================================
                        all_videos_avg_comments = all_videos_stdev_comments = 0
                        # ===================================================================================
                        all_videos_avg_views = all_videos_stdev_views = 0
                    elif videos_count_all == 1:  # in case the user has uploaded just one video
                        # ===================================================================================
                        all_videos_avg_likes = videos_likes_all[0]
                        all_videos_stdev_likes = 0
                        # ===================================================================================
                        all_videos_avg_comments = videos_comments_all[0]
                        all_videos_stdev_comments = 0
                        # ===================================================================================
                        all_videos_avg_views = videos_views_all[0]
                        all_videos_stdev_views = 0
                    else:  # in case the user has uploaded more than one video
                        print("===================================================================================")
                        print("all_videos_avg_likes")
                        # dtype=np.float64 -> more accurate results
                        all_videos_avg_likes = np.mean(videos_likes_all, dtype=np.float64)
                        print(all_videos_avg_likes)

                        print("all_videos_stdev_likes")
                        # dtype=np.float64 -> more accurate results
                        # ddof=0 -> default, interprete data as population, ddof=1 -> interprete data as samples, i.e. estimate true variance
                        all_videos_stdev_likes = np.std(videos_likes_all, dtype=np.float64, ddof=0)
                        print(all_videos_stdev_likes)
                        print("===================================================================================")
                        print("all_videos_avg_comments")
                        # dtype=np.float64 -> more accurate results
                        all_videos_avg_comments = np.mean(videos_comments_all, dtype=np.float64)
                        print(all_videos_avg_comments)

                        print("all_videos_stdev_comments")
                        # dtype=np.float64 -> more accurate results
                        # ddof=0 -> default, interprete data as population, ddof=1 -> interprete data as samples, i.e. estimate true variance
                        all_videos_stdev_comments = np.std(videos_comments_all, dtype=np.float64, ddof=0)
                        print(all_videos_stdev_comments)
                        print("===================================================================================")
                        print("all_videos_avg_views")
                        # dtype=np.float64 -> more accurate results
                        all_videos_avg_views = np.mean(videos_views_all, dtype=np.float64)
                        print(all_videos_avg_views)

                        print("all_videos_stdev_views")
                        # dtype=np.float64 -> more accurate results
                        # ddof=0 -> default, interprete data as population, ddof=1 -> interprete data as samples, i.e. estimate true variance
                        all_videos_stdev_views = np.std(videos_views_all, dtype=np.float64, ddof=0)
                        print(all_videos_stdev_views)

                    print("--- %.2f seconds ALL ---" % (time.time() - start_time))

                    # ======================================================================================================================

                    start_time1 = time.time()

                    print("===================================================================================")
                    print("user_profile_avg_10_percent_most_liked_commented_posts")

                    X_percentage = 30  # percentage of posts that should be downloaded

                    # sort post depending on like count
                    posts_sorted_by_likes = sorted(owner_profile_post,
                                                   key=lambda p: p.likes,
                                                   reverse=True)

                    # get statistics of 30% of most liked-commented posts
                    photos_likes_top_30_percent = []
                    photos_comments_top_30_percent = []

                    videos_likes_top_30_percent = []
                    videos_comments_top_30_percent = []
                    videos_views_top_30_percent = []
                    for profile_posts in islice(posts_sorted_by_likes, ceil(profile.mediacount * X_percentage / 100)):
                        # L.download_post(profile_posts, post.owner_username)
                        if profile_posts.is_video:  # if the post is video
                            # videos_likes_top_30_percent
                            if profile_posts.likes is not None:
                                videos_likes_top_30_percent.append(profile_posts.likes)
                            else:  # if there are no likes on the post
                                videos_likes_top_30_percent.append(0)

                            # videos_comments_top_30_percent
                            if profile_posts.comments is not None:
                                videos_comments_top_30_percent.append(profile_posts.comments)
                            else:  # if there are no comments on the post
                                videos_comments_top_30_percent.append(0)

                            # videos_views_top_30_percent
                            if profile_posts.video_view_count is not None:
                                videos_views_top_30_percent.append(profile_posts.video_view_count)
                            else:  # if there are no views on the post
                                videos_views_top_30_percent.append(0)
                        else:  # if the post is photo
                            # photos_likes_top_30_percent
                            if profile_posts.likes is not None:
                                photos_likes_top_30_percent.append(profile_posts.likes)
                            else:  # if there are no likes on the post
                                photos_likes_top_30_percent.append(0)

                            # photos_comments_top_30_percent
                            if profile_posts.comments is not None:
                                photos_comments_top_30_percent.append(profile_posts.comments)
                            else:  # if there are no comments on the post
                                photos_comments_top_30_percent.append(0)

                    print("likes_30_photos ", photos_likes_top_30_percent)
                    print("comments_30_photos ", photos_comments_top_30_percent)
                    print("likes_30_videos ", videos_likes_top_30_percent)
                    print("comments_30_videos ", videos_comments_top_30_percent)

                    print("===================================================================================")
                    # count the number of posts that are videos
                    videos_count_top_30_percent = len(videos_likes_top_30_percent)
                    print("videos_count_top_30_percent ", videos_count_top_30_percent)

                    # count the number of posts that are photos
                    photos_count_top_30_percent = len(photos_likes_top_30_percent)
                    print("photos_count_top_30_percent ", photos_count_top_30_percent)

                    if not photos_count_top_30_percent:  # in case the user does not have uploaded a photo
                        # ===================================================================================
                        photos_top_30_percent_avg_likes = 0
                        photos_top_30_percent_stdev_likes = 0
                        # ===================================================================================
                        photos_top_30_percent_avg_comments = 0
                        photos_top_30_percent_stdev_comments = 0
                        # ===================================================================================
                        photos_top_30_percent_avg_fan_engagement = 0
                    elif photos_count_top_30_percent == 1:  # in case the user has uploaded just one photo
                        # ===================================================================================
                        photos_top_30_percent_avg_likes = photos_likes_top_30_percent[0]
                        photos_top_30_percent_stdev_likes = 0
                        # ===================================================================================
                        photos_top_30_percent_avg_comments = photos_comments_top_30_percent[0]
                        photos_top_30_percent_stdev_comments = 0
                        # ===================================================================================
                        photos_top_30_percent_avg_fan_engagement = (
                                                                               photos_top_30_percent_avg_likes + photos_top_30_percent_avg_comments) / 2
                    else:  # in case the user has uploaded more than one photo
                        print("===================================================================================")
                        print("photos_top_30_percent_avg_likes")
                        # dtype=np.float64 -> more accurate results
                        photos_top_30_percent_avg_likes = np.mean(photos_likes_top_30_percent, dtype=np.float64)
                        print(photos_top_30_percent_avg_likes)

                        print("photos_top_30_percent_stdev_likes")
                        # dtype=np.float64 -> more accurate results
                        # ddof=0 -> default, interprete data as population, ddof=1 -> interprete data as samples, i.e. estimate true variance
                        photos_top_30_percent_stdev_likes = np.std(photos_likes_top_30_percent, dtype=np.float64,
                                                                   ddof=1)
                        print(photos_top_30_percent_stdev_likes)

                        print("===================================================================================")
                        print("photos_top_30_percent_avg_comments")
                        # dtype=np.float64 -> more accurate results
                        photos_top_30_percent_avg_comments = np.mean(photos_comments_top_30_percent, dtype=np.float64)
                        print(photos_top_30_percent_avg_comments)

                        print("photos_top_30_percent_stdev_comments")
                        # dtype=np.float64 -> more accurate results
                        # ddof=0 -> default, interprete data as population, ddof=1 -> interprete data as samples, i.e. estimate true variance
                        photos_top_30_percent_stdev_comments = np.std(photos_comments_top_30_percent, dtype=np.float64,
                                                                      ddof=1)
                        print(photos_top_30_percent_stdev_comments)

                        print("===================================================================================")
                        print("photos_top_30_percent_avg_fan_engagement")
                        # calculate a profile's popularity and human involvement by taking into account number of likes and comments of 30% of posts
                        photos_top_30_percent_avg_fan_engagement = (
                                                                               photos_top_30_percent_avg_likes + photos_top_30_percent_avg_comments) / 2
                        print(photos_top_30_percent_avg_fan_engagement)

                    if not videos_count_top_30_percent:  # in case the user does not have uploaded a video
                        # ===================================================================================
                        videos_top_30_percent_avg_likes = 0
                        videos_top_30_percent_stdev_likes = 0
                        # ===================================================================================
                        videos_top_30_percent_avg_comments = 0
                        videos_top_30_percent_stdev_comments = 0
                        # ===================================================================================
                        videos_top_30_percent_avg_views = 0
                        videos_top_30_percent_stdev_views = 0
                        # ===================================================================================
                        videos_top_30_percent_avg_fan_engagement = 0
                    elif videos_count_top_30_percent == 1:  # in case the user has uploaded just one video
                        # ===================================================================================
                        videos_top_30_percent_avg_likes = videos_likes_top_30_percent[0]
                        print("TESTING ", videos_top_30_percent_avg_likes, " here ", videos_likes_top_30_percent)
                        videos_top_30_percent_stdev_likes = 0
                        # ===================================================================================
                        videos_top_30_percent_avg_comments = videos_comments_top_30_percent[0]
                        videos_top_30_percent_stdev_comments = 0
                        # ===================================================================================
                        videos_top_30_percent_avg_views = videos_views_top_30_percent[0]
                        videos_top_30_percent_stdev_views = 0
                        # ===================================================================================
                        videos_top_30_percent_avg_fan_engagement = (videos_top_30_percent_avg_likes + videos_top_30_percent_avg_comments) / 2
                    else:  # in case the user has uploaded more than one video
                        print("===================================================================================")
                        print("videos_top_30_percent_avg_likes")
                        # dtype=np.float64 -> more accurate results
                        videos_top_30_percent_avg_likes = np.mean(videos_likes_top_30_percent, dtype=np.float64)
                        print(videos_top_30_percent_avg_likes)

                        print("videos_top_30_percent_stdev_likes")
                        # dtype=np.float64 -> more accurate results
                        # ddof=0 -> default, interprete data as population, ddof=1 -> interprete data as samples, i.e. estimate true variance
                        videos_top_30_percent_stdev_likes = np.std(videos_likes_top_30_percent, dtype=np.float64, ddof=1)
                        print(videos_top_30_percent_stdev_likes)
                        print("===================================================================================")
                        print("videos_top_30_percent_avg_comments")
                        # dtype=np.float64 -> more accurate results
                        videos_top_30_percent_avg_comments = np.mean(videos_comments_top_30_percent, dtype=np.float64)
                        print(videos_top_30_percent_avg_comments)

                        print("videos_top_30_percent_stdev_comments")
                        # dtype=np.float64 -> more accurate results
                        # ddof=0 -> default, interprete data as population, ddof=1 -> interprete data as samples, i.e. estimate true variance
                        videos_top_30_percent_stdev_comments = np.std(videos_comments_top_30_percent, dtype=np.float64, ddof=1)
                        print(videos_top_30_percent_stdev_comments)
                        print("===================================================================================")
                        print("videos_top_30_percent_avg_views")
                        # dtype=np.float64 -> more accurate results
                        videos_top_30_percent_avg_views = np.mean(videos_views_top_30_percent, dtype=np.float64)
                        print(videos_top_30_percent_avg_views)

                        print("videos_top_30_percent_stdev_views")
                        # dtype=np.float64 -> more accurate results
                        # ddof=0 -> default, interprete data as population, ddof=1 -> interprete data as samples, i.e. estimate true variance
                        videos_top_30_percent_stdev_views = np.std(videos_views_top_30_percent, dtype=np.float64, ddof=1)
                        print(videos_top_30_percent_stdev_views)

                        print("===================================================================================")
                        print("videos_top_30_percent_avg_fan_engagement")
                        # calculate a profile's popularity and human involvement by taking into account number of likes and comments of 30% of posts
                        videos_top_30_percent_avg_fan_engagement = (videos_top_30_percent_avg_likes + videos_top_30_percent_avg_comments) / 2
                        print(videos_top_30_percent_avg_fan_engagement)

                    print("--- %.2f seconds 30 ---" % (time.time() - start_time1))

                    # ======================================================================================================================

                    print()

                    print("===================================================================================")
                    print("END")
                    print("===================================================================================")

                    # ======================================================================================================================
                    # Insert data to mongoDB
                    # ======================================================================================================================

                    video_views = 0  # if the post is not a video, handle the none value
                    if post.video_view_count is not None:
                        video_views = post.video_view_count

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
                        "_location": post._location,
                        "video_view_count": video_views,
                        "is_video": post.is_video,
                        "owner_private": post.owner_profile.is_private,
                        "owner_viewable_story": post.owner_profile.has_viewable_story,
                        "owner_verified": post.owner_profile.is_verified,

                        "total_posts": videos_count_all + photos_count_all,
                        "videos_count_all": videos_count_all,
                        "photos_count_all": photos_count_all,
                        "all_photos_avg_likes": all_photos_avg_likes,
                        "all_photos_stdev_likes": all_photos_stdev_likes,
                        "all_photos_avg_comments": all_photos_avg_comments,
                        "all_photos_stdev_comments": all_photos_stdev_comments,
                        "all_videos_avg_likes": all_videos_avg_likes,
                        "all_videos_stdev_likes": all_videos_stdev_likes,
                        "all_videos_avg_comments": all_videos_avg_comments,
                        "all_videos_stdev_comments": all_videos_stdev_comments,
                        "all_videos_avg_views": all_videos_avg_views,
                        "all_videos_stdev_views": all_videos_stdev_views,

                        "top_30_percent_total_posts": videos_count_top_30_percent + photos_count_top_30_percent,
                        "videos_count_top_30_percent": videos_count_top_30_percent,
                        "photos_count_top_30_percent": photos_count_top_30_percent,
                        "photos_top_30_percent_avg_likes": photos_top_30_percent_avg_likes,
                        "photos_top_30_percent_stdev_likes": photos_top_30_percent_stdev_likes,
                        "photos_top_30_percent_avg_comments": photos_top_30_percent_avg_comments,
                        "photos_top_30_percent_stdev_comments": photos_top_30_percent_stdev_comments,
                        "videos_top_30_percent_avg_likes": videos_top_30_percent_avg_likes,
                        "videos_top_30_percent_stdev_likes": videos_top_30_percent_stdev_likes,
                        "videos_top_30_percent_avg_comments": videos_top_30_percent_avg_comments,
                        "videos_top_30_percent_stdev_comments": videos_top_30_percent_stdev_comments,
                        "videos_top_30_percent_avg_views": videos_top_30_percent_avg_views,
                        "videos_top_30_percent_stdev_views": videos_top_30_percent_stdev_views,
                        "photos_top_30_percent_avg_fan_engagement": photos_top_30_percent_avg_fan_engagement,
                        "videos_top_30_percent_avg_fan_engagement": videos_top_30_percent_avg_fan_engagement,

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

        except Exception as e:
            print(e)


# ======================================================================================================================
# Stating Point
# ======================================================================================================================

conn = pymongo.MongoClient('mongodb://localhost') # ('mongodb://localhost:27017')   ('localhost', 27017)
db = conn['Instagram_Data'] # client['countries_db']
coll = db['post_data']

start_stream(coll)

# ======================================================================================================================

conn.close()

# ======================================================================================================================
