from googleapiclient.discovery import build
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords


class DataComment:
    def __init__(self, youtube_api_key: str):
        """ Description of Method """
        self.youtube = build('youtube', 'v3', developerKey=youtube_api_key)

    def scrape_comments_with_replies(self, video_id):
        box = [['Name', 'Comment', 'Time', 'Likes', 'Reply Count']]
        data = self.youtube.commentThreads().list(part='snippet', videoId=video_id, maxResults='100',
                                                  textFormat="plainText").execute()

        for i in data["items"]:
            name = i["snippet"]['topLevelComment']["snippet"]["authorDisplayName"]
            comment = i["snippet"]['topLevelComment']["snippet"]["textDisplay"]
            published_at = i["snippet"]['topLevelComment']["snippet"]['publishedAt']
            likes = i["snippet"]['topLevelComment']["snippet"]['likeCount']
            replies = i["snippet"]['totalReplyCount']

            box.append([name, comment, published_at, likes, replies])

            totalReplyCount = i["snippet"]['totalReplyCount']

            if totalReplyCount > 0:

                parent = i["snippet"]['topLevelComment']["id"]

                data2 = self.youtube.comments().list(part='snippet', maxResults='100', parentId=parent,
                                                     textFormat="plainText").execute()

                for i in data2["items"]:
                    name = i["snippet"]["authorDisplayName"]
                    comment = i["snippet"]["textDisplay"]
                    published_at = i["snippet"]['publishedAt']
                    likes = i["snippet"]['likeCount']
                    replies = ""

                    box.append([name, comment, published_at, likes, replies])

        while ("nextPageToken" in data):

            data = self.youtube.commentThreads().list(part='snippet', videoId=video_id, pageToken=data["nextPageToken"],
                                                      maxResults='100', textFormat="plainText").execute()

            for i in data["items"]:
                name = i["snippet"]['topLevelComment']["snippet"]["authorDisplayName"]
                comment = i["snippet"]['topLevelComment']["snippet"]["textDisplay"]
                published_at = i["snippet"]['topLevelComment']["snippet"]['publishedAt']
                likes = i["snippet"]['topLevelComment']["snippet"]['likeCount']
                replies = i["snippet"]['totalReplyCount']

                box.append([name, comment, published_at, likes, replies])

                totalReplyCount = i["snippet"]['totalReplyCount']

                if totalReplyCount > 0:

                    parent = i["snippet"]['topLevelComment']["id"]

                    data2 = self.youtube.comments().list(part='snippet', maxResults='100', parentId=parent,
                                                         textFormat="plainText").execute()

                    for i in data2["items"]:
                        name = i["snippet"]["authorDisplayName"]
                        comment = i["snippet"]["textDisplay"]
                        published_at = i["snippet"]['publishedAt']
                        likes = i["snippet"]['likeCount']
                        replies = ''

                        box.append([name, comment, published_at, likes, replies])
        box.pop(0)
        df = pd.DataFrame({'Name': [i[0] for i in box], 'Comment': [i[1] for i in box], 'Time': [i[2] for i in box],
                           'Likes': [i[3] for i in box], 'Reply Count': [i[4] for i in box]})
        # sql_vids = pd.DataFrame([])
        # sql_vids = sql_vids.append(df, ignore_index=True)
        df['Clean Comment'] = df['Comment'].apply(self.superclean)
        return df

    @staticmethod
    def superclean(text):
        # tokens = text.split(" ")
        stop_words = stopwords.words('english')
        text = text.lower().replace("'", "").replace('[^\w\s]', ' ').replace(" \d+", " ").strip()
        tokens = nltk.word_tokenize(text)
        stop_tokens = [item for item in tokens if item not in stop_words]
        new_text = ' '.join(stop_tokens)
        return new_text
    
