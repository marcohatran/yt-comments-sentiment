from services.Comment.data_comment import DataComment
from services.Lda.lda_comment import topic_modeling
from services.Sentiment.sentiment import sentiments
from services.WordCloud.comment_word_cloud import word_cloud
from services.Emotion.comment_emotion import predict_emotions
from services.Topic.topicmapping import TopicMapping
from services.WordCloud.image_wordcloud import wordcloud_by_comments

class CommentAnalysis:
    def __init__(self, youtube_api_key, video_id):

        self.comment = DataComment(youtube_api_key=youtube_api_key)
        self.data = self.comment.scrape_comments_with_replies(video_id)

    def comments_topic_modeling_for_sentiment(self):
        return topic_modeling(self.data)

    def word_cloud_for_comment_sentiment(self):
        data_sentiment_all, positive, negative, neutral = sentiments(self.data)
        positive_comments = data_sentiment_all[data_sentiment_all['Analysis'] == 'Positive']
        most_like_positive_comment = positive_comments[positive_comments['Likes'] == positive_comments['Likes'].max()]
        negative_comments = data_sentiment_all[data_sentiment_all['Analysis'] == 'Negative']
        most_like_negative_comment = negative_comments[negative_comments['Likes'] == negative_comments['Likes'].max()]
        neutral_comments = data_sentiment_all[data_sentiment_all['Analysis'] == 'Neutral']
        most_like_neutral_comment = neutral_comments[neutral_comments['Likes'] == neutral_comments['Likes'].max()]
        # positive_comments_str = (' '.join(positive_comments['Clean Comment']))
        # negative_comments_str = (' '.join(negative_comments['Clean Comment']))
        # neutral_comments_str = (' '.join(neutral_comments['Clean Comment']))
        return {"positive_word_cloud": wordcloud_by_comments(positive_comments['Clean Comment'], 'positive wordcloud'),
                "negative_word_cloud": wordcloud_by_comments(negative_comments['Clean Comment'], 'negative wordcloud'),
                "neutral_word_cloud": wordcloud_by_comments(neutral_comments['Clean Comment'], 'neutral wordcloud'),
                "most_like_positive_comment": most_like_positive_comment['Comment'].values.tolist(),
                "most_like_negative_comment": most_like_negative_comment['Comment'].values.tolist(),
                "most_like_neutral_comment": most_like_neutral_comment['Comment'].values.tolist()}

    def comment_emotion_detection(self):
        self.data['emotion'] = self.data['Clean Comment'].apply(predict_emotions)
        df = self.data['emotion'].value_counts()
        return df.to_dict()

    def topic_modeling(self):
        topic = TopicMapping()
        return topic.fetch_topic_tfidf(self.data, 4)


if __name__ == '__main__':
    comment_analysis = CommentAnalysis(youtube_api_key='AIzaSyAvDyM4PVSaEWGheInZyvD7JWuWttBHqfg',
                                       video_id='-duwSMIgNMU')
    print(comment_analysis.topic_modeling())



