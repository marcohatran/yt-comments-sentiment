from services.Comment.data_comment import DataComment
from services.Lda.lda_comment import topic_modeling
from services.Sentiment.sentiment import sentiments
from services.WordCloud.comment_word_cloud import word_cloud
from services.Emotion.comment_emotion import CommentEmotion

class CommentAnalysis:
    def __init__(self, youtube_api_key, video_id):

        self.comment = DataComment(youtube_api_key=youtube_api_key)
        self.data = self.comment.scrape_comments_with_replies(video_id)

    def comments_topic_modeling_for_sentiment(self):
        return topic_modeling(self.data)

    def word_cloud_for_comment_sentiment(self):
        data_sentiment_all,positive, negative, neutral  = sentiments(self.data)
        positive_comments = data_sentiment_all[data_sentiment_all['Analysis'] == 'Positive']
        negative_comments = data_sentiment_all[data_sentiment_all['Analysis'] == 'Negative']
        neutral_comments = data_sentiment_all[data_sentiment_all['Analysis'] == 'Neutral']
        positive_comments_str = (' '.join(positive_comments['Clean Comment']))
        negative_comments_str = (' '.join(negative_comments['Clean Comment']))
        neutral_comments_str = (' '.join(neutral_comments['Clean Comment']))
        print(word_cloud(positive_comments_str))
        return {"positive_word_cloud": word_cloud(positive_comments_str),
                "negative_word_cloud": word_cloud(negative_comments_str),
                "neutral_word_cloud": word_cloud(neutral_comments_str)}






if __name__ == '__main__':
    comment_analysis = CommentAnalysis(youtube_api_key='AIzaSyAvDyM4PVSaEWGheInZyvD7JWuWttBHqfg',
                                       video_id='-duwSMIgNMU')
    print(comment_analysis.comment_emotion_detection())


