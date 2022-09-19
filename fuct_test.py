from services.Comment.comment import Comment
from services.Comment.data_comment import DataComment
from services.Lda.lda_comment import topic_modeling
from services.Sentiment.sentiment import sentiments



def comments_topic_modeling_for_sentiment(youtube_api_key, video_id):

    comment = DataComment(youtube_api_key=youtube_api_key)

    data = comment.scrape_comments_with_replies(video_id)

    return topic_modeling(data)


