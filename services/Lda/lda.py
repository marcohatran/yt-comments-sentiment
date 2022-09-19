from nltk.stem import WordNetLemmatizer, SnowballStemmer
from youtube_transcript_api import YouTubeTranscriptApi
from .model import main as model
from .model import get_dictionary_processed_docs, preprocess
import joblib
import warnings
warnings.filterwarnings("ignore")

class Lda:
    """
    Service for LDA Topic Modelling
    """
    def __init__(self, video_id: str):
        """
        Initializing Class
        """
        self.video_id = video_id

    def retrieve_transcript(self):
        """
        Helper to fetch the Transcript for the video.
        """
        output = YouTubeTranscriptApi.get_transcript(self.video_id)
        segments = []
        for e in output:
            line = e['text']
            line = line.replace('\n', '')
            line = line.replace('>', '')
            line = line.replace('--', '')
            line = line.replace('♪', '')
            segments.append(line)

        transcript = " ".join(segments)
        return transcript

    def lda(self, transcript):
        """
        Helper to fetch the topics from the transcript.
        """
        lda_model = joblib.load(model())
        dictionary, _ = get_dictionary_processed_docs()
        bow_vector = dictionary.doc2bow(preprocess(transcript))

        output = {}
        for index_outer, score in sorted(lda_model[bow_vector], key=lambda tup: -1 * tup[1]):
            for index_inner, topic in lda_model.show_topics(formatted=False, num_words=10):
                word_list = [w[0] for w in topic]
                output[str(index_inner)] = {
                    "score": float(score),
                    "words": word_list
                }
        return output

    def get_topics(self):
        video_transcript = self.retrieve_transcript()
        lda = self.lda(video_transcript)
        return lda
