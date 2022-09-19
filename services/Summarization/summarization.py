from youtube_transcript_api import YouTubeTranscriptApi
import gensim
from summarizer import Summarizer


class Summarization:
    """
    Service for Summarization
    """

    def __init__(self, video_id):
        self.video_id = video_id

    # function to get transcripts
    def get_transcript(self):
        try:
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

        except:
            print("An exception occurred")
            return None

    # Need a minimum of 10 sentences in the transcript
    def summarize_by_ratio(self, video_transcript):
        ratio = 0.3
        summary = gensim.summarization.summarize(video_transcript, ratio=ratio)
        return summary

    def summarize_by_word_count(self, video_transcript):
        max_word_count = 30
        summary = gensim.summarization.summarize(
            video_transcript, word_count=max_word_count)
        return summary

    def summarize_bert(self, video_transcript):
        model = Summarizer()
        summary = model(video_transcript)

        if summary is None or summary == '':
            return None

        result = {
            "summary": summary
        }
        return result
