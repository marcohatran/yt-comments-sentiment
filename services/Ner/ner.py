from youtube_transcript_api import YouTubeTranscriptApi
from nerd import ner
from collections import defaultdict


class Ner:
    """
    Description of Service
    The code is taken from: https://github.com/YouTubeNLP/NLP/tree/master/ner
    """

    def __init__(self, video_id):
        self.video_id = video_id

    def get_transcript(self, video_id):
        """ function to get transcripts """
        try:
            output = YouTubeTranscriptApi.get_transcript(video_id)
            segments = []
            for e in output:
                line = e["text"]
                line = line.replace("\n", "")
                line = line.replace(">", "")
                line = line.replace("--", "")
                line = line.replace("♪", "")
                segments.append(line)
                transcript = " ".join(segments)
            return transcript
        except Exception as e:
            print(e)
            return None

    def nerd_ner(self, document):
        """ function to apply ner using nerd library """
        doc = ner.name(document, language="en_core_web_sm")
        results = [(ent.text, ent.label_) for ent in doc]
        return results

    def get_ner(self):
        """ main method """
        try:
            video_transcript = self.get_transcript(self.video_id)
            ner_nerd = self.nerd_ner(video_transcript)
        except:
            return {
                "status": 500,
                "message": "Transcript is empty. Try another video"
            }
        ners = {"video_id": self.video_id, "entities": []}

        label_map = defaultdict(lambda: 0)
        label_count = 0
        for (text, label) in ner_nerd:
            if label not in label_map.keys():
                ners["entities"].append({
                    "entity": label,
                    "ner": [
                        text
                    ]
                })
                label_map[label] = label_count
                label_count += 1
            else:
                index = label_map[label]
                if text not in ners["entities"][index]["ner"]:
                    ners["entities"][index]["ner"].append(text)

        return dict(ners)
