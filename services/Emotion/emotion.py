import numpy as np
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion")
model = AutoModelWithLMHead.from_pretrained(
    "mrm8488/t5-base-finetuned-emotion")


class Emotion:
    """ To get emotions in the form of dictionary """

    def __init__(self, video_id: str):
        self.video_id = video_id

    def get_transcript(self, video_id):
        '''
        function to get transcripts
        parameters:
        video_url(string)=this is url of the video for which we want the transcript
        returns:
        result(string)= transcript of the url in required format
        '''

        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        global result
        result = ""
        for i in transcript:
            result += ' ' + i['text']
        result = result.lower()
        return result

    def token(self, text):
        '''
        function to encoding and decoding the text
        parameters:
        text(string)=transcript of the url index wise
        returns:
        label(string)= decoded text for the final output
        '''
        input_ids = tokenizer.encode(text + '</s>', return_tensors='pt')
        output = model.generate(input_ids=input_ids, max_length=2)
        dec = [tokenizer.decode(ids) for ids in output]
        label = dec[0]
        return label

    def emotion(self, result):
        '''
        function to get dictionary of emotions
        parameters:
        result(string)=transcript of the url in required format
        returns:
        final_dict(dict)= returns dictionary of the emotions where key is emotion
        '''
        output = result.split('.')
        if (len(output) > 10):
            chunks = result.split(' ')
            length = len(chunks)
            output = np.array_split(chunks, length / 10)

            def convert(lst):
                return ' '.join(lst)

            for i in range(len(output)):
                output[i] = convert(output[i])

            global outputlist
            outputlist = []
            for i in range(0, len(output)):
                outputlist.append(self.token(output[i]))
            outputlist = [w.replace('<pad>', '') for w in outputlist]
            outputlist = [w.replace(' ', '') for w in outputlist]

            global colors_dict
            colors_dict = {}
            for c in range(len(outputlist)):
                colors_dict[outputlist[c]] = outputlist.count(outputlist[c])

            global emojidictvalue
            emojidictvalue = list(colors_dict.values())

            global emojidict
            emojidict = []
            emojidict = list(colors_dict.keys())
            normalizer = 1 / float(sum(emojidictvalue))
            numListNormalized = [x * normalizer for x in emojidictvalue]

            final_dict = {}
            for i in range(len(emojidict)):
                final_dict[emojidict[i]] = numListNormalized[i]
            return final_dict

    def get(self):
        """ main method """
        try:
            video_transcript = self.get_transcript(self.video_id)
            emotion_list = self.emotion(video_transcript)
            return emotion_list

        except:
            return None
