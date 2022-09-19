from transformers import pipeline

emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')

emotion_labels = emotion("I'm sorry that the order got delayed")
print(emotion_labels)

