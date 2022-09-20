from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("sampathkethineedi/industry-classification")
model = AutoModelForSequenceClassification.from_pretrained("sampathkethineedi/industry-classification")

industry_tags = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)


def industry_classification(text):
    industry_tag = industry_tags(text)
    return industry_tag


