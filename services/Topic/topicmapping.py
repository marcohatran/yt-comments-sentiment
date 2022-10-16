import pandas as pd
import numpy as np
import seaborn as sns
import re
import matplotlib.pyplot as plt
import os
import streamlit as st
import contractions

from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import NMF


class TopicMapping:

    def load_dataset(self, df):
        clean_comments = df['Comment'].apply(self.remove_punctuations).apply(self.to_lowercase).apply(
            self.clean_html).apply(self.fix_apostrophe).apply(self.expand_contractions)
        return clean_comments, self.generate_word_cloud(clean_comments)

    def remove_punctuations(self, comment):
        return re.sub('[,\.!?]', '', comment)

    def fix_apostrophe(self, comment):
        apos = re.sub("&#39;", "''", comment)
        return re.sub("&quot", "", apos)

    def expand_contractions(self, comment):
        exp_comments = [contractions.fix(word) for word in comment.split()]
        return ' '.join(exp_comments)

    def clean_html(self, comment):
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', comment)
        return cleantext

    def to_lowercase(self, comments):
        return comments.lower()

    def generate_word_cloud(self, comments):
        long_string = ','.join(comments)
        word_cloud = WordCloud(background_color='white', max_words=5000, contour_width=5, contour_color='steelblue',
                               width=800, height=500)
        word_cloud.generate(long_string)
        return word_cloud.to_image()

    def plot_most_common_words(self, count_data, count_vectorizer, value):
        words = count_vectorizer.get_feature_names()
        total_counts = np.zeros(len(words))
        for t in count_data:
            total_counts += t.toarray()[0]
        count_dict = (zip(words, total_counts))
        count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[0:value]
        words = [w[0] for w in count_dict]
        counts = [w[1] for w in count_dict]
        x_pos = np.arange(len(words))
        plt.figure(2, figsize=(15, 15 / 1.6180))
        plt.subplot(title='asdasd')
        sns.set_context('notebook', font_scale=1.25, rc={'lines.linewidth': 2.5})
        sns.barplot(x_pos, counts, palette='husl')
        plt.xticks(x_pos, words, rotation=90)
        plt.xlabel('words')
        plt.ylabel('counts')
        plt.show()

    def print_topics(self, model, count_vectorizer, n_top_words):
        topic_list = []
        words = count_vectorizer.get_feature_names()
        for topic_idx, topic in enumerate(model.components_):
            print(topic)
            print("\nTopic #%d:" % topic_idx)
            print(" ".join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
            topic_list.append(" ".join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
        return topic_list

    def generate_topic(self, number_topics, number_words, count_vectorizer, count_data):
        lda = LDA(n_components=number_topics, n_jobs=-1)
        lda.fit(count_data)
        print("Topics generated using LDA:")
        topic_list = self.print_topics(lda, count_vectorizer, number_words)
        return topic_list

    def fetch_topic(self, df, no_of_topics):
        comments, result = self.load_dataset(df)
        count_vectorizer = CountVectorizer(stop_words='english')
        count_data = count_vectorizer.fit_transform(comments)
        topics = self.generate_topic(no_of_topics, 9, count_vectorizer, count_data)
        return topics

    @staticmethod
    def display_topics(model, features, no_top_words=5):
        topic_list = []
        all_words = features.get_feature_names()
        for topic, words in enumerate(model.components_):
            total = words.sum()
            largest = words.argsort()[::-1]  # invert sort order
            # topic_list.append(" ".join([all_words[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
            # topic_list.append(" ".join([all_words[largest[i]] for i in range(0, no_top_words)]))
            topic_result = {"topic_number": topic,
                            "topic_keywords": " ".join([all_words[largest[i]] for i in range(0, no_top_words)])}
            topic_list.append(topic_result)
        return topic_list



    def fetch_topic_tfidf(self, df, no_of_topics):
        comments, result = self.load_dataset(df)
        count_vectorizer = CountVectorizer(stop_words='english')
        count_data = count_vectorizer.fit_transform(comments)
        count_text_vectors = count_data
        tfidf_text_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_text_vectors = tfidf_text_vectorizer.fit_transform(comments)
        lda_text_model = LDA(n_components=no_of_topics, random_state=42)
        W_lda_text_matrix = lda_text_model.fit_transform(count_text_vectors)
        H_lda_text_matrix = lda_text_model.components_
        lda_result_df = pd.DataFrame(W_lda_text_matrix)
        comp_df = df
        comp_df['topic_number'] = lda_result_df.idxmax(axis=1)
        print(comp_df.columns)
        topic_keywords = self.display_topics(lda_text_model, tfidf_text_vectorizer)
        full_topic_result = {"topic_all": topic_keywords,
                             "topic_details": comp_df[['Comment', 'topic_number']].to_dict('records')
                             }

        return full_topic_result
