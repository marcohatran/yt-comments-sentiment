from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def topic_modeling(df):
    count_vect = CountVectorizer(max_df=0.8, min_df=2, stop_words='english')
    doc_term_matrix = count_vect.fit_transform(df['Clean Comment'].values.astype('U'))
    LDA = LatentDirichletAllocation(n_components=10, random_state=42)
    LDA.fit(doc_term_matrix)
    LDA_Advanced = LatentDirichletAllocation(n_components=10,
                                             max_iter=10,
                                             learning_method='online',
                                             random_state=100,
                                             batch_size=128,
                                             evaluate_every=-1,
                                             n_jobs=-1)

    LDA_Advanced.fit(doc_term_matrix)
    results = []
    for i, topic in enumerate(LDA_Advanced.components_):
        topic_data = {"topic_number": i,
                      "topic": [count_vect.get_feature_names()[i] for i in topic.argsort()[-10:]]}
        results.append(topic_data)

    return results
