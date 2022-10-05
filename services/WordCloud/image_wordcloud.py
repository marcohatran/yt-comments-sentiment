from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import base64
from io import BytesIO


def wordcloud_by_comments(comments, title):
    stopwords = set(STOPWORDS)
    wc = WordCloud(background_color="white", stopwords=stopwords, random_state=2016).generate(
        " ".join([i for i in comments.str.upper()]))
    plt.imshow(wc)
    plt.axis("off")
    plt.title(title)
    wc_img = wc.to_image()
    with BytesIO() as buffer:
        wc_img.save(buffer, 'png')
        img2 = base64.b64encode(buffer.getvalue()).decode()
    return img2
