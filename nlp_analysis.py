import json
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter, defaultdict
from textblob import TextBlob
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from transformers import pipeline
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from joypy import joyplot
import matplotlib.dates as mdates
import math


# Minimum number of sentences required for analysis
MIN_SENTENCES = 1

def setup_logger(log_folder):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_folder / f'nlp_analysis_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

log_folder = Path('./logs')
log_folder.mkdir(exist_ok=True)
logger = setup_logger(log_folder=log_folder)

logger.info(f"Starting NLP analysis with minimum {MIN_SENTENCES} sentences")

# Downloading the necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

logger.info("Loading JSON data")
with open('data/output_content_and_comments.json', 'r') as f:
    data = json.load(f)

# Function to perform NLP analysis on a single text
def analyze_text(text):

    sentences = sent_tokenize(text)
    if len(sentences) < MIN_SENTENCES:
        return None  # Return None for texts with less than MIN_SENTENCES sentences

    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)['compound']
    
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalnum() and word not in stop_words]
    keywords = Counter(words).most_common(5)
    
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = ' '.join([str(sentence) for sentence in summarizer(parser.document, sentences_count=min(MIN_SENTENCES, len(sentences)))])
    
    emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
    emotions = emotion_classifier(text[:512])[0]
    
    blob = TextBlob(text)
    if blob.sentiment.polarity > 0.1:
        stance = 'Positive'
    elif blob.sentiment.polarity < -0.1:
        stance = 'Negative'
    else:
        stance = 'Neutral'
    
    discourse_markers = ['however', 'although', 'in contrast', 'on the other hand', 'furthermore']
    discourse_complexity = sum(1 for sent in sentences if any(marker in sent.lower() for marker in discourse_markers))
    
    return {
        'sentiment': sentiment,
        'keywords': keywords,
        'summary': summary,
        'emotions': emotions,
        'stance': stance,
        'discourse_complexity': discourse_complexity
    }

logger.info("Preparing data for analysis")
analyzed_data = {}

for key, value in tqdm(data.items(), desc="Analyzing posts"):
    if value['source'] in ['RedditSource', 'LemmySource']:
        body_text = value['content']['body'] if value['content'] else ''
        body_analysis = analyze_text(body_text)
        
        post_data = {
            'source': value['source'],
            'timestamp': value['content']['timestamp'] if value['content'] else '',
            'title': value['content']['title'] if value['content'] else '',
            'body': {
                'text': body_text,
                'analysis': body_analysis
            },
            'comments': []
        }
        
        for comment_id, comment_text in value['comments'].items():
            comment_analysis = analyze_text(comment_text)
            if comment_analysis:  
                post_data['comments'].append({
                    'id': comment_id,
                    'text': comment_text,
                    'analysis': comment_analysis
                })
        
        if body_analysis or post_data['comments']: 
            analyzed_data[key] = post_data

outputs_folder = Path('./outputs')
outputs_folder.mkdir(exist_ok=True)

# Trend Analysis (combined for body and comments)
logger.info("Performing trend analysis")
trend_data = []
for key, value in analyzed_data.items():
    date = pd.to_datetime(value['timestamp']).date()
    trend_data.append({
        'date': date,
        'type': 'post',
        'sentiment': value['body']['analysis']['sentiment'] if value['body']['analysis'] else None
    })
    for comment in value['comments']:
        trend_data.append({
            'date': date,
            'type': 'comment',
            'sentiment': comment['analysis']['sentiment']
        })

df_trend = pd.DataFrame(trend_data)
df_trend['date'] = pd.to_datetime(df_trend['date'])
df_trend['sentiment'] = pd.to_numeric(df_trend['sentiment'], errors='coerce')

plt.figure(figsize=(12, 6))
sns.histplot(data=df_trend, x='date', hue='type', multiple='stack', bins=30)
plt.title('Number of Posts and Comments Over Time')
plt.xlabel('Date')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Type')
plt.tight_layout()
plt.savefig(outputs_folder / 'post_comment_trend.png')
plt.close()

plt.figure(figsize=(12, 6))
df_trend.groupby(['date', 'type'])['sentiment'].mean().unstack().plot(kind='line')
plt.title('Average Sentiment of Posts and Comments Over Time')
plt.xlabel('Date')
plt.ylabel('Average Sentiment')
plt.legend(title='Type')
plt.tight_layout()
plt.savefig(outputs_folder / 'sentiment_trend.png')
plt.close()

logger.info("Saving results to JSON")
with open(outputs_folder / 'nlp_analysis_results_combined.json', 'w') as f:
    json.dump(analyzed_data, f, indent=4, default=str)

logger.info("Analysis complete. Results saved in the 'outputs' folder.")

def flatten(l):
    return [item for sublist in l for item in sublist]

all_keywords = flatten([
    (post['body']['analysis']['keywords'] if post['body']['analysis'] else []) + 
    flatten([comment['analysis']['keywords'] for comment in post['comments']])
    for post in analyzed_data.values()
])

all_stances = flatten([
    ([post['body']['analysis']['stance']] if post['body']['analysis'] else []) + 
    [comment['analysis']['stance'] for comment in post['comments']]
    for post in analyzed_data.values()
])

all_discourse_complexity = flatten([
    ([post['body']['analysis']['discourse_complexity']] if post['body']['analysis'] else []) + 
    [comment['analysis']['discourse_complexity'] for comment in post['comments']]
    for post in analyzed_data.values()
])

emotion_data = defaultdict(list)
emotion_trend_data = []
for post in analyzed_data.values():
    date = pd.to_datetime(post['timestamp']).date()
    if post['body']['analysis']:
        for emotion in post['body']['analysis']['emotions']:
            emotion_data[emotion['label']].append(emotion['score'])
            emotion_trend_data.append({'date': date, 'emotion': emotion['label'], 'score': emotion['score'], 'type': 'post'})
    for comment in post['comments']:
        for emotion in comment['analysis']['emotions']:
            emotion_data[emotion['label']].append(emotion['score'])
            emotion_trend_data.append({'date': date, 'emotion': emotion['label'], 'score': emotion['score'], 'type': 'comment'})

emotion_df = pd.DataFrame({emotion: scores for emotion, scores in emotion_data.items()})
emotion_df_melted = emotion_df.melt(var_name='Emotion', value_name='Score')

plt.figure(figsize=(12, 6))
sns.violinplot(x='Emotion', y='Score', data=emotion_df_melted, hue='Emotion', legend=False)
plt.title('Distribution of Emotion Scores (Violin Plot)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(outputs_folder / 'emotion_violin_plot.png')
plt.close()

plt.figure(figsize=(12, 6))
sns.stripplot(x='Emotion', y='Score', data=emotion_df_melted, size=1, jitter=True, alpha=0.5, hue='Emotion', legend=False)
plt.title('Distribution of Emotion Scores (Strip Plot)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(outputs_folder / 'emotion_strip_plot.png')
plt.close()

# Combination of Box Plot and Strip Plot
plt.figure(figsize=(12, 6))
sns.boxplot(x='Emotion', y='Score', data=emotion_df_melted, whis=[0, 100], width=0.6, hue='Emotion', palette="vlag", legend=False)
sns.stripplot(x='Emotion', y='Score', data=emotion_df_melted, size=1, jitter=True, alpha=0.3, color=".3")
plt.title('Distribution of Emotion Scores (Box Plot with Strip Plot)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(outputs_folder / 'emotion_box_strip_plot.png')
plt.close()

# Stacked KDE Plot
plt.figure(figsize=(12, 6))
for emotion in emotion_data.keys():
    sns.kdeplot(data=emotion_df_melted[emotion_df_melted['Emotion'] == emotion], 
                x='Score', 
                label=emotion,
                fill=True, 
                alpha=0.5)
plt.title('Distribution of Emotion Scores (Stacked KDE Plot)')
plt.xlabel('Score')
plt.ylabel('Density')
plt.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(outputs_folder / 'emotion_stacked_kde_plot.png', bbox_inches='tight')
plt.close()

# Ridgeline Plot
plt.figure(figsize=(10, 8))
joyplot(
    data=emotion_df,
    colormap=plt.cm.viridis,
    title='Distribution of Emotion Scores (Ridgeline Plot)',
    labels=emotion_data.keys()
)
plt.tight_layout()
plt.savefig(outputs_folder / 'emotion_ridgeline_plot.png')
plt.close()

logger.info("Emotion distribution plots generated and saved in the 'outputs' folder.")


logger.info("Generating Word Cloud")
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(all_keywords))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Keywords')
plt.tight_layout(pad=0)
plt.savefig(outputs_folder / 'keyword_wordcloud.png')
plt.close()


logger.info("Generating Stance Distribution")
stance_counts = Counter(all_stances)
plt.figure(figsize=(10, 5))
sns.barplot(x=list(stance_counts.keys()), y=list(stance_counts.values()))
plt.title('Distribution of Stances')
plt.xlabel('Stance')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(outputs_folder / 'stance_distribution.png')
plt.close()

logger.info("Generating Discourse Complexity Distribution")
plt.figure(figsize=(12, 6))
sns.histplot(all_discourse_complexity, kde=True, bins=20)
plt.title('Distribution of Discourse Complexity')
plt.xlabel('Discourse Complexity Score')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(outputs_folder / 'discourse_complexity_distribution.png')
plt.close()


df_emotion_trend = pd.DataFrame(emotion_trend_data)
df_emotion_trend['date'] = pd.to_datetime(df_emotion_trend['date'])
df_emotion_trend['score'] = df_emotion_trend['score'].astype(float)

logger.info("Generating Emotion Trends Over Time")
emotions = df_emotion_trend['emotion'].unique()
n_emotions = len(emotions)
n_cols = 2  # adjust this to change the layout
n_rows = math.ceil(n_emotions / n_cols)

fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows), sharex=True)
fig.suptitle('Average Emotion Scores Over Time', fontsize=16)

for i, emotion in enumerate(emotions):
    row = i // n_cols
    col = i % n_cols
    ax = axs[row, col] if n_rows > 1 else axs[col]
    
    emotion_data = df_emotion_trend[df_emotion_trend['emotion'] == emotion]
    sns.lineplot(data=emotion_data, x='date', y='score', hue='type', ax=ax)
    
    ax.set_title(f'{emotion.capitalize()}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Average Score')
    ax.legend(title='Type')
    
    ax.set_ylim(0, 1)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')


if n_emotions % n_cols != 0:
    for j in range(n_emotions, n_rows * n_cols):
        fig.delaxes(axs.flatten()[j])

plt.tight_layout()
plt.savefig(outputs_folder / 'emotion_trends_over_time.png')
plt.close()

logger.info("Emotion trends over time plot generated and saved in the 'outputs' folder.")

logger.info("All visualizations generated and saved in the 'outputs' folder.")