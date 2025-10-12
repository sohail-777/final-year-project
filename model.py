from transformers import pipeline

# Load the sentiment analysis model from Hugging Face
sentiment_pipeline = pipeline(
    "text-classification", model="tabularisai/multilingual-sentiment-analysis")


def analyze_sentiment(texts):
    results = sentiment_pipeline(texts)
    sentiment_counts = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}

    for result in results:
        label = result['label'].upper()
        if 'NEG' in label:
            sentiment_counts['NEGATIVE'] += 1
        elif 'NEU' in label:
            sentiment_counts['NEUTRAL'] += 1
        else:
            sentiment_counts['POSITIVE'] += 1

    total = sum(sentiment_counts.values())
    overall = max(sentiment_counts, key=sentiment_counts.get)

    return sentiment_counts, overall
