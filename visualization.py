import matplotlib.pyplot as plt
import io
import base64

def plot_topics(topics):
    # Create a bar chart for the extracted topics
    topic_names = [f"Topic {i+1}" for i in range(len(topics))]
    topic_values = [len(topic.split()) for topic in topics]  # Example: using the length of topics
    
    plt.figure(figsize=(10, 6))
    plt.barh(topic_names, topic_values, color='skyblue')
    plt.xlabel('Number of Words')
    plt.ylabel('Topics')
    plt.title('Topics Distribution')
    
    # Save the plot to a PNG image in memory and encode it
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

def plot_sentiment(sentiment_results):
    # Create a pie chart for sentiment analysis
    sentiments = {'positive': 0, 'neutral': 0, 'negative': 0}
    
    for _, (category, _) in sentiment_results.items():
        sentiments[category] += 1
    
    labels = sentiments.keys()
    sizes = sentiments.values()
    
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['#66b3ff', '#ffcc99', '#ff6666'], startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Sentiment Distribution')
    
    # Save the plot to a PNG image in memory and encode it
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

def plot_speaker_metadata(speaker_metadata):
    # Create a bar chart for speaker metadata (words per speaker)
    speakers = list(speaker_metadata['speaker_words'].keys())
    word_counts = [len(words) for words in speaker_metadata['speaker_words'].values()]
    
    plt.figure(figsize=(10, 6))
    plt.barh(speakers, word_counts, color='lightgreen')
    plt.xlabel('Number of Words')
    plt.ylabel('Speakers')
    plt.title('Words Per Speaker')
    
    # Save the plot to a PNG image in memory and encode it
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url
