import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pydub import AudioSegment
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from langdetect import detect
from pyannote.audio import Pipeline  # For speaker diarization
from gensim import corpora, models
from gensim.models.phrases import Phrases, Phraser
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from transcribe import transcribe_audio  # Importing the transcription function from transcribe.py
import nltk
import string
import math
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
import matplotlib.pyplot as plt
import io
import base64
from transformers import GPT2Tokenizer, GPTNeoForCausalLM

# Download NLTK data (punkt tokenizer, wordnet lemmatizer, and stopwords)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
model_name = 'EleutherAI/gpt-neo-125M'  # or 'EleutherAI/gpt-j-6B' for larger models
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)

def analyze_sentiment(text):
    """
    Analyze sentiment of the text using TextBlob.
    
    :param text: The text to analyze
    :return: Tuple containing sentiment category and polarity score
    """
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    
    if sentiment_score > 0:
        sentiment_category = 'positive'
    elif sentiment_score < 0:
        sentiment_category = 'negative'
    else:
        sentiment_category = 'neutral'
    
    return sentiment_category, sentiment_score

def extract_speaker_statements(speaker_metadata):
    """
    Extract and analyze sentiment for each speaker's statements.
    
    :param speaker_metadata: Dictionary containing speaker words
    :return: Dictionary with sentiment analysis results
    """
    sentiment_results = {}
    for speaker, words in speaker_metadata['speaker_words'].items():
        statement = ' '.join(words)
        sentiment_category, sentiment_score = analyze_sentiment(statement)
        sentiment_results[speaker] = (sentiment_category, sentiment_score)
    
    return sentiment_results

# Preprocess text: Tokenization, Stopword Removal, Lemmatization
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in string.punctuation]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

# Extract bigrams and trigrams
def get_bigrams_trigrams(tokens):
    bigram = Phrases(tokens, min_count=5, threshold=100)
    trigram = Phrases(bigram[tokens], threshold=100)
    bigram_mod = Phraser(bigram)
    trigram_mod = Phraser(trigram)
    return trigram_mod[bigram_mod[tokens]]

def extract_metadata(file_path, transcription_text, keywords=[]):
    metadata = {}
    audio = AudioSegment.from_file(file_path)
    duration = len(audio) / 1000.0  # Convert from milliseconds to seconds
    metadata['duration'] = round(duration, 2)
    num_words = len(transcription_text.split())
    metadata['num_words'] = num_words
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert bytes to MB
    metadata['file_size_mb'] = round(file_size, 2)
    metadata['sample_rate'] = audio.frame_rate
    file_format = os.path.splitext(file_path)[1].replace('.', '').upper()
    metadata['file_format'] = file_format
    try:
        language = detect(transcription_text)
        metadata['language'] = language
    except:
        metadata['language'] = 'Unknown'
    word_tokens = preprocess_text(transcription_text)
    word_count = Counter(word_tokens)
    keyword_frequencies = {keyword: word_count[keyword] for keyword in keywords}
    metadata['keyword_frequencies'] = keyword_frequencies
    return metadata

def extract_speaker_metadata(file_path, transcription_text):
    speaker_metadata = {}

    # Initialize the speaker diarization pipeline from pyannote (use your HF token here)
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="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")
    diarization = pipeline(file_path)
    
    # Calculate the total number of words and the total duration of the audio
    total_duration = diarization.get_timeline().extent().duration
    words = transcription_text.split()
    word_duration = total_duration / len(words)

    # Initialize a dictionary to store the words of each speaker
    speaker_words = {}

    # Assign words to each speaker based on their speaking time
    current_word_idx = 0
    for segment, _, label in diarization.itertracks(yield_label=True):
        speaker_words.setdefault(label, [])
        segment_duration = segment.end - segment.start
        num_words_in_segment = math.ceil(segment_duration / word_duration)

        # Extract the number of words within the segment and append to speaker's list
        for _ in range(num_words_in_segment):
            if current_word_idx < len(words):
                speaker_words[label].append(words[current_word_idx])
                current_word_idx += 1

    # Populate speaker metadata
    speaker_metadata['num_speakers'] = len(speaker_words)
    speaker_metadata['speaker_words'] = speaker_words

    return speaker_metadata

def extract_topics_lda(text):
    """
    Extract topics using LDA for a single document.
    
    :param text: The transcribed text
    :return: Topics and probabilities
    """
    # Preprocess the text
    tokens = preprocess_text(text)
    tokens = get_bigrams_trigrams(tokens)
    
    # Create a dictionary and corpus needed for LDA
    dictionary = corpora.Dictionary([tokens])
    corpus = [dictionary.doc2bow(tokens)]
    
    # Create and train the LDA model
    lda_model = LdaModel(corpus, num_topics=1, id2word=dictionary, passes=15)
    
    # Get topics and probabilities
    topics = lda_model.show_topics(formatted=False, num_topics=1)
    probabilities = lda_model[corpus[0]]
    
    # Extract just the topic words (not probabilities)
    topic_words = [word for word, prob in topics[0][1]]
    
    return topic_words, probabilities

def generate_dynamic_insights(transcription_text, speaker_metadata, topic_words, sentiment_results):
    """
    Generate dynamic insights from the transcription, speaker metadata, topics, and sentiment analysis.
    
    :param transcription_text: The transcribed text of the audio
    :param speaker_metadata: Metadata regarding the speakers and their statements
    :param topic_words: Key topic words extracted using LDA
    :param sentiment_results: Sentiment analysis results for each speaker
    :return: A list of dynamic insights based on the transcribed text
    """
    insights = []

    # Step 1: Analyzing key topics discussed
    insights.append(f"Key topics discussed: {', '.join(topic_words)}")

    # Step 2: Sentiment analysis for each speaker
    for speaker, (sentiment_category, sentiment_score) in sentiment_results.items():
        if sentiment_category == 'positive' and sentiment_score > 0.5:
            insights.append(f"{speaker} expressed a strongly positive sentiment.")
        elif sentiment_category == 'negative' and sentiment_score < -0.5:
            insights.append(f"{speaker} expressed a strongly negative sentiment.")
        elif sentiment_category == 'neutral':
            insights.append(f"{speaker} remained neutral throughout the conversation.")

    # Step 3: Identifying repetition of phrases or topics
    word_tokens = preprocess_text(transcription_text)
    word_count = Counter(word_tokens)
    repeated_words = [word for word, count in word_count.items() if count > 3]
    
    if repeated_words:
        insights.append(f"Frequently mentioned words: {', '.join(repeated_words)}")

    # Step 4: Insight on the length of speaking time and speaker contributions
    num_speakers = speaker_metadata['num_speakers']
    if num_speakers > 1:
        insights.append(f"{num_speakers} speakers were involved in the conversation.")
    
    for speaker, words in speaker_metadata['speaker_words'].items():
        insights.append(f"{speaker} contributed {len(words)} words.")
    
    # Step 5: Determine if the discussion was balanced or dominated by one speaker
    speaker_word_counts = {speaker: len(words) for speaker, words in speaker_metadata['speaker_words'].items()}
    max_speaker = max(speaker_word_counts, key=speaker_word_counts.get)
    total_words = sum(speaker_word_counts.values())
    
    if speaker_word_counts[max_speaker] / total_words > 0.5:
        insights.append(f"{max_speaker} dominated the conversation with more than 50% of the speech.")
    else:
        insights.append(f"The conversation was balanced among the speakers.")
    
    # Step 6: Customer-agent specific analysis (if applicable)
    if 'agent' in transcription_text.lower() or 'customer' in transcription_text.lower():
        # Customer satisfaction
        customer_satisfaction = None
        for speaker, (sentiment_category, sentiment_score) in sentiment_results.items():
            if 'customer' in speaker.lower():
                if sentiment_category == 'positive':
                    customer_satisfaction = "Customer seems satisfied with the service."
                elif sentiment_category == 'negative':
                    customer_satisfaction = "Customer seems dissatisfied with the service."
                else:
                    customer_satisfaction = "Customer's satisfaction is neutral or unclear."
        
        if customer_satisfaction:
            insights.append(customer_satisfaction)

        # Common complaints
        complaints_keywords = ['issue', 'problem', 'complaint', 'trouble', 'concern']
        complaints = [word for word in word_tokens if word in complaints_keywords]
        if complaints:
            insights.append(f"Customer raised the following complaints: {', '.join(complaints)}")
        else:
            insights.append("No significant complaints were detected.")

        # Suggestions for improvement
        improvement_keywords = ['improve', 'better', 'suggest', 'recommend']
        suggestions = [word for word in word_tokens if word in improvement_keywords]
        if suggestions:
            insights.append(f"Suggestions for improvement: {', '.join(suggestions)}")
        else:
            insights.append("No direct suggestions for improvement were detected.")
    
    return insights


def query_llm(prompt):
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(inputs['input_ids'], max_length=1024, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_llm_insights(transcription_text, speaker_metadata, topic_words, sentiment_results):
    context = f"""
    Transcription: {transcription_text}
    Topics: {', '.join(topic_words)}
    Sentiment Analysis: {', '.join([f'{speaker}: {sentiment_category}' for speaker, (sentiment_category, _) in sentiment_results.items()])}
    Speaker Metadata: {speaker_metadata}
    """
    insights = query_llm(context)
    return insights.split('\n')



# Visualization Functions
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
