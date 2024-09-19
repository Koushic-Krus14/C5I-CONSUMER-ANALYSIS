# Import necessary functions from submodules within the app package
from .transcribe import transcribe_audio
from .topic_modeling import extract_topics
from .sentiment_analysis import analyze_sentiment

def process_file(file_path):
    """
    This function processes a given audio or video file by transcribing it, extracting topics,
    and performing sentiment analysis.
    
    :param file_path: Path to the audio or video file
    :return: Dictionary containing transcription, topics, and sentiment analysis
    """
    # Step 1: Transcribe the file (audio or video)
    print("Starting transcription...")
    transcription = transcribe_audio(file_path)
    print(f"Transcription complete: {transcription[:100]}...")  # Print the first 100 characters of transcription
    
    # Step 2: Extract topics from the transcription text
    print("Extracting topics...")
    topics = extract_topics(transcription)
    print(f"Topics extracted: {topics}")
    
    # Step 3: Analyze sentiment from the transcription text
    print("Analyzing sentiment...")
    sentiment = analyze_sentiment(transcription)
    print(f"Sentiment analysis complete: {sentiment}")
    
    # Return a dictionary of the results
    return {
        "transcription": transcription,
        "topics": topics,
        "sentiment": sentiment
    }
