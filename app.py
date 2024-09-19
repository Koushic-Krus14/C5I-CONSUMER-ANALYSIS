from flask import Flask, request, render_template, redirect, url_for, session
from werkzeug.utils import secure_filename
import os
from transcribe import transcribe_audio
from topic_modelling import extract_metadata, extract_speaker_metadata, extract_topics_lda, extract_speaker_statements, generate_dynamic_insights, generate_llm_insights, plot_topics, plot_sentiment, plot_speaker_metadata

app = Flask(__name__)

# Set a secret key for session management (generated using the secrets module)
app.config['SECRET_KEY'] = 'b9d8a587ad743fbf6e1043cfb92d731e'  # Secure random key

# Folder to store uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded file
        if 'file' not in request.files:
            return "No file uploaded", 400
        file = request.files['file']

        if file.filename == '':
            return "No selected file", 400

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Step 1: Transcribe the audio/video file
            transcription = transcribe_audio(file_path)
            
            # Step 2: Extract metadata
            keywords = ['speaker', 'conversation', 'audio']  # You can customize this
            metadata = extract_metadata(file_path, transcription, keywords)
            
            # Step 3: Perform speaker diarization and extract speaker metadata
            speaker_metadata = extract_speaker_metadata(file_path, transcription)
            
            # Step 4: Perform topic modeling
            topics, probabilities = extract_topics_lda(transcription)

            # Step 5: Sentiment analysis
            sentiment_results = extract_speaker_statements(speaker_metadata)
            
            # Step 6: Generate insights
            insights = generate_dynamic_insights(transcription, speaker_metadata, topics, sentiment_results)

            # Generate visualizations
            topic_plot_url = plot_topics(topics)
            sentiment_plot_url = plot_sentiment(sentiment_results)
            speaker_metadata_plot_url = plot_speaker_metadata(speaker_metadata)

            # Store results in session for access in the query route
            session['transcription'] = transcription
            session['metadata'] = metadata
            session['speaker_metadata'] = speaker_metadata
            session['topics'] = topics
            session['sentiment_results'] = sentiment_results
            session['insights'] = insights
            session['topic_plot_url'] = topic_plot_url
            session['sentiment_plot_url'] = sentiment_plot_url
            session['speaker_metadata_plot_url'] = speaker_metadata_plot_url

            return render_template('results.html', 
                                   transcription=transcription, 
                                   metadata=metadata, 
                                   speaker_metadata=speaker_metadata, 
                                   topics=topics, 
                                   sentiment_results=sentiment_results, 
                                   insights=insights, 
                                   topic_plot_url=topic_plot_url, 
                                   sentiment_plot_url=sentiment_plot_url, 
                                   speaker_metadata_plot_url=speaker_metadata_plot_url)

    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    query_text = request.form.get('query_text')
    if query_text:
        # Retrieve results from session
        transcription = session.get('transcription')
        topics = session.get('topics')
        sentiment_results = session.get('sentiment_results')
        speaker_metadata = session.get('speaker_metadata')
        # Generate LLM insights based on the query and transcription
        speaker_metadata = session.get('speaker_metadata')
        llm_insight = generate_llm_insights(query_text, transcription,topics,sentiment_results,)

        return render_template('results.html', 
                               transcription=transcription, 
                               metadata=session.get('metadata'), 
                               speaker_metadata=session.get('speaker_metadata'), 
                               topics=session.get('topics'), 
                               sentiment_results=session.get('sentiment_results'), 
                               insights=session.get('insights'), 
                               topic_plot_url=session.get('topic_plot_url'), 
                               sentiment_plot_url=session.get('sentiment_plot_url'), 
                               speaker_metadata_plot_url=session.get('speaker_metadata_plot_url'), 
                               query_text=query_text, 
                               llm_insight=llm_insight)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
