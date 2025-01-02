from flask import Flask, request, render_template, jsonify, url_for, send_from_directory
from dotenv import load_dotenv
import os
import subprocess
from pyannote.audio import Pipeline
from pyannote.audio import Audio
import whisper
import openai
import json

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Flask app initialization
app = Flask(__name__)

# Define folder for uploaded audios
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/uploads/<filename>')
def uploads(filename):
    """Serve uploaded audio files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


audio = Audio()

token = os.getenv("PYANNOTE_AUTH_TOKEN")
# Load pre-trained speaker diarization pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)

def analyze_with_openai(text):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """
                    You are a content reviewer assistant identifying violations in text regarding royalty, religion, and race. 
                    Additionally, you perform sentiment analysis on the provided text. Then, you also detect any curse words in the provided text.
                    Return the summary of the text, sentiment analysis, and information about violations in the defined JSON format and categories only:
                    {
                        "Summary": "",
                        "Sentiment": "", // positive, neutral or negative
                        "Offensive Language": "", //sentence with offensive language
                        "Violations": true/false,
                        "Race": [
                            {"sentence": "",
                            "severity": "",
                            "reasoning": ""} 
                        ],
                        "Religion": [
                            {"sentence": "",
                            "severity": "",
                            "reasoning": ""} 
                        ],
                        "Royalty": [
                            {"sentence": "",
                            "severity": "",
                            "reasoning": ""} 
                        ]
                    }
                """
            },
            {
                "role": "user",
                "content": text
            }
        ]
    )
    return response.choices[0].message['content']

def save_results_to_file(audio_name, num_speaker, transcript, openai_results):
    # Create a file name based on the audio name (with a timestamp or other unique identifier)
    output_filename = f"results_{audio_name.replace('.wav', '')}.txt"

    # Open the file in append mode to avoid overwriting previous results
    with open(output_filename, 'a') as f:
        # Write results to the file
        f.write(f"Audio File: {audio_name}\n")
        f.write(f"Number of Speakers: {num_speaker}\n")
        f.write(f"Transcript:\n{transcript}\n")
        f.write(f"\nSummary:\n{openai_results['summary']}\n")
        f.write(f"Sentiment: {openai_results['sentiment']}\n")
        f.write(f"Violations: {openai_results['violations']}\n")
        f.write(f"Offensive Language: {openai_results['offensive']}\n")
        f.write(f"Race Violations: {openai_results['race']}\n")
        f.write(f"Religion Violations: {openai_results['religion']}\n")
        f.write(f"Royalty Violations: {openai_results['royalty']}\n")
        f.write("="*50 + "\n\n")

# Update the transcribe_and_segmentize function to call save_results_to_file
def transcribe_and_segmentize(path, language, model_size, audio_name): 
    # Step 1: Convert audio to a compatible format (if not already)
    subprocess.call(["ffmpeg", '-i', str(path), 'audio.wav', '-y'])
    path = 'audio.wav'

    # Step 2: Transcribe the audio with Whisper
    model = whisper.load_model(model_size)
    result = model.transcribe(path, fp16=False, language=language)
    segments = result.get("segments", [])
    transcript_text = result.get("text", "")

    # Step 3: Run PyAnnote's speaker diarization
    diarization = pipeline({"audio": path})

    # Step 4: Assign speakers to Whisper segments
    speaker_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })

    # Step 5: Map PyAnnote speaker IDs to friendly names
    speaker_mapping = {}
    speaker_counter = 1
    for speaker_segment in speaker_segments:
        speaker_id = speaker_segment["speaker"]
        if speaker_id not in speaker_mapping:
            speaker_mapping[speaker_id] = f"Speaker {speaker_counter}"
            speaker_counter += 1

    # Step 6: Align Whisper transcription with diarization
    for segment in segments:
        segment_start = segment.get("start", 0.0)
        segment_end = segment.get("end", 0.0)

        overlaps = [
            (speaker_segment, min(segment_end, speaker_segment["end"]) - max(segment_start, speaker_segment["start"]))
            for speaker_segment in speaker_segments
            if speaker_segment["start"] < segment_end and speaker_segment["end"] > segment_start
        ]

        if overlaps:
            # Choose the speaker with the maximum overlap
            best_match = max(overlaps, key=lambda x: x[1])[0]
            segment["speaker"] = speaker_mapping.get(best_match["speaker"], "Unknown")
        else:
            segment["speaker"] = "Unknown"

    # Step 7: Combine results into a readable format
    transcript_text_with_speakers = ""
    for segment in segments:
        speaker = segment.get("speaker", "Unknown")
        start = segment.get("start", 0.0)
        end = segment.get("end", 0.0)
        text = segment.get("text", "").strip()
        transcript_text_with_speakers += f"{speaker} ({start}s - {end}s): {text}\n"

    # Count number of speakers in transcription (ignoring diarization overlaps)
    num_speaker = len(set(segment.get("speaker") for segment in segments))

    # Step 8: Perform OpenAI analysis on the transcribed text
    offensive_sentences = []
    for segment in segments:
        start_time = segment.get("start", 0.0)
        end_time = segment.get("end", 0.0)
        text = segment.get("text", "").strip()

        # Analyze each segment for offensive language
        openai_results_segment = analyze_with_openai(text)
        
        try:
            openai_results_json = json.loads(openai_results_segment)
            offensive = openai_results_json.get("Offensive Language", "")
            if offensive:
                offensive_sentences.append({
                    "text": offensive,
                    "start_time": start_time,
                    "end_time": end_time
                })
        except json.JSONDecodeError:
            continue

    # Combine offensive sentences into a readable format
    offensive_results = "\n".join(
        [f"({sentence['start_time']}s - {sentence['end_time']}s): {sentence['text']}" for sentence in offensive_sentences]
    )

    # Step 9: Parse OpenAI results for the full transcript
    try:
        openai_results = analyze_with_openai(transcript_text.strip())
        openai_results_json = json.loads(openai_results)
        summary = openai_results_json.get("Summary", "")
        sentiment = openai_results_json.get("Sentiment", "")
        violations = openai_results_json.get("Violations", False)
        race_violations = openai_results_json.get("Race", [])
        religion_violations = openai_results_json.get("Religion", [])
        royalty_violations = openai_results_json.get("Royalty", [])
    except json.JSONDecodeError:
        summary = "Error parsing OpenAI response."
        sentiment = "Error parsing OpenAI response."
        violations = False
        race_violations = []
        religion_violations = []
        royalty_violations = []

    # Step 10: Save results to file with the correct filename
    save_results_to_file(
        audio_name, num_speaker, transcript_text_with_speakers, {
            "summary": summary,
            "sentiment": sentiment,
            "violations": violations,
            "offensive": offensive_results,
            "race": race_violations,
            "religion": religion_violations,
            "royalty": royalty_violations
        }
    )

    # Step 11: Return results
    return transcript_text_with_speakers, {
        "summary": summary,
        "sentiment": sentiment,
        "violations": violations,
        "offensive": offensive_results,
        "race": race_violations,
        "religion": religion_violations,
        "royalty": royalty_violations,
        "num_speaker": num_speaker
    }

# Flask route for uploading audio and getting results
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle existing uploaded audio or new upload
        uploaded_audio = request.form.get("uploaded_audio")
        if uploaded_audio:
            # If already uploaded, use the existing audio file
            audio_file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_audio)
        else:
            # Handle new audio upload
            audio_file = request.files.get("audio")
            if not audio_file:
                return jsonify({"error": "No audio file provided."}), 400

            audio_file_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)
            audio_file.save(audio_file_path)

        # Get language and model size
        language = request.form.get("language", "en")
        model_size = request.form.get("model_size", "base")

        try:
            # Transcription and analysis
            transcript, openai_results = transcribe_and_segmentize(
                audio_file_path, language, model_size, audio_file.filename  
            )

            # Render results and audio player
            return render_template(
                "full-index.html",
                audio_name=os.path.basename(audio_file_path),
                transcript=transcript,
                summary=openai_results["summary"],
                sentiment=openai_results["sentiment"],
                violations=openai_results["violations"],
                offensive=openai_results["offensive"],
                race_violations=openai_results["race"],
                religion_violations=openai_results["religion"],
                royalty_violations=openai_results["royalty"],
                num_speaker=openai_results["num_speaker"],
                audio_url=url_for("uploads", filename=os.path.basename(audio_file_path)),
                language=language,
                model_size=model_size,
            )

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Render empty form for GET request
    return render_template("full-index.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)