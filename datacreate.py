import speech_recognition as sr
import sounddevice as sd 
import soundfile as sf
import os
import time

# Setup Speech Recognition
r = sr.Recognizer()

# Create data directories
AUDIO_DIR = 'recordings'
TRANS_DIR = 'transcripts'

if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)

if not os.path.exists(TRANS_DIR):
    os.makedirs(TRANS_DIR)

# Save last index
last_index = 0
index_file = 'index.txt'

if os.path.exists(index_file):
    with open(index_file) as f:
        last_index = int(f.read())

# Set the number of files to record at a time
files_to_record = 50

for i in range(last_index + 1, last_index + 1 + files_to_record):

    print(f"Recording {i}")

    fs = 48000  
    seconds = 5  

    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)

    sd.wait()  
    time.sleep(1)

    # Save audio in PCM WAV format
    audio_path = os.path.join(AUDIO_DIR, f"rec_{i}.wav")
    sf.write(audio_path, recording, fs, format='WAV', subtype='PCM_16')

    try:
        # Transcribe audio file   
        with sr.AudioFile(audio_path) as source:
            audio_data = r.record(source)  
            text = r.recognize_google(audio_data)

        # Save transcript  
        trans_path = os.path.join(TRANS_DIR, f"rec_{i}.txt")
        with open(trans_path, 'w') as f:
            f.write(text)

        print("Transcript saved successfully")

    except Exception as e:
        print(f"Transcription failed. Error: {e}")
        
        # Delete the WAV file if transcription fails
        os.remove(audio_path)
        print(f"Deleted {audio_path}")

# Update last index
with open(index_file, 'w') as f:
    f.write(str(i))

print(f"{files_to_record} Recordings collected")
