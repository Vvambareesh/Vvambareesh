from pydub import AudioSegment
import os

# Set the full path to the FFmpeg executable
ffmpeg_path = r"C:\path\to\ffmpeg.exe"  # Replace with the actual path

# Set the FFmpeg executable path in the AudioSegment module
AudioSegment.ffmpeg = ffmpeg_path

def combine_and_convert_aif_to_ogg(input_folder, output_folder, segment_duration=15 * 60 * 1000):
    # Rest of the script remains unchanged
    # Ensure proper indentation for the function body
    for i in range(0, len(aiff_files), segment_duration):
        # Rest of the loop body and the script

if __name__ == "__main__":
    input_folder = r"E:\ai\New folder"
    output_folder = r"E:\ai\outputaudio"

    combine_and_convert_aif_to_ogg(input_folder, output_folder)
