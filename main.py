import subprocess
import os
from pydub import AudioSegment
from openai import OpenAI
import keys
import uuid
from tqdm import tqdm
client = OpenAI(api_key=keys.openai_key)

def video_to_text(video_path):
    temp_name = str(uuid.uuid4()).split("-")[0]
    # Extract audio from video
    output_file1 = f"/tmp/{temp_name}.ac3"
    command = f"ffmpeg -i \"{video_path}\" -map 0:a -acodec copy \"{output_file1}\""
    subprocess.run(command, shell=True)

    output_file2 = f"/tmp/{temp_name}.mp3"
    command = f"ffmpeg -i \"{output_file1}\" -vn -ar 24000 -ac 2 -b:a 192k \"{output_file2}\""
    subprocess.run(command, shell=True)

    print("> Audio extracted")

    # chunk audio
    os.makedirs(f"/tmp/{temp_name}", exist_ok=True)
    audio = AudioSegment.from_mp3(output_file2)
    chunk_length = 10*60*1000
    chunks = [audio[i:i+chunk_length] for i in range(0, len(audio), chunk_length)]

    for i, chunk in enumerate(chunks):
        chunk.export(f"/tmp/{temp_name}/{i}.mp3", format="mp3")
    
    print("> Audio chunked")

    buffer = ""
    for i, chunk in tqdm(enumerate(chunks), desc="Transcribing audio chunks", total=len(chunks)):
        with open(f"/tmp/{temp_name}/{i}.mp3", "rb") as f:
            transcript = client.audio.transcriptions.create( model="whisper-1", file=f, response_format="text")
            buffer += transcript + " "

    # clean up temp files
    os.remove(output_file1)
    os.remove(output_file2)
    for i in range(len(chunks)):
        os.remove(f"/tmp/{temp_name}/{i}.mp3")
    os.rmdir(f"/tmp/{temp_name}")

    return buffer

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # all mp4 files in the current directory
    videos = [f for f in os.listdir() if f.endswith(".mp4")]
    for video in videos:
        if os.path.exists(f"{video}_transcript.txt"):
            print(f"Transcript for {video} already exists")
            continue
        print(f"Transcribing {video}")
        text = video_to_text(video)
        with open(f"{video}_transcript.txt", "w") as f:
            f.write(text)
        print("\n\n")
