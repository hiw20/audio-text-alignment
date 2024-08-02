import os
import subprocess
import sys

def find_and_process_files(root_directory):
    for subdir, _, files in os.walk(root_directory):
        text_files = [f for f in files if f.endswith('.txt')]
        mp3_files = [f for f in files if f.endswith('.mp3')]

        for text_file in text_files:
            chapter_base = os.path.splitext(text_file)[0]
            matching_mp3 = f"{chapter_base}.mp3"

            if matching_mp3 in mp3_files:
                emissions_file = os.path.join(subdir, f"{chapter_base}_emissions.txt")
                emission_timesteps_file = os.path.join(subdir, f"{chapter_base}_emission_timesteps.txt")
                input_mp3 = os.path.join(subdir, matching_mp3)
                input_text = os.path.join(subdir, text_file)

                cmd = [
                    "python", "script_name.py",
                    "--emissions", emissions_file,
                    "--emission_timesteps", emission_timesteps_file,
                    "--input_mp3", input_mp3,
                    "--input_text", input_text
                ]

                print(f"Running command: {' '.join(cmd)}")
                subprocess.run(cmd)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_files.py <root_directory>")
        sys.exit(1)
    
    root_directory = sys.argv[1]
    find_and_process_files(root_directory)
