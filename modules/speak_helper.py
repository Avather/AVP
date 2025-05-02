import subprocess

def speak(text):
    """Uses macOS 'say' command to provide speech output."""
    try:
        subprocess.Popen(["say", text])
    except Exception as e:
        print(f"[Speech Error] {e}")

