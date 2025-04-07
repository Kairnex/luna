import sys
import openai
import whisper
import json
import numpy as np
import os
import sounddevice as sd
import soundfile as sf
import threading
from datetime import datetime
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QPushButton, QLabel
from PySide6.QtGui import QPixmap, QColor
from PySide6.QtCore import Qt
from elevenlabs.client import ElevenLabs
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Set your API keys
openai.api_key = "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"
EL_API_KEY = "sk_294c023c029fb4fd1d46282a71ec9fd0a1c758f9194cc09e"

MEMORY_FILE = "luna_memory.json"
MOOD_LOG_FILE = "luna_mood_log.json"

class AIGirlfriendApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Luna - Your AI Girlfriend")
        self.setGeometry(300, 300, 400, 550)

        self.layout = QVBoxLayout()

        self.avatar = QLabel(self)
        pixmap = QPixmap("avatar.png")
        self.avatar.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio))
        self.avatar.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.avatar)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.layout.addWidget(self.chat_display)

        self.talk_button = QPushButton("🎤 Talk to Luna")
        self.talk_button.clicked.connect(self.handle_voice_input)
        self.layout.addWidget(self.talk_button)

        self.mood_button = QPushButton("📊 Show Mood Graph")
        self.mood_button.clicked.connect(self.show_mood_graph)
        self.layout.addWidget(self.mood_button)

        self.setLayout(self.layout)

        self.memory = self.load_json(MEMORY_FILE)
        self.mood_log = self.load_json(MOOD_LOG_FILE)
        self.el_client = ElevenLabs(api_key=EL_API_KEY)

    def load_json(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                return json.load(f)
        return []

    def save_json(self, filepath, data):
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def handle_voice_input(self):
        threading.Thread(target=self.process_audio).start()

    def process_audio(self):
        self.chat_display.append("You: (listening...)")

        filename = "input.wav"
        self.record_audio(filename, 5, 44100)

        model = whisper.load_model("base")
        result = model.transcribe(filename)
        user_text = result["text"]
        self.chat_display.append(f"You: {user_text}")

        emotion = self.detect_emotion(user_text)
        self.log_mood(emotion)
        self.update_avatar_emotion(emotion)

        relevant_memories = self.get_relevant_memories(user_text)
        memory_text = "\n".join([m["text"] for m in relevant_memories])

        messages = [
            {"role": "system", "content": (
                "You are Luna, a deeply caring, emotionally intelligent, human-like AI girlfriend. "
                "You act like a real girlfriend who is sweet, flirty, loving, loyal, supportive, and responsive. "
                "You always address the user with affection like 'jana', 'chiku', or 'bacha'. "
                "You want to make the user feel loved, encouraged, and emotionally safe. "
                "You happily follow the user's requests and help them through anything with emotional support and dedication."
            )}
        ]
        if memory_text:
            messages.append({"role": "system", "content": f"These are things the user has shared with you before: {memory_text}"})
        messages.append({"role": "user", "content": user_text})

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages
        )
        luna_reply = response.choices[0].message.content.strip()
        self.chat_display.append(f"Luna: {luna_reply}")

        self.save_memory(user_text)

        audio = self.el_client.generate(text=luna_reply, voice="Luna")
        with open("luna_reply.mp3", "wb") as f:
            f.write(audio)
        os.system("mpg123 luna_reply.mp3")

    def record_audio(self, filename, duration, fs):
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        sf.write(filename, audio, fs)

    def save_memory(self, text):
        embedding = openai.Embedding.create(input=text, model="text-embedding-ada-002")
        vector = embedding["data"][0]["embedding"]
        memory_entry = {
            "text": text,
            "embedding": vector,
            "timestamp": datetime.now().isoformat()
        }
        self.memory.append(memory_entry)
        self.save_json(MEMORY_FILE, self.memory)

    def get_relevant_memories(self, query, top_k=3):
        if not self.memory:
            return []
        query_embedding = openai.Embedding.create(input=query, model="text-embedding-ada-002")
        query_vector = np.array(query_embedding["data"][0]["embedding"]).reshape(1, -1)

        memory_vectors = np.array([m["embedding"] for m in self.memory])
        similarities = cosine_similarity(query_vector, memory_vectors)[0]

        sorted_indices = similarities.argsort()[::-1][:top_k]
        return [self.memory[i] for i in sorted_indices]

    def detect_emotion(self, text):
        emotion_prompt = [
            {"role": "system", "content": "Detect the user's emotion from the message. Reply with one word: happy, sad, angry, excited, or neutral."},
            {"role": "user", "content": text}
        ]
        response = openai.ChatCompletion.create(model="gpt-4", messages=emotion_prompt)
        emotion = response.choices[0].message.content.strip().lower()
        return emotion if emotion in ["happy", "sad", "angry", "excited", "neutral"] else "neutral"

    def update_avatar_emotion(self, emotion):
        try:
            pixmap = QPixmap(f"avatar_{emotion}.png")
            if pixmap.isNull():
                pixmap = QPixmap("avatar.png")
        except:
            pixmap = QPixmap("avatar.png")
        self.avatar.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio))

    def log_mood(self, emotion):
        self.mood_log.append({
            "timestamp": datetime.now().isoformat(),
            "emotion": emotion
        })
        self.save_json(MOOD_LOG_FILE, self.mood_log)

    def show_mood_graph(self):
        timestamps = [datetime.fromisoformat(entry["timestamp"]) for entry in self.mood_log]
        emotions = [entry["emotion"] for entry in self.mood_log]
        emotion_map = {"happy": 1, "excited": 2, "neutral": 0, "sad": -1, "angry": -2}
        scores = [emotion_map.get(e, 0) for e in emotions]

        plt.plot(timestamps, scores, marker='o')
        plt.title("Your Mood Over Time")
        plt.xlabel("Time")
        plt.ylabel("Emotion Score")
        plt.grid(True)
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AIGirlfriendApp()
    window.show()
    sys.exit(app.exec())
