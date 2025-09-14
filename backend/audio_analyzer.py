import io
import base64
import logging
import wave
from typing import Tuple
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend

from fastapi import HTTPException

from config import openai_api_key, logger
import openai


class AudioAnalyzer:
    def __init__(self):
        self.sample_rate = 22050
        self.n_mfcc = 13
        self.n_fft = 2048
        self.hop_length = 512

    def load_audio(self, audio_bytes: bytes, filename: str) -> Tuple[np.ndarray, int]:
        try:
            buffer = io.BytesIO(audio_bytes)
            audio, sr = librosa.load(buffer, sr=self.sample_rate)
            logger.info(f"Successfully loaded audio with librosa: {len(audio)} samples, {sr}Hz")
            return audio, sr
        except Exception as e:
            logger.warning(f"Librosa direct loading failed: {e}")

            if filename.lower().endswith('.wav'):
                try:
                    return self._load_wav_manually(audio_bytes)
                except Exception as e2:
                    logger.error(f"Manual WAV loading failed: {e2}")

            try:
                return self._resample_audio(audio_bytes, filename)
            except Exception as e3:
                logger.error(f"All audio loading methods failed: {e3}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Error loading audio: {str(e)}. Please try with a WAV file."
                )

    def _load_wav_manually(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        try:
            with wave.open(io.BytesIO(audio_bytes), 'rb') as wav_file:
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frame_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()

                raw_data = wav_file.readframes(n_frames)

                if sample_width == 1:
                    dtype = np.uint8
                    audio_data = np.frombuffer(raw_data, dtype=dtype)
                    audio_data = (audio_data - 128) / 128.0
                elif sample_width == 2:
                    dtype = np.int16
                    audio_data = np.frombuffer(raw_data, dtype=dtype)
                    audio_data = audio_data / 32768.0
                else:
                    raise ValueError(f"Unsupported sample width: {sample_width}")

                if n_channels > 1:
                    audio_data = audio_data.reshape(-1, n_channels)
                    audio_data = np.mean(audio_data, axis=1)

                if frame_rate != self.sample_rate:
                    audio_data = librosa.resample(audio_data, orig_sr=frame_rate, target_sr=self.sample_rate)

                return audio_data, self.sample_rate

        except Exception as e:
            logger.error(f"Manual WAV loading failed: {e}")
            raise

    def _resample_audio(self, audio_bytes: bytes, filename: str) -> Tuple[np.ndarray, int]:
        try:
            buffer = io.BytesIO(audio_bytes)
            audio, sr = librosa.load(buffer, sr=self.sample_rate, mono=True)
            return audio, sr
        except Exception as e:
            logger.error(f"Resampling failed: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio format. Please try with a WAV, MP3, or FLAC file."
            )

    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        features = []
        if len(audio) == 0:
            return np.zeros(40)

        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc)
        features.extend(np.mean(mfccs, axis=1))
        features.extend(np.std(mfccs, axis=1))

        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        features.append(np.mean(spectral_centroids))
        features.append(np.std(spectral_centroids))

        zcr = librosa.feature.zero_crossing_rate(audio)
        features.append(np.mean(zcr))
        features.append(np.std(zcr))

        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
        features.append(np.mean(rolloff))
        features.append(np.std(rolloff))

        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
        features.extend(np.mean(chroma, axis=1))

        try:
            tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            features.append(tempo)
        except Exception:
            features.append(0)

        while len(features) < 40:
            features.append(0)

        return np.array(features[:40])

    def classify_sound(self, features: np.ndarray) -> str:
        if len(features) < 29:
            return "unknown"

        spectral_centroid = features[26] if len(features) > 26 else 0
        zcr = features[28] if len(features) > 28 else 0
        tempo = features[-1] if len(features) > 0 else 0

        if zcr > 0.1 and spectral_centroid > 2000:
            return "speech"
        elif tempo > 60 and spectral_centroid < 3000:
            return "music"
        else:
            return "noise"

    def generate_waveform(self, audio: np.ndarray, sr: int) -> str:
        plt.figure(figsize=(12, 4))
        max_points = 10000
        if len(audio) > max_points:
            step = len(audio) // max_points
            audio = audio[::step]

        time = np.linspace(0, len(audio) / sr, len(audio))
        plt.plot(time, audio)
        plt.title('Audio Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return image_base64

    def generate_spectrogram(self, audio: np.ndarray, sr: int) -> str:
        plt.figure(figsize=(12, 6))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)

        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Audio Spectrogram')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return image_base64

    async def generate_description(self, sound_type: str, features: np.ndarray) -> str:
        if not openai_api_key:
            logger.warning("No OpenAI API key found, returning fallback description")
            return (
                f"This audio clip has been classified as {sound_type}. The analysis shows various "
                f"audio characteristics including spectral features and temporal patterns."
            )
        try:
            prompt = (
                f"Describe this sound: it is detected as {sound_type}. Please provide a friendly, "
                f"conversational description of what this audio might sound like to a human listener. "
                f"Keep it under 100 words and make it engaging."
            )

            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert audio analyst who provides clear, friendly descriptions of audio content."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=150,
                temperature=0.7,
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return (
                f"This audio clip has been classified as {sound_type}. The analysis reveals various "
                f"audio characteristics that suggest this type of sound content."
            )


