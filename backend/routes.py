from fastapi import APIRouter, File, UploadFile, HTTPException

from audio_analyzer import AudioAnalyzer
from config import logger

router = APIRouter()
analyzer = AudioAnalyzer()

@router.get("/")
async def root():
    return {"message": "Audio Analysis API is running!"}

@router.post("/analyze-audio")
async def analyze_audio(audio: UploadFile = File(...)):
    if not audio.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")

    try:
        content = await audio.read()
        logger.info(f"Read {len(content)} bytes from uploaded audio file: {audio.filename}")

        audio_data, sr = analyzer.load_audio(content, audio.filename)
        duration = len(audio_data) / sr
        logger.info(f"Audio loaded successfully: {duration:.2f}s duration, {sr}Hz sample rate")

        features = analyzer.extract_features(audio_data)
        logger.info(f"Extracted {len(features)} features from audio")

        sound_type = analyzer.classify_sound(features)
        logger.info(f"Classified audio as: {sound_type}")

        waveform_image = analyzer.generate_waveform(audio_data, sr)
        spectrogram_image = analyzer.generate_spectrogram(audio_data, sr)
        logger.info("Generated visualizations successfully")

        description = await analyzer.generate_description(sound_type, features)
        logger.info("Generated AI description successfully")

        return {
            "waveformImage": waveform_image,
            "spectrogramImage": spectrogram_image,
            "soundType": sound_type,
            "description": description,
            "duration": duration,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analyze_audio endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


@router.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Audio Analysis API is running"}


