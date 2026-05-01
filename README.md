# Real-Time Audio Classification (TinyML on ESP32-S3)

## Overview
This project implements a TinyML-based real-time audio classification system deployed on ESP32-S3 Nano (Waveshare).

## Features
- 7-class audio classification (bathroom activity sounds)
- Log-Mel Spectrogram feature extraction (16kHz)
- Lightweight CNN (~1.8K parameters)
- INT8 quantized TensorFlow Lite model
- Runs on ESP32 without PSRAM

## Model Details
- Input: 32x32x1 spectrogram
- Architecture: Depthwise Separable CNN
- Parameters: ~1.8K (~7 KB)
- Accuracy: ~95%

## Pipeline
1. Audio preprocessing (Librosa)
2. Feature extraction (Mel Spectrogram)
3. Model training (TensorFlow)
4. TFLite INT8 conversion
5. Deployment on ESP32

## Hardware
- ESP32-S3 Nano (Waveshare)

## Applications
- Elderly monitoring (bathroom activity detection)
- Edge AI / TinyML systems
