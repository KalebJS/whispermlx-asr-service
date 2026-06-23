#!/bin/bash
# test-api.sh - Smoke test script for the native whispermlx ASR API
#
# Usage:
#   ./test-api.sh                              # default: localhost:9001
#   ./test-api.sh 127.0.0.1 9001 path/to/audio.wav
#
# The service runs natively (uvicorn on port 9001, no Docker, no Ray).

# Configuration
HOST=${1:-"localhost"}
PORT=${2:-"9001"}
BASE_URL="http://${HOST}:${PORT}"

echo "========================================"
echo "Whispermlx ASR API Test Script"
echo "========================================"
echo "Testing endpoint: ${BASE_URL}"
echo ""

# Test 1: Health Check
echo "Test 1: Health Check"
echo "--------------------"
response=$(curl -s "${BASE_URL}/health")
if [ $? -eq 0 ]; then
    echo "✓ Health check successful"
    echo "Response: ${response}"
else
    echo "✗ Health check failed"
    echo "Error: Cannot connect to ${BASE_URL}"
    exit 1
fi
echo ""

# Test 2: Root Endpoint
echo "Test 2: Root Endpoint"
echo "--------------------"
response=$(curl -s "${BASE_URL}/")
if [ $? -eq 0 ]; then
    echo "✓ Root endpoint successful"
    echo "Response: ${response}"
else
    echo "✗ Root endpoint failed"
fi
echo ""

# Test 3: API Documentation
echo "Test 3: API Documentation"
echo "------------------------"
echo "Visit ${BASE_URL}/docs for interactive API documentation"
echo ""

# Test 4: Sample Transcription (if audio file provided)
if [ -f "$3" ]; then
    echo "Test 4: Sample Transcription"
    echo "---------------------------"
    echo "Testing with file: $3"

    response=$(curl -s -X POST "${BASE_URL}/asr" \
        -F "audio_file=@$3" \
        -F "language=en" \
        -F "model=tiny" \
        -F "output_format=json" \
        -F "diarize=false")

    if [ $? -eq 0 ]; then
        echo "✓ Transcription successful"
        echo "Response preview:"
        echo "${response}" | head -c 500
        echo "..."
    else
        echo "✗ Transcription failed"
        echo "Error: ${response}"
    fi
else
    echo "Test 4: Sample Transcription"
    echo "---------------------------"
    echo "⚠ Skipped - No audio file provided"
    echo "Usage: $0 [host] [port] [audio_file.wav]"
fi
echo ""

echo "========================================"
echo "Test Summary"
echo "========================================"
echo "Endpoint: ${BASE_URL}"
echo "All basic tests completed"
echo ""
echo "To test transcription, run:"
echo "  $0 ${HOST} ${PORT} path/to/audio.mp3"
echo ""
