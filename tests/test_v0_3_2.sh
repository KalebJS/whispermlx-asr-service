#!/usr/bin/env bash
# Smoke tests for the native MLX (whispermlx) ASR service.
#
# Covers:
#   - Health check (/health, serve_mode=simple)
#   - Prometheus /metrics (OpenMetrics text, MLX metrics, no CUDA)
#   - /queue-metrics JSON
#   - /v1/models built from the MLX model map (no faster_whisper, no distil-*)
#   - /asr alias resolution (whisper-tiny → tiny)
#   - Prometheus counters increment after /asr requests
#   - service_info labels (version/device/serve_mode)
#   - VRAM gauge (MLX active memory or 0, no CUDA)
#
# Usage:
#   ./tests/test_v0_3_2.sh                     # default base URL + audio
#   BASE_URL=http://host:9001 ./tests/test_v0_3_2.sh
#   AUDIO=tests/testfiles/sample.wav ./tests/test_v0_3_2.sh

set -uo pipefail

BASE_URL="${BASE_URL:-http://localhost:9001}"
AUDIO="${AUDIO:-tests/testfiles/sample.wav}"
PASS=0
FAIL=0

color() {
    case "$1" in
        red)    printf '\033[31m%s\033[0m' "$2" ;;
        green)  printf '\033[32m%s\033[0m' "$2" ;;
        yellow) printf '\033[33m%s\033[0m' "$2" ;;
        *) printf '%s' "$2" ;;
    esac
}

step() { echo; echo "=== $1 ==="; }
ok()   { color green "  PASS"; echo " $1"; PASS=$((PASS+1)); }
bad()  { color red   "  FAIL"; echo " $1"; FAIL=$((FAIL+1)); }
note() { color yellow "  NOTE"; echo " $1"; }

if [ ! -f "$AUDIO" ]; then
    echo "Audio file not found: $AUDIO"
    echo "Set AUDIO=path/to/file.wav or place a file at tests/testfiles/sample.wav"
    exit 2
fi

step "Health check"
HEALTH=$(curl -fsS "${BASE_URL}/health")
if echo "$HEALTH" | grep -q '"status":"healthy"'; then
    ok "/health returns healthy"
    echo "    $HEALTH"
else
    bad "/health did not return healthy: $HEALTH"
    exit 1
fi

if echo "$HEALTH" | grep -q '"serve_mode":"simple"'; then
    ok "serve_mode is 'simple' (no Ray)"
else
    bad "serve_mode is not 'simple': $HEALTH"
fi

step "GET /metrics returns Prometheus OpenMetrics text (not JSON)"
METRICS=$(curl -fsS "${BASE_URL}/metrics")
if echo "$METRICS" | head -1 | grep -q '^# HELP'; then
    ok "/metrics starts with '# HELP' (OpenMetrics text)"
else
    bad "/metrics does not start with '# HELP'. First line: $(echo "$METRICS" | head -1)"
fi

for metric in whisperx_requests_total whisperx_request_duration_seconds \
              whisperx_active_transcriptions whisperx_loaded_models \
              whisperx_audio_duration_seconds whisperx_audio_size_megabytes \
              whisperx_vram_allocated_bytes whisperx_service_info; do
    if echo "$METRICS" | grep -q "^# HELP ${metric} "; then
        ok "metric ${metric} is registered"
    else
        bad "metric ${metric} missing"
    fi
done

step "GET /queue-metrics returns JSON"
QM=$(curl -fsS "${BASE_URL}/queue-metrics")
if echo "$QM" | grep -q '"serve_mode"'; then
    ok "/queue-metrics returns the legacy JSON shape"
    echo "    $QM"
else
    bad "/queue-metrics did not return expected JSON: $QM"
fi

step "GET /v1/models is sourced from the MLX model map (no faster_whisper)"
MODELS_JSON=$(curl -fsS "${BASE_URL}/v1/models")
MODEL_COUNT=$(echo "$MODELS_JSON" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['data']))")
if [ "$MODEL_COUNT" -ge 15 ]; then
    ok "/v1/models lists ${MODEL_COUNT} models (expected 15+)"
else
    bad "/v1/models lists only ${MODEL_COUNT} models (expected 15+)"
fi
if echo "$MODELS_JSON" | grep -q '"id":"whisper-1"'; then
    ok "/v1/models includes whisper-1 alias"
else
    bad "/v1/models is missing the whisper-1 alias"
fi

# MLX canonical models that should be present
for canonical in tiny base small medium large-v3 large-v3-turbo; do
    if echo "$MODELS_JSON" | grep -q "\"id\":\"${canonical}\""; then
        ok "/v1/models lists MLX canonical name '${canonical}'"
    else
        bad "/v1/models is missing MLX canonical name '${canonical}'"
    fi
done

# distil-* models must NOT be present (faster-whisper only)
if echo "$MODELS_JSON" | grep -q '"id":"distil-'; then
    bad "/v1/models contains distil-* (faster-whisper-only) models"
else
    ok "/v1/models has no distil-* models"
fi

step "Snapshot baseline /metrics counters before /asr request"
BASELINE=$(curl -fsS "${BASE_URL}/metrics")
BASELINE_OK_COUNT=$(echo "$BASELINE" \
    | grep -E '^whisperx_requests_total\{endpoint="/asr",status="ok"\} ' \
    | awk '{print $2}')
BASELINE_OK_COUNT=${BASELINE_OK_COUNT:-0}
echo "    baseline whisperx_requests_total{status=ok} = $BASELINE_OK_COUNT"

step "POST /asr with model=whisper-tiny (alias resolution)"
ASR_OUT=$(curl -fsS -X POST \
    "${BASE_URL}/asr?model=whisper-tiny&output_format=text&diarize=false" \
    -F "audio_file=@${AUDIO}")
if echo "$ASR_OUT" | grep -q '"text"'; then
    ok "/asr accepted whisper-tiny alias and returned a transcription"
    echo "    snippet: $(echo "$ASR_OUT" | head -c 150)..."
else
    bad "/asr did not return text. Output: $ASR_OUT"
fi

step "POST /asr with canonical model=tiny"
ASR_OUT2=$(curl -fsS -X POST \
    "${BASE_URL}/asr?model=tiny&output_format=text&diarize=false" \
    -F "audio_file=@${AUDIO}")
if echo "$ASR_OUT2" | grep -q '"text"'; then
    ok "/asr accepted canonical tiny and returned a transcription"
else
    bad "/asr did not return text for canonical tiny. Output: $ASR_OUT2"
fi

step "Verify Prometheus counters incremented after /asr requests"
sleep 1
AFTER=$(curl -fsS "${BASE_URL}/metrics")

AFTER_OK_COUNT=$(echo "$AFTER" \
    | grep -E '^whisperx_requests_total\{endpoint="/asr",status="ok"\} ' \
    | awk '{print $2}')
AFTER_OK_COUNT=${AFTER_OK_COUNT:-0}

INCREASED=$(python3 -c "print(int(float('${AFTER_OK_COUNT}') > float('${BASELINE_OK_COUNT}')))")
if [ "$INCREASED" = "1" ]; then
    ok "whisperx_requests_total{status=ok} increased: ${BASELINE_OK_COUNT} -> ${AFTER_OK_COUNT}"
else
    bad "whisperx_requests_total{status=ok} did not increase: ${BASELINE_OK_COUNT} -> ${AFTER_OK_COUNT}"
fi

DUR_COUNT=$(echo "$AFTER" \
    | grep -E '^whisperx_request_duration_seconds_count\{endpoint="/asr"\} ' \
    | awk '{print $2}')
if [ -n "$DUR_COUNT" ] && [ "$(python3 -c "print(int(float('${DUR_COUNT}') >= 2))")" = "1" ]; then
    ok "request duration histogram observed >= 2 samples (got ${DUR_COUNT})"
else
    bad "request duration histogram count is unexpected: ${DUR_COUNT}"
fi

step "Verify whisperx_service_info has version/device/serve_mode labels"
INFO_LINE=$(echo "$AFTER" | grep '^whisperx_service_info{' | head -1)
if echo "$INFO_LINE" | grep -q 'version=' \
    && echo "$INFO_LINE" | grep -q 'device=' \
    && echo "$INFO_LINE" | grep -q 'serve_mode='; then
    ok "service_info contains version/device/serve_mode labels"
    echo "    $INFO_LINE"
else
    bad "service_info missing expected labels: $INFO_LINE"
fi

step "Verify VRAM gauge is populated (MLX active memory, no CUDA)"
VRAM=$(echo "$AFTER" | grep '^whisperx_vram_allocated_bytes ' | awk '{print $2}')
if [ -n "$VRAM" ]; then
    # MLX active memory may be 0 if no inference ran in this process, or > 0 if it did.
    # The key invariant: the gauge is present and does not error from missing torch.cuda.
    ok "VRAM gauge is present (value: ${VRAM} bytes, MLX active memory or 0)"
else
    bad "VRAM gauge is missing from /metrics"
fi

step "Verify serve_mode='simple' in service_info (no Ray)"
if echo "$INFO_LINE" | grep -q 'serve_mode="simple"'; then
    ok "service_info reports serve_mode='simple'"
else
    bad "service_info does not report serve_mode='simple': $INFO_LINE"
fi

step "Summary"
echo "  Passed: $(color green ${PASS})"
echo "  Failed: $(color red ${FAIL})"
if [ "$FAIL" -eq 0 ]; then
    color green "ALL TESTS PASSED"; echo
    exit 0
else
    color red   "TESTS FAILED"; echo
    exit 1
fi
