# Voice latency audit

Reviewed against [Bluejay’s twelve recommendations](https://getbluejay.ai/blog/12-ways-to-reduce-voice-agent-latency).

| # | Recommendation | App status |
|---|---|---|
| 1 | Thinking phrases | Added a brief local acknowledgement after 2.2 seconds. |
| 2 | Context preloading | Models warmed; tenant index preloaded and cached. |
| 3 | Correlated traces | Added call, turn and trace identifiers. |
| 4 | Endpointing | Already reduced to 550 ms; now measured. |
| 5 | STT partials | Browser recognition supports them; local Whisper remains batch. |
| 6 | Early retrieval | Added cancellable prefetch for stable browser partials. |
| 7 | LLM first token | Short context and fast voice model already configured; connections reused. |
| 8 | Tool overhead | Added bounded timeouts; no external business tools currently. |
| 9 | LLM → TTS | Short opening synthesis overlaps generation; the remaining text follows in order. |
| 10 | TTS first audio | Added provider-to-browser MP3 streaming. |
| 11 | Honest scaffolds | Acknowledgement cancels immediately when reply audio arrives. |
| 12 | Tail latency | Added bounded inference, cancellation, percentiles and a benchmark gate. |

## Implementation and limits

The acknowledgement says “Un instant, s’il vous plaît.” through browser speech synthesis, once per slow turn. It adds no provider call, is skipped if browser speech is unavailable, and is excluded from answer timing.

The browser uses MediaSource when `audio/mpeg` is supported. Other browsers keep buffered playback. A MediaSource failure before playback also falls back to buffered audio; a failure after playback begins stops without replaying the same words.

The knowledge cache holds at most 16 file versions and keys by tenant and file metadata. Replacing an index invalidates its version; deletion clears the cache. Browser partial retrieval is reused only when the final transcript matches exactly. Local Whisper produces no partials: repeatedly transcribing incomplete recordings would compete with final transcription on this CPU. A streaming STT provider would be a separate integration.

There are no business API tool calls to batch or co-locate. Deployment regions, autoscaling and production load tests need a real deployment. These have not been claimed as completed. Replies start with a useful 6–12-word sentence. A completed opening sentence or clause can enter synthesis immediately; remaining generated text follows in the same audio timeline. At most two speech requests are made per reply, avoiding a request at every punctuation mark. Decimal prices are not split. Full response text is retained and spoken, including the trailing confirmation question.

## Measurements

Three real Edge TTS requests after this change produced first bytes at **2285 / 313 / 299 ms**, versus complete audio at **3208 / 469 / 449 ms**. Streaming made bytes available **923 / 156 / 150 ms earlier**. These are synthesis timings for a test sentence, not measured full-call savings. Startup/network variation remains substantial.

`GET /api/v1/metrics/recent` now includes p50/p95/p99 and sample counts. `client_ttfa_ms` is recorded at the browser's first `playing` event for reply audio. Speech-end is the last microphone RMS sample above threshold on both browser and local capture paths. If microphone activity cannot be measured, those totals are omitted rather than approximated from transcript events. This includes endpointing, upload, transcription, retrieval, generation, synthesis and browser buffering; it does not measure physical speaker hardware delay. Acknowledgements do not reset the clock. Server `tts_first_audio_ms` starts when the server accepts a turn and ends on its first audio chunk. Do not compare these two fields as if they had the same origin.

## Checks

From `backend`: `.venv/Scripts/python.exe -m unittest discover -s tests -v`

From `frontend`: `node tests/streamingAudio.test.cjs` and `node node_modules/typescript/bin/tsc --noEmit`

Provider benchmark: `.venv/Scripts/python.exe -m scripts.benchmark_voice_tts --runs 5`. Add `--max-p95-ms` with a measured deployment baseline to make it fail on regressions. Five requests are only a smoke check, not a reliable production p99 estimate.

Restart the backend and refresh the browser before measuring new calls. Physical microphone/speaker latency and noisy/accented-call quality still need live testing.

## Further changes after the fourth timing export

The AI HTTP pool now retains idle connections for up to 120 seconds instead of the SDK's five-second default. After a longer idle period, a bounded read-only models request can prepare the connection while transcription runs. This avoids generating filler tokens or extra answers, but cannot remove model inference or network propagation time.

The call UI offers an opt-in “Reconnaissance en direct · Chrome / Edge” mode. It uses the browser speech service during the utterance and falls back to the configured server recognizer after network/service errors. It does not require a new API key, but browser availability and service connectivity vary; audio is processed by the browser's recognition service. The default remains local/server transcription. Browser-mode end-of-speech timing is approximate, as noted in the debug panel.

A small same-model CPU check compared 4, 2 and 6 threads. Six threads saved only about 30–45 ms on most samples, while two were slower. No model/thread default was changed based on these small results. Further reliable reductions in the local transcription stage need a different runtime/hardware or a streaming transcription integration; shrinking the model also needs accuracy evaluation.
# Optional ElevenLabs text-input streaming

Set `TTS_PROVIDER=elevenlabs`, `ELEVENLABS_API_KEY`, and `ELEVENLABS_VOICE_ID` in the backend environment and restart it. `ELEVENLABS_MODEL` defaults to `eleven_flash_v2_5`. Edge remains the default; configuring a key alone does not activate paid synthesis.

Voice replies feed complete sentences as the LLM generates them into one WebSocket, while audio is received concurrently. `auto_mode=true` avoids character schedules; complete sentences follow the provider's quality guidance. The final incomplete sentence is flushed at LLM completion. MP3 output preserves existing browser streaming playback and fallback. Cancellation closes the socket and cancels pending text input; partial audio is never automatically retried. Existing phrase caching belongs to the Edge path, not the new ElevenLabs path.

The debug panel measures first sentence preparation under `llm_phrase` and connection/synthesis to first audio under `tts`. Compare end-of-speech to actual playback, not provider inference-only claims. Mocked tests verify concurrent text/audio and cancellation; real provider latency and browser playback still require a configured account. This change does not relocate the backend or promise a 1.5-second total.

Provider guidance: https://elevenlabs.io/docs/eleven-api/guides/how-to/best-practices/latency-optimization and https://elevenlabs.io/docs/api-reference/text-to-speech/v-1-text-to-speech-voice-id-stream-input
# French voice latency comparison (2026-09-13)

Switched the local Edge voice and default from VivienneMultilingualNeural to DeniseNeural. Five distinct French phrases per voice, with sockets primed before timing and no phrase-cache hits, measured request-to-first-audio as follows: Vivienne 862/995/838/704/996 ms; Denise 187/231/149/280/379 ms. The last two comparisons reversed voice order. Median fell from 862 to 231 ms in this small sample. These are provider timings, not microphone-to-speaker totals; they exclude connection preparation and do not guarantee production latency. Voice timbre changes and should be evaluated in a real call. All 31 backend tests pass.
# STT tuning (2026-09-13)

Local Whisper remains Base: Tiny ran in 268/284/367 ms versus Base 501/523/538 ms on three synthetic French samples, but Tiny mistranscribed currency ("dinars"). Both models also made errors on a spoken customer number. Synthetic samples do not establish accuracy for real microphones or accents.

For Base on this machine, three warmed runs of the same phrase gave medians of 520 ms (4 CPU threads), 503 ms (6), and 544 ms (8). The local backend environment now uses `WHISPER_CPU_THREADS=6`; the portable default stays 4. This is a small inference improvement, not a claim of sub-300-ms recognition.

The frontend end-of-speech wait defaults to 450 ms instead of 600 ms. With the recorder's 80-ms polling interval, nominal endpointing changes from 640 to 480 ms. Browser recognition uses the same timeout after transcript updates, which is not an exact speech-end measurement. To accommodate longer pauses, set `NEXT_PUBLIC_VOICE_SILENCE_MS=600` in `frontend/.env.local` and restart the frontend. Earlier endpointing can split hesitant speech. TypeScript checks and all 31 backend tests pass; real-call timing remains to be measured.
# Optional local GPU STT (2026-09-14)

`WHISPER_DEVICE=cuda` runs the existing Whisper model with float16 on NVIDIA GPUs; the default remains `cpu` with int8. `auto` selects CUDA when CTranslate2 detects a GPU, but still requires compatible runtime libraries. The model is warmed during API startup. On Windows, STT prepends venv NVIDIA `bin` folders to the process PATH and calls `add_dll_directory`, without editing the machine PATH or installing a display driver.

Install the optional runtime with `backend/.venv/Scripts/python.exe -m pip install -r backend/requirements-cuda.txt`. Set `WHISPER_DEVICE=cuda` in `backend/.env` and restart the API after confirming GPU inference. To revert, set it to `cpu`. Missing CUDA libraries fail warm-up rather than silently claiming GPU operation. Browser WebM audio now enters the decoder through memory, avoiding temporary-file writes.

From the backend directory, run `.venv/Scripts/python.exe -m scripts.benchmark_voice_stt samples/stt-gpu-0.mp3 samples/stt-gpu-1.mp3 samples/stt-gpu-2.mp3` to compare CPU and GPU on identical audio. On this RTX 3050, Whisper Base warmed inference was about 590–640 ms on CPU int8 and 82–98 ms on CUDA float16; transcripts matched, including “30 dinars”. Timings exclude model initialization and are not end-to-end voice latency.
