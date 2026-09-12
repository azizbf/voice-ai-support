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

`GET /api/v1/metrics/recent` now includes p50/p95/p99 and sample counts. `client_ttfa_ms` is recorded at the browser's first `playing` event for reply audio. Local capture starts its timer at the last detected speech sample; browser recognition uses its last transcript event as an approximation. This includes endpointing, upload, transcription, retrieval, generation, synthesis and browser buffering; it does not measure physical speaker hardware delay. Acknowledgements do not reset the clock. Server `tts_first_audio_ms` starts when the server accepts a turn and ends on its first audio chunk. Do not compare these two fields as if they had the same origin.

## Checks

From `backend`: `.venv/Scripts/python.exe -m unittest discover -s tests -v`

From `frontend`: `node tests/streamingAudio.test.cjs` and `node node_modules/typescript/bin/tsc --noEmit`

Provider benchmark: `.venv/Scripts/python.exe -m scripts.benchmark_voice_tts --runs 5`. Add `--max-p95-ms` with a measured deployment baseline to make it fail on regressions. Five requests are only a smoke check, not a reliable production p99 estimate.

Restart the backend and refresh the browser before measuring new calls. Physical microphone/speaker latency and noisy/accented-call quality still need live testing.
