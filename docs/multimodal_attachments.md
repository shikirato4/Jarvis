# Multimodal Attachments

The attachment subsystem is provider-neutral and request scoped. UI code never sends a filesystem path or secret to a model provider.

## Contracts

`AttachmentDescriptor` records an opaque ID, request ID, safe name, detected MIME/category, byte size, SHA-256, protected storage token, state, required modality and bounded derived metadata. Raw bytes and storage paths are excluded from public descriptors and conversation serialization.

`AttachmentTypeRegistry` uses magic bytes/content before extension. It rejects absolute/traversal names, executables, active HTML/SVG, MIME mismatches, corrupt PDF/DOCX/JSON, configurable size/count overflow and likely credentials. ZIP-based DOCX names are checked for archive traversal.

`AttachmentStorage` restricts reads/removes to the request that owns the token. `MultimodalPipeline` caches derived output by content hash to avoid processing the same bytes twice while each descriptor retains its own request permission.

## Routes

| Category | Native route | Local derived route |
| --- | --- | --- |
| Image | Provider image transport when verified | Tesseract/vision OCR |
| Audio | Provider audio transport when verified | FFmpeg normalization and local STT with segments |
| Video | Provider video transport when verified | FFprobe/FFmpeg frames, audio and timestamped transcript |
| PDF/DOCX | None currently | Local extraction, pages and bounded chunks; OCR remains available for scans |
| TXT/Markdown/CSV/JSON/code | Text prompt after local validation | Local parsing and bounded chunks |

AUTO chooses the route for the provider assigned to that request. NVIDIA API may receive explicit native files or only necessary derived content. LOCAL never invokes NVIDIA. Changing mode leaves pending descriptors untouched.

## UI behavior

The `+` button exposes category filters and a multi-file picker. Drag/drop and clipboard paste use the same service. File chips show safe metadata/state and support removal, cancellation and retry. Keyboard users can open the menu with Enter/Space and close it with Escape. Sending text plus files creates one request; successful completion clears only those sent IDs.

## Configuration

- `JARVIS_ATTACHMENT_MAX_FILE_BYTES` (default 50 MiB)
- `JARVIS_ATTACHMENT_MAX_REQUEST_BYTES` (default 200 MiB)
- `JARVIS_ATTACHMENT_MAX_FILES_PER_REQUEST` (default 16)
- `JARVIS_ATTACHMENT_PROCESSING_TIMEOUT_SECONDS` (default 120)
- `JARVIS_ATTACHMENT_VIDEO_MAX_FRAMES` (default 12)

Tesseract, FFmpeg/FFprobe and a configured local STT backend are operational dependencies for their respective fallback routes. Missing tools produce a specific retryable error; they do not cause a silent mock or fabricated result.
