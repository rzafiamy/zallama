# TeleOCR 1.2B — integration notes (2026-10-02)

Integrated as backend `teleocr-server` (`modality: ocr`, `POST /v1/ocr`),
engine [teleocr-rs](https://github.com/rzafiamy/teleocr-rs) (local checkout
/bank2/teleocr-rs): a Rust/Candle port, not llama.cpp. Usage in README.md
("Document OCR"), params in CONFIG.md; measurements in teleocr-rs
docs/performance.md.

## Why not llama.cpp

- Config says `Qwen2_5_VLForConditionalGeneration`, but the decoder is
  Qwen3-style: hidden 1024, 16 heads, `head_dim: 128` (≠ 1024/16), per-head
  QK RMSNorm, 28 layers, tied embeddings. The vision tower is a stock
  Qwen2.5-VL ViT (32 blocks, width 1280, 112 px windows, full attention in
  blocks 7/15/23/31).
- llama.cpp (our 7e4c0a968 and upstream master 4ebdf2c74 of 2026-10-02)
  fails with `check_tensor_dims: tensor 'blk.0.attn_q.weight' has wrong
  shape`; community GGUF `nandraj/NaviDC-OCR-GGUF` needs a patched build and
  was made from the older NaviDC-OCR checkpoint.

## Pipeline facts (official client, github.com/caipeng328/TeleOCR)

- Two stages: layout on a 1036×1036 bicubic copy (`\nAnalyze the image
  layout.`, or `\nMulti-point Layout Segmentation Analysis.` for photos),
  output lines `<box:x1 y1 x2 y2><label:TYPE><up|right|down|left>`, coords
  0-1000 (polygons in segmentation mode); then each block cropped from the
  full page (rotated upright, ≥ 28 px edge) with a per-type prompt.
- Decoding of the official pipeline ≠ model card: greedy, repetition 1.0,
  presence 1.0 + frequency 0.05 (0.005 for tables / seals / figures),
  `no_repeat_ngram_size` 100; layout with no penalties. Model card:
  repetition 1.05. Prompts start with `\n`.
- PDFs rendered at 200 DPI (pypdfium2).

## Results (RTX 4090)

- Token-identical to transformers on the model card's five samples, F32 and
  Q8_0, CPU and CUDA. q8v GGUF (Q8_0 text + vision, 1.5 GB) = q8_0 (F16
  vision) on 6 paper pages, identical Markdown.
- ~225 tok/s per sequence; ~3.2 s per page at batch 8 (layouts of all pages
  batched, then all blocks). VRAM +6.4 GB at batch 8 (F32 KV cache), so it
  cannot sit next to the 21 GB 27B: default eviction group `primary`.
