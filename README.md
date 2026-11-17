# Sign Lens

Upload a photo of a store sign → find the largest word → guess its font → redraw the word in that font.
Everything runs in the browser: no server needed.

Made for Programming from A to Z (ITP, Fall 2026).

## Files

| File | What it is |
|---|---|
| `index.html` | Loads p5.js, Tesseract.js, onnxruntime-web, then `lens.js` and `sketch.js` |
| `lens.js` | Browser port of the Lens pipeline: OCR → crop → resize/pad → ResNet18 |
| `sketch.js` | The p5 sketch (upload, draw the photo, redraw the word) |
| `model/lens.onnx` | Lens font classifier converted from PyTorch to ONNX (983 Google Fonts families) |
| `model/fonts.json` | Class index → font name, Google Fonts family, category, font file URLs |

## Run locally

The model is loaded with `fetch`, so open the folder through a local server, not by double-clicking:

```
cd sign-lens-web
python3 -m http.server 8000
```

Then visit http://localhost:8000

## Credit

Font recognition: [Mixfont Lens](https://github.com/mixfont/lens) — open-weights model and pipeline,
used under its personal / academic / non-commercial license. The model was converted to ONNX and the
Python pipeline was ported to JavaScript for this project.
OCR: [Tesseract.js](https://github.com/naptha/tesseract.js). Inference: [onnxruntime-web](https://onnxruntime.ai).
