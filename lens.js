// lens.js — Mixfont Lens font recognition, running entirely in the browser.
// A JavaScript port of the Python pipeline in https://github.com/mixfont/lens
// (personal / academic / non-commercial license — credit the original author).
//
//   1. Tesseract.js finds the largest word in the photo        (was: pytesseract)
//   2. The word crop is resized/padded/normalized              (same math as the Python code)
//   3. The ResNet18 model runs with onnxruntime-web            (was: PyTorch)
//
// Needs these scripts on the page first:
//   https://cdn.jsdelivr.net/npm/tesseract.js@7/dist/tesseract.min.js
//   https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.min.js
//
// Usage:
//   await Lens.load({ modelUrl: "model/lens.onnx", fontsUrl: "model/fonts.json" });
//   const result = await Lens.recognize(imgOrCanvas);   // { word, word_box, input_image, font_matches }
//   const cssFamily = await Lens.loadFont(result.font_matches[0].name);

const Lens = (() => {
  // ---- Settings (same values as the Python code unless noted) ----
  const MAX_SIDE = 1200;          // photos are shrunk to this first (server used 1600; smaller = faster in the browser)
  const PAD_RATIO = 0.25 * 2.0;   // DEFAULT_PAD_RATIO * PAD_RATIO_MULTIPLIER
  const MIN_WORD_LEN = 3;
  const MIN_CONF = -1;
  const CROP_PADDING = 3;         // WORD_CROP_PADDING_PX
  const UPSCALE = 2.0;            // UPSCALE_FACTOR (only used if nothing is found at normal size)
  const MODEL_H = 128, MODEL_W = 384;
  const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];

  let session = null;   // onnxruntime session
  let worker = null;    // Tesseract.js worker
  let labels = [];      // class index -> font name
  let fonts = {};       // font name -> { family, category, files: [{ style, weight, url }] }
  const loadedFonts = {};

  // ---------------- Loading ----------------

  async function load({ modelUrl = "model/lens.onnx", fontsUrl = "model/fonts.json", onProgress = () => {} } = {}) {
    onProgress("Loading font list…");
    const data = await (await fetch(fontsUrl)).json();
    labels = data.labels;
    fonts = data.fonts;

    onProgress("Loading font model (~45 MB)…");
    session = await ort.InferenceSession.create(modelUrl, { executionProviders: ["wasm"] });

    onProgress("Loading OCR…");
    worker = await Tesseract.createWorker("eng");
    onProgress("Ready");
  }

  // ---------------- Grayscale images ----------------
  // A "gray image" here is { w, h, data: Uint8ClampedArray } with one byte per pixel.

  function grayFromSource(source) {
    const sw = source.naturalWidth || source.videoWidth || source.width;
    const sh = source.naturalHeight || source.videoHeight || source.height;
    const s = Math.min(1, MAX_SIDE / Math.max(sw, sh));
    const w = Math.max(1, Math.round(sw * s)), h = Math.max(1, Math.round(sh * s));
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#fff";           // transparent PNGs → white background, like the Python code
    ctx.fillRect(0, 0, w, h);
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(source, 0, 0, w, h);
    const rgba = ctx.getImageData(0, 0, w, h).data;
    const data = new Uint8ClampedArray(w * h);
    for (let i = 0; i < w * h; i++) {
      // PIL convert("L"): L = R*299/1000 + G*587/1000 + B*114/1000
      data[i] = (rgba[i * 4] * 299 + rgba[i * 4 + 1] * 587 + rgba[i * 4 + 2] * 114 + 500) / 1000;
    }
    return { w, h, data };
  }

  function grayToCanvas(img) {
    const canvas = document.createElement("canvas");
    canvas.width = img.w;
    canvas.height = img.h;
    const ctx = canvas.getContext("2d");
    const out = ctx.createImageData(img.w, img.h);
    for (let i = 0; i < img.w * img.h; i++) {
      out.data[i * 4] = out.data[i * 4 + 1] = out.data[i * 4 + 2] = img.data[i];
      out.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(out, 0, 0);
    return canvas;
  }

  // Average of the border pixels, sampling every (size // 50) pixels — same as the Python helper
  function edgeAverage(img) {
    const { w, h, data } = img;
    const stepX = Math.max(1, Math.floor(w / 50)), stepY = Math.max(1, Math.floor(h / 50));
    let total = 0, count = 0;
    for (let x = 0; x < w; x += stepX) {
      total += data[x]; count++;
      if (h > 1) { total += data[(h - 1) * w + x]; count++; }
    }
    for (let y = 0; y < h; y += stepY) {
      total += data[y * w]; count++;
      if (w > 1) { total += data[y * w + w - 1]; count++; }
    }
    return count ? Math.round(total / count) : 255;
  }

  function pad(img, left, top, right, bottom, fill) {
    const w = img.w + left + right, h = img.h + top + bottom;
    const data = new Uint8ClampedArray(w * h).fill(fill);
    for (let y = 0; y < img.h; y++) {
      data.set(img.data.subarray(y * img.w, (y + 1) * img.w), (y + top) * w + left);
    }
    return { w, h, data };
  }

  function crop(img, left, top, right, bottom) {
    const w = right - left, h = bottom - top;
    const data = new Uint8ClampedArray(w * h);
    for (let y = 0; y < h; y++) {
      data.set(img.data.subarray((y + top) * img.w + left, (y + top) * img.w + right), y * w);
    }
    return { w, h, data };
  }

  // PIL-style bicubic resize (a = -0.5, antialiased when shrinking), horizontal pass then vertical
  function bicubicWeight(x) {
    const a = -0.5;
    x = Math.abs(x);
    if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
    if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
    return 0;
  }

  function resampleCoeffs(inSize, outSize) {
    const scale = inSize / outSize;
    const filterScale = Math.max(scale, 1);
    const support = 2 * filterScale;
    const list = [];
    for (let xx = 0; xx < outSize; xx++) {
      const center = (xx + 0.5) * scale;
      const xmin = Math.max(Math.trunc(center - support + 0.5), 0);
      const xmax = Math.min(Math.trunc(center + support + 0.5), inSize);
      const weights = [];
      let sum = 0;
      for (let x = xmin; x < xmax; x++) {
        const k = bicubicWeight((x - center + 0.5) / filterScale);
        weights.push(k);
        sum += k;
      }
      list.push({ xmin, weights: weights.map((k) => (sum ? k / sum : 0)) });
    }
    return list;
  }

  function resize(img, outW, outH) {
    let cur = img;
    if (outW !== cur.w) {
      const coeffs = resampleCoeffs(cur.w, outW);
      const data = new Uint8ClampedArray(outW * cur.h);
      for (let y = 0; y < cur.h; y++) {
        for (let x = 0; x < outW; x++) {
          const { xmin, weights } = coeffs[x];
          let v = 0;
          for (let i = 0; i < weights.length; i++) v += cur.data[y * cur.w + xmin + i] * weights[i];
          data[y * outW + x] = Math.round(v);
        }
      }
      cur = { w: outW, h: cur.h, data };
    }
    if (outH !== cur.h) {
      const coeffs = resampleCoeffs(cur.h, outH);
      const data = new Uint8ClampedArray(cur.w * outH);
      for (let y = 0; y < outH; y++) {
        const { xmin, weights } = coeffs[y];
        for (let x = 0; x < cur.w; x++) {
          let v = 0;
          for (let i = 0; i < weights.length; i++) v += cur.data[(xmin + i) * cur.w + x] * weights[i];
          data[y * cur.w + x] = Math.round(v);
        }
      }
      cur = { w: cur.w, h: outH, data };
    }
    return cur;
  }

  // ---------------- OCR: find the largest word ----------------

  const hasAlnum = (text) => /[\p{L}\p{N}]/u.test(text);

  // Python compares tuples: (has letters/numbers, area, confidence, length)
  function score(box) {
    return [hasAlnum(box.text) ? 1 : 0, box.width * box.height, Math.max(box.conf, 0), box.text.length];
  }
  function better(a, b) {
    const sa = score(a), sb = score(b);
    for (let i = 0; i < sa.length; i++) if (sa[i] !== sb[i]) return sa[i] > sb[i];
    return false;
  }

  async function ocrWords(canvas, psm) {
    await worker.setParameters({ tessedit_pageseg_mode: String(psm) });
    const { data } = await worker.recognize(canvas, {}, { blocks: true });
    const words = [];
    for (const block of data.blocks || []) {
      for (const para of block.paragraphs) {
        for (const line of para.lines) {
          for (const word of line.words) {
            const text = (word.text || "").trim();
            const { x0, y0, x1, y1 } = word.bbox;
            if (text && x1 > x0 && y1 > y0) {
              words.push({ text, conf: word.confidence, left: x0, top: y0, width: x1 - x0, height: y1 - y0 });
            }
          }
        }
      }
    }
    return words;
  }

  function largestWord(words, minLen, minConf) {
    let best = null;
    for (const w of words) {
      if (w.text.length < minLen || w.conf < minConf) continue;
      if (!best || better(w, best)) best = w;
    }
    return best;
  }

  function scaleBoxDown(box, scale, baseW, baseH) {
    let left = Math.max(0, Math.round(box.left / scale));
    let top = Math.max(0, Math.round(box.top / scale));
    let right = Math.min(baseW, Math.max(left + 1, Math.round((box.left + box.width) / scale)));
    let bottom = Math.min(baseH, Math.max(top + 1, Math.round((box.top + box.height) / scale)));
    left = Math.min(left, Math.max(0, right - 1));
    top = Math.min(top, Math.max(0, bottom - 1));
    return { ...box, left, top, width: right - left, height: bottom - top };
  }

  // Relaxed thresholds, like _build_relaxed_thresholds(): length 3 → 2 → 1
  let minConfidence = MIN_CONF;

  function pickBest(wordLists) {
    const thresholds = [[MIN_WORD_LEN, minConfidence], [2, minConfidence], [1, minConfidence]];
    for (const [minLen, minConf] of thresholds) {
      let best = null;
      for (const words of wordLists) {
        const c = largestWord(words, minLen, minConf);
        if (c && (!best || better(c, best))) best = c;
      }
      if (best) return best;
    }
    return null;
  }

  async function findLargestWord(padded) {
    const canvas = grayToCanvas(padded);
    // Stages 1–2: normal size, page modes 6, 11, 3
    const lists = [];
    for (const psm of [6, 11, 3]) lists.push(await ocrWords(canvas, psm));
    let best = pickBest(lists);
    if (best) return best;

    // Stage 3 (only if nothing was found): upscale ×2 and try again
    const big = resize(padded, Math.round(padded.w * UPSCALE), Math.round(padded.h * UPSCALE));
    const bigCanvas = grayToCanvas(big);
    const bigLists = [];
    for (const psm of [6, 11, 3]) {
      bigLists.push((await ocrWords(bigCanvas, psm)).map((b) => scaleBoxDown(b, UPSCALE, padded.w, padded.h)));
    }
    return pickBest(bigLists);
  }

  // ---------------- Model ----------------

  // ResizePadToSize + Grayscale + ToTensor + Normalize
  function toTensor(img) {
    const fill = edgeAverage(img);
    const scale = Math.min(MODEL_W / img.w, MODEL_H / img.h);
    const nw = Math.min(MODEL_W, Math.max(1, Math.round(img.w * scale)));
    const nh = Math.min(MODEL_H, Math.max(1, Math.round(img.h * scale)));
    const resized = resize(img, nw, nh);
    const left = Math.floor((MODEL_W - nw) / 2), top = Math.floor((MODEL_H - nh) / 2);
    const padded = pad(resized, left, top, MODEL_W - nw - left, MODEL_H - nh - top, fill);

    const n = MODEL_W * MODEL_H;
    const input = new Float32Array(3 * n);
    for (let c = 0; c < 3; c++) {
      for (let i = 0; i < n; i++) input[c * n + i] = (padded.data[i] / 255 - MEAN[c]) / STD[c];
    }
    return { tensor: new ort.Tensor("float32", input, [1, 3, MODEL_H, MODEL_W]), modelInput: padded };
  }

  async function classify(img, topK) {
    const { tensor, modelInput } = toTensor(img);
    const output = await session.run({ [session.inputNames[0]]: tensor });
    const logits = output[session.outputNames[0]].data;
    const max = Math.max(...logits);
    const exps = Array.from(logits, (v) => Math.exp(v - max));
    const total = exps.reduce((a, b) => a + b, 0);
    const ranked = exps.map((e, i) => [e / total, i]).sort((a, b) => b[0] - a[0]).slice(0, topK);
    const matches = ranked.map(([p, i]) => {
      const name = labels[i];
      const info = fonts[name] || { family: name, category: "unknown", files: [] };
      return { name, score: Math.round(p * 100) / 100, category: info.category, fonts: info.files };
    });
    return { matches, modelInput };
  }

  // ---------------- Public: recognize ----------------

  // minConf: ignore OCR words below this confidence (0–100).
  // The Python original uses -1 (accept anything); raise it to skip textures misread as words.
  async function recognize(source, { topK = 3, minConf = MIN_CONF } = {}) {
    if (!session) throw new Error("Call Lens.load() first");
    minConfidence = minConf;
    const gray = grayFromSource(source);

    // Pad with the edge color so words touching the border are still found
    const padX = Math.max(1, Math.round(gray.w * PAD_RATIO));
    const padY = Math.max(1, Math.round(gray.h * PAD_RATIO));
    const padded = pad(gray, padX, padY, padX, padY, edgeAverage(gray));

    let wordImage = gray;
    let wordBox = null;
    const best = await findLargestWord(padded);
    if (best) {
      // Grow the box a little, crop the padded image, then map the box back to the photo
      const l = Math.max(0, best.left - CROP_PADDING), t = Math.max(0, best.top - CROP_PADDING);
      const r = Math.max(l + 1, Math.min(padded.w, best.left + best.width + CROP_PADDING));
      const b = Math.max(t + 1, Math.min(padded.h, best.top + best.height + CROP_PADDING));
      wordImage = crop(padded, l, t, r, b);

      const left = Math.max(0, Math.min(l - padX, gray.w - 1));
      const top = Math.max(0, Math.min(t - padY, gray.h - 1));
      const right = Math.max(left + 1, Math.min(r - padX, gray.w));
      const bottom = Math.max(top + 1, Math.min(b - padY, gray.h));
      wordBox = { text: best.text, left, top, width: right - left, height: bottom - top };
    }

    const { matches, modelInput } = await classify(wordImage, topK);
    return {
      word: wordBox ? wordBox.text : null,
      word_box: wordBox ? { left: wordBox.left, top: wordBox.top, width: wordBox.width, height: wordBox.height } : null,
      input_image: { width: gray.w, height: gray.h },   // word_box is in this (resized) size
      font_matches: matches,
      model_input: grayToCanvas(modelInput),            // what the model saw (handy for debugging)
    };
  }

  // ---------------- Public: load a matched font for drawing ----------------
  // Tries Google Fonts first (all Lens fonts are Google Fonts), then Mixfont's copy.
  // Returns a CSS font-family name you can pass to p5's textFont().

  async function loadFont(name, { weight = 400 } = {}) {
    const key = `${name}:${weight}`;
    if (loadedFonts[key]) return loadedFonts[key];
    const info = fonts[name] || { family: name, files: [] };
    const family = info.family;

    // 1) Google Fonts stylesheet
    const base = "https://fonts.googleapis.com/css2?family=" + encodeURIComponent(family).replace(/%20/g, "+");
    for (const url of [`${base}:wght@${weight}&display=block`, `${base}&display=block`]) {
      try {
        const res = await fetch(url);
        if (!res.ok) continue;
        const style = document.createElement("style");
        style.textContent = await res.text();
        document.head.appendChild(style);
        const faces = await document.fonts.load(`${weight} 40px "${family}"`, "Aa");
        if (faces.length) return (loadedFonts[key] = family);
      } catch (e) { /* try next */ }
    }

    // 2) Mixfont file
    const file = info.files.find((f) => f.weight === weight && !/italic/i.test(f.url)) || info.files[0];
    if (file && file.url) {
      try {
        const cssName = `Lens ${family}`;
        const face = new FontFace(cssName, `url("${encodeURI(file.url)}")`);
        await face.load();
        document.fonts.add(face);
        return (loadedFonts[key] = cssName);
      } catch (e) { /* give up below */ }
    }
    throw new Error(`Could not load font ${name}`);
  }

  // ---------------- Public: classify an image you already cropped (no OCR) ----------------

  async function classifyImage(source, { topK = 3 } = {}) {
    if (!session) throw new Error("Call Lens.load() first");
    const { matches, modelInput } = await classify(grayFromSource(source), topK);
    return { font_matches: matches, model_input: grayToCanvas(modelInput) };
  }

  // ---------------- Public: read the text in a cropped region ----------------
  // Treats the crop as one line of text and removes spaces, so widely spaced
  // letters like "F L E A" come back as "FLEA".

  async function readText(source) {
    if (!worker) throw new Error("Call Lens.load() first");
    const gray = grayFromSource(source);
    const g = edgeAverage(gray);
    const padded = pad(gray, 20, 20, 20, 20, g);
    await worker.setParameters({ tessedit_pageseg_mode: "7" });
    const { data } = await worker.recognize(grayToCanvas(padded));
    return (data.text || "").replace(/\s+/g, "").trim();
  }

  // ---------------- Public: font file URLs (.ttf) for p5's loadFont() ----------------
  // First choice: Google's own font files on GitHub (they allow cross-origin loading);
  // second choice: Mixfont's copy.

  function fontFileUrls(name, { weight = 400 } = {}) {
    const info = fonts[name];
    if (!info || !info.files.length) return [];
    const file =
      info.files.find((f) => f.weight === weight && f.style === "normal") ||
      info.files.find((f) => f.style === "normal") ||
      info.files[0];
    const urls = [];
    if (file.gh) urls.push("https://raw.githubusercontent.com/google/fonts/main/" + encodeURI(file.gh).replace(/\[/g, "%5B").replace(/\]/g, "%5D"));
    if (file.url) urls.push(encodeURI(file.url).replace(/\[/g, "%5B").replace(/\]/g, "%5D"));
    return urls;
  }

  return { load, recognize, classifyImage, readText, loadFont, fontFileUrls };
})();
