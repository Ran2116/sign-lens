// Sign Lens: upload a storefront photo → find a word and its closest font
// → redraw that word in the matched font, right where it sits in the photo.
// Everything runs in the browser (see lens.js). Font recognition by Mixfont Lens.
//
// Two ways to pick the word:
//   - automatic: right after upload, OCR looks for the largest confident word
//   - manual: drag a box around the word on the photo

let photo;          // p5 image for drawing
let photoEl;        // the same photo as an <img>, for Lens
let result;         // { word, word_box, input_image, font_matches }
let fontFamily;     // CSS name of the loaded matched font
let ready = false;
let busy = false;
let status = "Loading…";

// Where the photo sits on the canvas (set in draw, used by the mouse)
let view = { ox: 0, oy: 0, s: 1 };
// Drag selection, in canvas coordinates
let dragStart = null, dragEnd = null;

let fileInput, overlayCheckbox, wordInput, infoDiv;

async function setup() {
  createCanvas(800, 560);
  createP("Upload a photo of a store sign, then drag a box around a word if the automatic pick is wrong:");
  fileInput = createFileInput(handleFile);
  overlayCheckbox = createCheckbox(" Show the word in the matched font", true);
  createSpan("Word: ");
  wordInput = createInput("");
  wordInput.input(() => {
    if (result) result.word = wordInput.value();
  });
  infoDiv = createDiv("");
  createDiv("Font recognition: Mixfont Lens (github.com/mixfont/lens), running in the browser.").class("credit");

  // Load the model, font list and OCR (first visit downloads ~50 MB, then the browser caches it)
  await Lens.load({
    modelUrl: "model/lens.onnx",
    fontsUrl: "model/fonts.json",
    onProgress: (msg) => (status = msg),
  });
  ready = true;
  status = "Ready! Upload a storefront photo";
}

function draw() {
  background(235);

  if (photo) {
    // Fit the photo inside the canvas
    let s = min(width / photo.width, (height - 30) / photo.height);
    let w = photo.width * s, h = photo.height * s;
    let ox = (width - w) / 2, oy = 0;
    view = { ox, oy, s };
    image(photo, ox, oy, w, h);

    if (result && result.word_box) {
      // word_box is in result.input_image's size → scale it to the canvas
      let k = w / result.input_image.width;
      let b = result.word_box;
      let bx = ox + b.left * k, by = oy + b.top * k;
      let bw = b.width * k, bh = b.height * k;

      if (overlayCheckbox.checked() && fontFamily && result.word) {
        // Cover the original word, then write it again in the matched font
        noStroke();
        fill(255, 235);
        rect(bx, by, bw, bh);
        fill(0);
        textFont(fontFamily);
        textAlign(CENTER, CENTER);
        textSize(100);
        let size = 100 * (bw / max(1, textWidth(result.word)));
        textSize(min(size, bh * 1.4));
        text(result.word, bx + bw / 2, by + bh / 2);
      }

      // Detection box
      noFill();
      stroke(40, 220, 100);
      strokeWeight(2);
      rect(bx, by, bw, bh);
    }

    // Box being dragged
    if (dragStart && dragEnd) {
      noFill();
      stroke(255, 60, 60);
      strokeWeight(2);
      drawingContext.setLineDash([6, 4]);
      rect(dragStart.x, dragStart.y, dragEnd.x - dragStart.x, dragEnd.y - dragStart.y);
      drawingContext.setLineDash([]);
    }
  }

  // Status bar
  noStroke();
  fill(0);
  textFont("sans-serif");
  textAlign(LEFT, CENTER);
  textSize(14);
  text(status, 8, height - 14);
}

// ---------- Upload: automatic pick ----------

async function handleFile(file) {
  if (file.type !== "image") return;
  if (!ready) {
    status = "Still loading the model, try again in a moment";
    return;
  }
  photo = await loadImage(file.data);
  photoEl = new Image();
  photoEl.src = file.data;
  await photoEl.decode();

  result = null;
  fontFamily = null;
  infoDiv.html("");
  wordInput.value("");
  busy = true;
  status = "Looking for the largest word…";

  // minConf 60: skip textures that OCR misreads as words
  result = await Lens.recognize(photoEl, { topK: 5, minConf: 60 });
  busy = false;
  if (result.word) {
    status = `Found "${result.word}" — drag a box to pick a different word`;
    wordInput.value(result.word);
    showMatches();
    await useMatch(0);
  } else {
    result = null;
    status = "No clear word found — drag a box around a word";
  }
}

// ---------- Drag a box: manual pick ----------

function overPhoto(x, y) {
  return photo && x >= view.ox && x <= view.ox + photo.width * view.s && y >= view.oy && y <= view.oy + photo.height * view.s;
}

function mousePressed() {
  if (!ready || busy || !overPhoto(mouseX, mouseY)) return;
  dragStart = { x: mouseX, y: mouseY };
  dragEnd = null;
}

function mouseDragged() {
  if (!dragStart) return;
  dragEnd = {
    x: constrain(mouseX, view.ox, view.ox + photo.width * view.s),
    y: constrain(mouseY, view.oy, view.oy + photo.height * view.s),
  };
}

async function mouseReleased() {
  if (!dragStart || !dragEnd) {
    dragStart = dragEnd = null;
    return;
  }
  let x0 = min(dragStart.x, dragEnd.x), y0 = min(dragStart.y, dragEnd.y);
  let x1 = max(dragStart.x, dragEnd.x), y1 = max(dragStart.y, dragEnd.y);
  dragStart = dragEnd = null;
  if (x1 - x0 < 10 || y1 - y0 < 10) return;   // ignore clicks

  // Canvas → original photo pixels
  let k = photoEl.naturalWidth / (photo.width * view.s);
  let box = {
    left: round((x0 - view.ox) * k),
    top: round((y0 - view.oy) * k),
    width: round((x1 - x0) * k),
    height: round((y1 - y0) * k),
  };
  await analyzeRegion(box);
}

async function analyzeRegion(box) {
  busy = true;
  status = "Reading the selected area…";

  // Crop the selected area at full resolution
  let c = document.createElement("canvas");
  c.width = box.width;
  c.height = box.height;
  c.getContext("2d").drawImage(photoEl, box.left, box.top, box.width, box.height, 0, 0, box.width, box.height);

  let fontResult = await Lens.classifyImage(c, { topK: 5 });
  let text = await Lens.readText(c);
  busy = false;

  result = {
    word: text || wordInput.value() || "?",
    word_box: box,
    input_image: { width: photoEl.naturalWidth, height: photoEl.naturalHeight },
    font_matches: fontResult.font_matches,
    model_input: fontResult.model_input,
  };
  wordInput.value(result.word);
  status = text ? `Read "${text}" — fix it in the Word box if needed` : "Couldn't read the text — type it in the Word box";
  showMatches();
  await useMatch(0);
}

// ---------- Font list ----------

// List the top matches; click one to redraw the word in that font
function showMatches() {
  infoDiv.html("");
  result.font_matches.forEach((match, i) => {
    let row = createDiv(`${i + 1}. ${match.name} <small>(${match.category})</small> — ${nf(match.score * 100, 0, 0)}%`);
    row.class("match");
    row.parent(infoDiv);
    row.mousePressed(() => useMatch(i));
  });
  // What the model actually looked at
  createP("<small>What the model saw:</small>").parent(infoDiv);
  let seen = createImg(result.model_input.toDataURL(), "model input");
  seen.style("border", "1px solid #ccc");
  seen.parent(infoDiv);
}

async function useMatch(i) {
  selectAll(".match").forEach((row, j) => (j === i ? row.addClass("active") : row.removeClass("active")));
  let match = result.font_matches[i];
  try {
    fontFamily = await Lens.loadFont(match.name);
  } catch (err) {
    status = err.message;
  }
}
