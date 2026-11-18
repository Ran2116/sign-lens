// Sign Lens 3D: upload a photo with text → drag a box around a word → find its closest font
// → the word pops out of the photo as 3D letters in that font.
// Everything runs in the browser (see lens.js). Font recognition by Mixfont Lens.

let photo;          // p5 image for drawing
let photoEl;        // the same photo as an <img>, for Lens
let result;         // { word, word_box, input_image, font_matches }
let geom;           // the 3D word (p5.Geometry)
let geomBox;        // its bounding box
let popStart = 0;   // millis() when the word started popping out
let ready = false;
let busy = false;

// Where the photo sits on the canvas (set in draw, used by the mouse)
let view = { ox: 0, oy: 0, s: 1 };
// Drag selection, in canvas coordinates
let dragStart = null, dragEnd = null;

let fileInput, rotateCheckbox, wordInput, depthSlider, colorPicker, statusP, infoDiv;

async function setup() {
  createCanvas(800, 560, WEBGL);

  statusP = createP("Loading…");
  createP("1. Upload a photo with text &nbsp; 2. Drag a box around a word on the photo");
  fileInput = createFileInput(handleFile);
  createSpan(" Word: ");
  wordInput = createInput("");
  wordInput.size(140);
  wordInput.changed(() => buildWord());
  createSpan(" Depth: ");
  depthSlider = createSlider(0, 80, 30);
  depthSlider.changed(() => buildWord());
  createSpan(" Color: ");
  colorPicker = createColorPicker("#ffcc33");
  createDiv("");
  rotateCheckbox = createCheckbox(" Rotate view (drag on the canvas to orbit, scroll to zoom)", false);
  infoDiv = createDiv("");
  createDiv("Font recognition: Mixfont Lens (github.com/mixfont/lens), running in the browser.").class("credit");

  // Load the model, font list and OCR (first visit downloads ~50 MB, then the browser caches it)
  await Lens.load({
    modelUrl: "model/lens.onnx",
    fontsUrl: "model/fonts.json",
    onProgress: (msg) => setStatus(msg),
  });
  ready = true;
  setStatus("Ready! Upload a photo with text");
}

function setStatus(msg) {
  statusP.html(msg);
}

function draw() {
  background(235);

  // 3D view: let the mouse orbit the scene; otherwise keep the camera fixed for selecting
  if (rotateCheckbox.checked()) orbitControl();

  // Work in 2D-style coordinates: (0, 0) = top-left of the canvas, y goes down
  translate(-width / 2, -height / 2);

  if (!photo) return;

  // Fit the photo inside the canvas
  let s = min(width / photo.width, height / photo.height);
  let w = photo.width * s, h = photo.height * s;
  let ox = (width - w) / 2, oy = (height - h) / 2;
  view = { ox, oy, s };
  noStroke();
  image(photo, ox, oy, w, h);

  if (result && result.word_box) {
    // word_box is in result.input_image's size → scale it to the canvas
    let k = w / result.input_image.width;
    let b = result.word_box;
    let bx = ox + b.left * k, by = oy + b.top * k;
    let bw = b.width * k, bh = b.height * k;

    // Selection outline, slightly in front of the photo
    push();
    translate(0, 0, 0.5);
    noFill();
    stroke(40, 220, 100);
    strokeWeight(2);
    rect(bx, by, bw, bh);
    pop();

    if (geom) drawWord(bx, by, bw, bh);
  }

  // Box being dragged
  if (dragStart && dragEnd) {
    push();
    translate(0, 0, 1);
    noFill();
    stroke(255, 60, 60);
    strokeWeight(2);
    rect(dragStart.x, dragStart.y, dragEnd.x - dragStart.x, dragEnd.y - dragStart.y);
    pop();
  }
}

// The 3D word: fitted to the selection box, popping out of the photo
function drawWord(bx, by, bw, bh) {
  let size = geomBox.size;
  let fit = min(bw / max(size.x, 1), (bh * 1.2) / max(size.y, 1));
  let t = constrain((millis() - popStart) / 800, 0, 1);
  let ease = 1 - pow(1 - t, 3);

  push();
  ambientLight(120);
  directionalLight(255, 255, 255, 0.3, 0.6, -1);
  translate(bx + bw / 2, by + bh / 2, 1 + ease * (size.z * fit * 0.5 + 40));
  rotateY(sin(millis() / 900) * 0.25 * ease);
  rotateX(sin(millis() / 1300) * 0.1 * ease);
  scale(fit * (0.6 + 0.4 * ease));
  translate(-geomBox.offset.x, -geomBox.offset.y, -geomBox.offset.z);
  noStroke();
  fill(colorPicker.color());
  specularMaterial(60);
  shininess(20);
  model(geom);
  pop();
}

// ---------- Upload ----------

async function handleFile(file) {
  if (file.type !== "image") return;
  if (!ready) {
    setStatus("Still loading the model, try again in a moment");
    return;
  }
  photo = await loadImage(file.data);
  photoEl = new Image();
  photoEl.src = file.data;
  await photoEl.decode();

  result = null;
  geom = null;
  infoDiv.html("");
  wordInput.value("");
  rotateCheckbox.checked(false);
  setStatus("Drag a box around a word on the photo");
}

// ---------- Drag a box ----------

function overPhoto(x, y) {
  return photo && x >= view.ox && x <= view.ox + photo.width * view.s && y >= view.oy && y <= view.oy + photo.height * view.s;
}

function selecting() {
  return ready && !busy && !rotateCheckbox.checked();
}

function mousePressed() {
  if (!selecting() || !overPhoto(mouseX, mouseY)) return;
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
  geom = null;
  setStatus("Reading the selected area…");

  // Crop the selected area at full resolution
  let c = document.createElement("canvas");
  c.width = box.width;
  c.height = box.height;
  c.getContext("2d").drawImage(photoEl, box.left, box.top, box.width, box.height, 0, 0, box.width, box.height);

  let fontResult = await Lens.classifyImage(c, { topK: 5 });
  let text = await Lens.readText(c);

  result = {
    word: text || wordInput.value() || "?",
    word_box: box,
    input_image: { width: photoEl.naturalWidth, height: photoEl.naturalHeight },
    font_matches: fontResult.font_matches,
    model_input: fontResult.model_input,
    font: null,
  };
  wordInput.value(result.word);
  showMatches();
  await useMatch(0);
  busy = false;
}

// ---------- Fonts + 3D model ----------

// List the top matches; click one to rebuild the 3D word in that font
function showMatches() {
  infoDiv.html("");
  createP("<b>Closest fonts</b> (click to switch):").parent(infoDiv);
  result.font_matches.forEach((match, i) => {
    let row = createDiv(`${i + 1}. ${match.name} <small>(${match.category})</small> — ${nf(match.score * 100, 0, 0)}%`);
    row.class("match");
    row.parent(infoDiv);
    row.mousePressed(() => useMatch(i));
  });
  createP("<small>What the model saw:</small>").parent(infoDiv);
  let seen = createImg(result.model_input.toDataURL(), "model input");
  seen.style("border", "1px solid #ccc");
  seen.parent(infoDiv);
}

async function useMatch(i) {
  selectAll(".match").forEach((row, j) => (j === i ? row.addClass("active") : row.removeClass("active")));
  let match = result.font_matches[i];
  setStatus(`Loading ${match.name}…`);

  result.font = null;
  for (let url of Lens.fontFileUrls(match.name)) {
    try {
      result.font = await loadFont(url);
      break;
    } catch (err) {
      console.log(`Could not load ${url}`);
    }
  }
  if (!result.font) {
    setStatus(`Could not load ${match.name} — try another match`);
    return;
  }
  result.match = match;
  buildWord();
}

// Turn the word into extruded 3D geometry
function buildWord() {
  if (!result || !result.font) return;
  result.word = wordInput.value() || result.word;
  push();
  textSize(100);
  textAlign(LEFT, BASELINE);
  geom = result.font.textToModel(result.word, 0, 0, { sampleFactor: 0.5, extrude: depthSlider.value() });
  pop();
  geom.clearColors();
  geom.computeNormals();
  geomBox = geom.calculateBoundingBox();
  popStart = millis();
  let m = result.match;
  setStatus(`"${result.word}" → ${m.name} (${m.category}). Edit the word, depth or color, or tick "Rotate view".`);
}
