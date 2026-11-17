// Sign Lens: upload a storefront photo → find the largest word and its closest font
// → redraw that word in the matched font, right where it sits in the photo.
// Everything runs in the browser (see lens.js). Font recognition by Mixfont Lens.

let photo;          // p5 image for drawing
let result;         // { word, word_box, input_image, font_matches, model_input }
let fontFamily;     // CSS name of the loaded matched font
let ready = false;
let status = "Loading…";

let fileInput, overlayCheckbox, infoDiv;

async function setup() {
  createCanvas(800, 560);
  createP("Upload a photo of a store sign:");
  fileInput = createFileInput(handleFile);
  overlayCheckbox = createCheckbox(" Show the word in the matched font", true);
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
    image(photo, ox, oy, w, h);

    if (result && result.word_box) {
      // word_box is in Lens's resized image → scale it to the canvas
      let k = w / result.input_image.width;
      let b = result.word_box;
      let bx = ox + b.left * k, by = oy + b.top * k;
      let bw = b.width * k, bh = b.height * k;

      if (overlayCheckbox.checked() && fontFamily) {
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
  }

  // Status bar
  noStroke();
  fill(0);
  textFont("sans-serif");
  textAlign(LEFT, CENTER);
  textSize(14);
  text(status, 8, height - 14);
}

async function handleFile(file) {
  if (file.type !== "image") return;
  if (!ready) {
    status = "Still loading the model, try again in a moment";
    return;
  }
  photo = await loadImage(file.data);
  result = null;
  fontFamily = null;
  infoDiv.html("");
  status = "Recognizing…";

  // Lens needs a regular <img> element
  let img = new Image();
  img.src = file.data;
  await img.decode();

  result = await Lens.recognize(img, { topK: 5 });
  status = result.word ? `Found "${result.word}"` : "No word found — font guess for the whole image";
  showMatches();
  await useMatch(0);
}

// List the top matches; click one to redraw the word in that font
function showMatches() {
  infoDiv.html("");
  createP(`<b>Word:</b> ${result.word ?? "(none)"}`).parent(infoDiv);
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
  status = `Loading ${match.name}…`;
  try {
    fontFamily = await Lens.loadFont(match.name);
    status = result.word ? `"${result.word}" → ${match.name}` : `Font guess: ${match.name}`;
  } catch (err) {
    status = err.message;
  }
}
