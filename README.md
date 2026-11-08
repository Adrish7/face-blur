# Face Blur

Blur every face in a folder of photos, right in your browser. Choose a folder and get back a new folder of blurred copies. Your photos are never uploaded, and nothing is stored once you close the tab.

## Using it

1. **Optional:** under **Leave clear**, add a photo of each person whose face should stay visible. Add more photos of someone (**+ photo**) to make the match more reliable.
2. **Choose a folder** (or individual photos, or drop them on the page). JPG, PNG, HEIC, WebP and AVIF all work.
3. Check each photo:
    - **Click a face** to blur or unblur it.
    - **Drag** to blur a face it missed.
    - **Hold Space** (or **Original**) to see the photo without blur.
    - **← →** steps through photos, **⌘Z** undoes, **⌘ + scroll** zooms.
4. **Save to a folder** (Chrome and Edge on a computer) makes a new folder inside the one you pick, laid out like the folder you started with. Other browsers get a **ZIP** of that same folder.

PNGs stay PNG. Everything else is saved as JPEG. Saved photos carry no location, date or camera details.

## Good to know

- It can miss faces that are turned away, looking straight down, or mostly hidden. Check every photo before you share it.
- Someone small in the frame, or side-on, may stay blurred even if they're under **Leave clear**. Click them to unblur.
- The first visit downloads about 57 MB (the face models and their runtime). The browser keeps them after that.
- It runs on the GPU where the browser supports WebGPU, and on several CPU threads otherwise.

## How it works

It's a static site with no server and no build step:

- `faces.js` finds faces with InsightFace's **SCRFD-10G** and fingerprints them with **ArcFace MobileFaceNet**, run by [ONNX Runtime Web](https://onnxruntime.ai). The photo is checked at three sizes so both close-ups and the back row are found.
- A face counts as someone under **Leave clear** when its fingerprint's cosine similarity to one of their photos is 0.40 or more.
- `decode.js` opens HEIC with [libheif-js](https://github.com/catdad-experiments/libheif-js) in browsers that can't do it natively.
- `coi-sw.js` is a small service worker that adds the two headers browsers need before they let the models use several CPU threads. GitHub Pages can't send those headers itself.

ONNX Runtime and libheif load from jsDelivr at pinned versions. The models are in `models/`.

To run it locally: `npm start`, then open http://localhost:5177.

## Licences

The face models are InsightFace's and are released for **non-commercial use only**. ONNX Runtime is MIT-licensed and libheif-js is LGPL.
