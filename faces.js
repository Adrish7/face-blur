// Face finding (SCRFD-10G) and face matching (ArcFace MobileFaceNet), run on
// the visitor's own device: photos are never uploaded anywhere.
//
// The ONNX runtime comes from jsDelivr at a pinned version. The two models are
// InsightFace's, released for non-commercial use, and sit next to this file.

const ORT_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
const WASM_URL = `${ORT_BASE}ort-wasm-simd-threaded.asyncify.wasm`;
const DETECTOR_URL = new URL("./models/det_10g.onnx", import.meta.url).href;
const RECOGNISER_URL = new URL("./models/w600k_mbf.onnx", import.meta.url).href;
// For the download progress bar (a gzipped response doesn't say its real size)
const BYTES = { [WASM_URL]: 26781914, [DETECTOR_URL]: 16923827, [RECOGNISER_URL]: 13616099 };

// Where ArcFace expects the eyes, nose tip and mouth corners in its 112x112 input
const ARCFACE_POINTS = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

const DETECT = {
    // Below InsightFace's 0.5: a missed face is worse than a blur clicked off
    threshold: 0.4,
    // The detector loses faces much over ~400px (a close-up scored 0.15 at
    // 1280 and 0.9 at 640), and faces much under ~15px, so a photo is seen at
    // three sizes: whole at 640 for close-ups, whole at 1280 for ordinary
    // faces, and in tiles at 2560 for the small faces at the back of a room
    coarseSide: 640,
    fullSide: 1280,
    tileSide: 2560,
    tile: 1280,
    overlap: 224,
};

let ort = null;
let detector = null;
let recogniser = null;
let loading = null;

async function download(url, onBytes) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Couldn't download ${url.split("/").pop()} (${res.status})`);
    if (!res.body) {
        const bytes = new Uint8Array(await res.arrayBuffer());
        onBytes(bytes.length);
        return bytes;
    }
    const reader = res.body.getReader();
    const chunks = [];
    let length = 0;
    for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        length += value.length;
        onBytes(value.length);
    }
    const bytes = new Uint8Array(length);
    let at = 0;
    for (const chunk of chunks) {
        bytes.set(chunk, at);
        at += chunk.length;
    }
    return bytes;
}

async function openModel(bytes, preferGpu) {
    const order = preferGpu && "gpu" in navigator ? ["webgpu", "wasm"] : ["wasm"];
    let lastError;
    for (const ep of order) {
        try {
            // The detector takes any photo size, so ORT's "unexpected output
            // shape" warnings are noise: log errors only
            const session = await ort.InferenceSession.create(bytes, {
                executionProviders: [ep],
                logSeverityLevel: 3,
            });
            return { session, ep, bytes };
        } catch (error) {
            lastError = error;
        }
    }
    throw lastError;
}

/**
 * Loads the runtime and both models once (about 57 MB, cached by the browser
 * afterwards). `onProgress` gets 0 to 1. Resolves to a short description of
 * where the models run, such as "GPU" or "CPU, 4 threads".
 */
export function loadModels(onProgress = () => {}) {
    loading ??= (async () => {
        const total = Object.values(BYTES).reduce((a, b) => a + b, 0);
        let got = 0;
        const tick = (n) => {
            got += n;
            onProgress(Math.min(1, got / total));
        };
        const [mod, wasm, det, rec] = await Promise.all([
            import(`${ORT_BASE}ort.webgpu.min.mjs`),
            download(WASM_URL, tick),
            download(DETECTOR_URL, tick),
            download(RECOGNISER_URL, tick),
        ]);
        ort = mod;
        ort.env.wasm.wasmPaths = ORT_BASE;
        ort.env.wasm.wasmBinary = wasm;
        // Several threads need a cross-origin isolated page (see coi-sw.js)
        ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
        ort.env.logLevel = "error";
        // One at a time: ORT refuses a second WebGPU session while one is being created
        detector = await openModel(det, true);
        recogniser = await openModel(rec, true);
        return backend();
    })();
    loading.catch(() => (loading = null));
    return loading;
}

function backend() {
    if (detector?.ep === "webgpu") return "GPU";
    const threads = ort.env.wasm.numThreads;
    return threads > 1 ? `CPU, ${threads} threads` : "CPU";
}

// A model can open on the GPU and still fail on its first run: drop to CPU once
async function run(model, feeds) {
    try {
        return await model.session.run(feeds);
    } catch (error) {
        if (model.ep !== "webgpu") throw error;
        console.warn("Face model failed on the GPU, switching to the CPU", error);
        Object.assign(model, await openModel(model.bytes, false));
        return model.session.run(feeds);
    }
}

function canvas2d(w, h, willRead = false) {
    const canvas = new OffscreenCanvas(w, h);
    const ctx = canvas.getContext("2d", { willReadFrequently: willRead });
    return { canvas, ctx };
}

// Part of the photo, scaled into a black canvas whose sides are multiples of 32
function detectorInput(source, rect, scale) {
    const dw = Math.max(1, Math.round(rect.w * scale));
    const dh = Math.max(1, Math.round(rect.h * scale));
    const W = Math.ceil(dw / 32) * 32;
    const H = Math.ceil(dh / 32) * 32;
    const { ctx } = canvas2d(W, H, true);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, W, H);
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(source, rect.x, rect.y, rect.w, rect.h, 0, 0, dw, dh);
    const px = ctx.getImageData(0, 0, W, H).data;
    const n = W * H;
    const f = new Float32Array(3 * n);
    for (let i = 0, p = 0; i < n; i++, p += 4) {
        f[i] = (px[p] - 127.5) / 128;
        f[i + n] = (px[p + 1] - 127.5) / 128;
        f[i + 2 * n] = (px[p + 2] - 127.5) / 128;
    }
    return {
        tensor: new ort.Tensor("float32", f, [1, 3, H, W]),
        W,
        dw,
        dh,
        sx: dw / rect.w,
        sy: dh / rect.h,
    };
}

async function detectPass(source, rect, scale, edges) {
    const model = detector;
    const { tensor, W, dw, dh, sx, sy } = detectorInput(source, rect, scale);
    const out = await run(model, { [model.session.inputNames[0]]: tensor });
    const names = model.session.outputNames; // scores x3, boxes x3, points x3 (strides 8, 16, 32)
    const found = [];
    const margin = 3;
    [8, 16, 32].forEach((stride, k) => {
        const scores = out[names[k]].data;
        const boxes = out[names[k + 3]].data;
        const pts = out[names[k + 6]].data;
        const cols = W / stride;
        for (let i = 0; i < scores.length; i++) {
            const score = scores[i];
            if (score < DETECT.threshold) continue;
            const cell = i >> 1; // two anchors per grid cell
            const cx = (cell % cols) * stride;
            const cy = Math.floor(cell / cols) * stride;
            const b = i * 4;
            const x1 = cx - boxes[b] * stride;
            const y1 = cy - boxes[b + 1] * stride;
            const x2 = cx + boxes[b + 2] * stride;
            const y2 = cy + boxes[b + 3] * stride;
            // In the tiled pass, a face cut by an inner tile edge is found whole in the next tile
            if (edges) {
                if (edges.left && x1 < margin) continue;
                if (edges.top && y1 < margin) continue;
                if (edges.right && x2 > dw - margin) continue;
                if (edges.bottom && y2 > dh - margin) continue;
            }
            const kps = [];
            for (let j = 0; j < 5; j++) {
                kps.push([rect.x + (cx + pts[i * 10 + 2 * j] * stride) / sx, rect.y + (cy + pts[i * 10 + 2 * j + 1] * stride) / sy]);
            }
            found.push({
                score,
                x1: rect.x + x1 / sx,
                y1: rect.y + y1 / sy,
                x2: rect.x + x2 / sx,
                y2: rect.y + y2 / sy,
                kps,
            });
        }
    });
    tensor.dispose?.();
    for (const t of Object.values(out)) t.dispose?.();
    return found;
}

function overlap(a, b) {
    const ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
    const iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
    const inter = ix * iy;
    const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    return { iou: inter / (areaA + areaB - inter), within: inter / Math.min(areaA, areaB) };
}

// Keep the strongest of overlapping finds, including a tile's partial copy of a face
function suppress(faces) {
    const kept = [];
    for (const face of [...faces].sort((a, b) => b.score - a.score)) {
        if (
            kept.every((k) => {
                const o = overlap(k, face);
                return o.iou < 0.4 && o.within < 0.65;
            })
        )
            kept.push(face);
    }
    return kept;
}

function tileStarts(length, tile, overlapBy) {
    if (length <= tile) return [0];
    const count = Math.ceil((length - overlapBy) / (tile - overlapBy));
    const step = (length - tile) / (count - 1);
    return Array.from({ length: count }, (_, i) => Math.round(i * step));
}

// Tilt the blur with the head, but only for a roughly front-on face: on a side
// profile the eye points bunch together and their angle means nothing
function headTilt(face) {
    const [l, r] = face.kps;
    if (Math.hypot(r[0] - l[0], r[1] - l[1]) < 0.3 * (face.x2 - face.x1)) return 0;
    const limit = Math.PI / 6;
    return Math.max(-limit, Math.min(limit, Math.atan2(r[1] - l[1], r[0] - l[0])));
}

/**
 * Every face in the photo, close-ups to the back row (see DETECT). `thorough`
 * adds the tiled pass for small faces; a headshot doesn't need it.
 */
export async function detectFaces(source, { thorough = true } = {}) {
    await loadModels();
    const W = source.width;
    const H = source.height;
    const long = Math.max(W, H);
    const whole = { x: 0, y: 0, w: W, h: H };
    const fullScale = Math.min(2, DETECT.fullSide / long);
    let faces = await detectPass(source, whole, fullScale, null);

    const coarseScale = DETECT.coarseSide / long;
    if (coarseScale < fullScale * 0.75) {
        faces = faces.concat(await detectPass(source, whole, coarseScale, null));
    }

    const tileScale = Math.min(1.5, DETECT.tileSide / long);
    if (thorough && tileScale > fullScale * 1.3) {
        const tile = DETECT.tile / tileScale; // in photo pixels
        const overlapBy = DETECT.overlap / tileScale;
        for (const y of tileStarts(H, tile, overlapBy)) {
            for (const x of tileStarts(W, tile, overlapBy)) {
                const rect = { x, y, w: Math.min(tile, W - x), h: Math.min(tile, H - y) };
                const edges = {
                    left: x > 0,
                    top: y > 0,
                    right: x + rect.w < W - 1,
                    bottom: y + rect.h < H - 1,
                };
                faces = faces.concat(await detectPass(source, rect, tileScale, edges));
            }
        }
    }
    return suppress(faces).map((face) => ({ ...face, angle: headTilt(face) }));
}

// Least-squares similarity transform (scale, rotation, shift) taking src onto dst
function similarity(src, dst) {
    const n = src.length;
    let smx = 0,
        smy = 0,
        dmx = 0,
        dmy = 0;
    for (let i = 0; i < n; i++) {
        smx += src[i][0];
        smy += src[i][1];
        dmx += dst[i][0];
        dmy += dst[i][1];
    }
    smx /= n;
    smy /= n;
    dmx /= n;
    dmy /= n;
    let dot = 0,
        cross = 0,
        norm = 0;
    for (let i = 0; i < n; i++) {
        const px = src[i][0] - smx;
        const py = src[i][1] - smy;
        const qx = dst[i][0] - dmx;
        const qy = dst[i][1] - dmy;
        dot += px * qx + py * qy;
        cross += px * qy - py * qx;
        norm += px * px + py * py;
    }
    const a = dot / norm;
    const b = cross / norm;
    return [a, b, dmx - (a * smx - b * smy), dmy - (b * smx + a * smy)];
}

/** A face's 512-number fingerprint, used only to recognise the people to leave clear. */
export async function embedFace(source, face) {
    await loadModels();
    const model = recogniser;
    // Crop around the face at a moderate size first, so the final warp doesn't alias
    const side = Math.max(face.x2 - face.x1, face.y2 - face.y1) * 2;
    const crop = { x: (face.x1 + face.x2) / 2 - side / 2, y: (face.y1 + face.y2) / 2 - side / 2 };
    const cropPx = 256;
    const s = cropPx / side;
    const first = canvas2d(cropPx, cropPx);
    first.ctx.imageSmoothingQuality = "high";
    first.ctx.drawImage(source, crop.x, crop.y, side, side, 0, 0, cropPx, cropPx);
    const kps = face.kps.map(([x, y]) => [(x - crop.x) * s, (y - crop.y) * s]);
    const [a, b, tx, ty] = similarity(kps, ARCFACE_POINTS);
    const { ctx } = canvas2d(112, 112, true);
    ctx.imageSmoothingQuality = "high";
    ctx.setTransform(a, b, -b, a, tx, ty);
    ctx.drawImage(first.canvas, 0, 0);
    const px = ctx.getImageData(0, 0, 112, 112).data;
    const n = 112 * 112;
    const f = new Float32Array(3 * n);
    for (let i = 0, p = 0; i < n; i++, p += 4) {
        f[i] = (px[p] - 127.5) / 127.5;
        f[i + n] = (px[p + 1] - 127.5) / 127.5;
        f[i + 2 * n] = (px[p + 2] - 127.5) / 127.5;
    }
    const input = new ort.Tensor("float32", f, [1, 3, 112, 112]);
    const out = await run(model, { [model.session.inputNames[0]]: input });
    const v = Float32Array.from(out[model.session.outputNames[0]].data);
    input.dispose?.();
    for (const t of Object.values(out)) t.dispose?.();
    let sumSq = 0;
    for (const x of v) sumSq += x * x;
    const length = Math.sqrt(sumSq) || 1;
    for (let i = 0; i < v.length; i++) v[i] /= length;
    return v;
}

export function cosine(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
    return sum;
}
