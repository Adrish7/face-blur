import { cosine, detectFaces, embedFace, loadModels } from "./faces.js";
import { decodePhoto, PHOTO_NAME } from "./decode.js";
import { blurShape, drawPhoto, exportPhoto } from "./blur.js";
import { makeZip } from "./zip.js";

// Blurs every face in a batch of photos, except the people the visitor asks to
// leave clear. Everything happens in this browser tab: nothing is uploaded and
// nothing is stored once the tab closes.

// A face this close (cosine) to a "leave clear" photo is that person. In
// testing, strangers scored up to 0.30 and the same person 0.39 to 0.53, so a
// side-on face can stay blurred (a click fixes it) but a stranger isn't left clear.
const MATCH = 0.4;
// Smaller faces (in photo pixels) are too soft to recognise, so they stay blurred
const MIN_RECOGNISE = 36;
// A "leave clear" photo needs a face at least this big to match against
const MIN_REFERENCE = 48;
// A ZIP tops out at 4 GB and the browser builds it in memory, so big batches split
const ZIP_PART_BYTES = 1.5 * 1024 ** 3;
const SVG = "http://www.w3.org/2000/svg";
const canSaveFolder = typeof window.showDirectoryPicker === "function";
const touchFirst = matchMedia("(pointer: coarse)").matches;

const $ = (id) => document.getElementById(id);
const els = {
    status: $("status"),
    progress: $("progress"),
    count: $("count"),
    position: $("position"),
    prev: $("prev"),
    next: $("next"),
    hint: $("hint"),
    dropText: $("dropzone-text"),
    progressBar: $("progress-bar"),
    size: $("size"),
    addFolder: $("add-folder"),
    addPhotos: $("add-photos"),
    clear: $("clear"),
    zip: $("zip"),
    save: $("save"),
    addPerson: $("add-person"),
    peopleHint: $("people-hint"),
    chips: $("chips"),
    list: $("list"),
    viewer: $("viewer"),
    empty: $("empty"),
    pickFolder: $("pick-folder"),
    pickPhotos: $("pick-photos"),
    stage: $("stage"),
    view: $("view"),
    overlay: $("overlay"),
    labels: $("labels"),
    busy: $("busy"),
    peek: $("peek"),
    undo: $("undo"),
    dropzone: $("dropzone"),
    toast: $("toast"),
    folderInput: $("folder-input"),
    photosInput: $("photos-input"),
    personInput: $("person-input"),
};

const state = {
    photos: [],
    people: [],
    currentId: null,
    nextId: 1,
    personCount: 0,
    ready: null,
    status: "Loading the face models…",
    working: false,
    // What the save or ZIP button is doing right now, and how far along it is
    saving: "",
    savingKind: "",
    saveProgress: 0,
    size: 1,
    zoom: 1,
    peek: false,
    // Photos the worker or the saver is using right now, so their images stay in memory
    pinned: new Set(),
    // A person a new photo is being picked for ("+ photo" on their card)
    addingTo: null,
};

const nextId = () => state.nextId++;
const message = (error) => (error instanceof Error ? error.message : String(error));
const current = () => state.photos.find((p) => p.id === state.currentId) ?? null;
const isPhotoFile = (file) => !file.name.startsWith(".") && (PHOTO_NAME.test(file.name) || file.type.startsWith("image/"));
const area = (f) => (f.x2 - f.x1) * (f.y2 - f.y1);
const recognisable = (f) => !f.manual && f.score >= 0.5 && Math.min(f.x2 - f.x1, f.y2 - f.y1) >= MIN_RECOGNISE;
const fingerprints = () => state.people.flatMap((person) => person.refs.filter((r) => r.fp).map((r) => ({ person, fp: r.fp })));
const personName = (person) => person.name.trim() || person.label;

let toastTimer = 0;
function toast(text) {
    els.toast.textContent = text;
    els.toast.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (els.toast.hidden = true), 6000);
}

// ---------- photo memory: a 24 MP photo is ~100 MB decoded, so keep 3 ----------
const bitmaps = new Map();
function remember(id, bitmap) {
    bitmaps.delete(id);
    bitmaps.set(id, bitmap);
    for (const [oldId, old] of bitmaps) {
        if (bitmaps.size <= 3) break;
        if (oldId === id || oldId === state.currentId || state.pinned.has(oldId)) continue;
        bitmaps.delete(oldId);
        old.close();
    }
}
async function getBitmap(p) {
    const cached = bitmaps.get(p.id);
    if (cached) {
        remember(p.id, cached);
        return cached;
    }
    const bitmap = await createImageBitmap(p.blob, { imageOrientation: "from-image" });
    remember(p.id, bitmap);
    return bitmap;
}
function forget(id) {
    bitmaps.get(id)?.close();
    bitmaps.delete(id);
}

async function shrink(bitmap, sx, sy, sw, sh, w, h) {
    const canvas = new OffscreenCanvas(w, h);
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingQuality = "high";
    ctx.fillStyle = "#888";
    ctx.fillRect(0, 0, w, h);
    ctx.drawImage(bitmap, sx, sy, sw, sh, 0, 0, w, h);
    return URL.createObjectURL(await canvas.convertToBlob({ type: "image/jpeg", quality: 0.8 }));
}
const photoThumb = (bitmap) => shrink(bitmap, 0, 0, bitmap.width, bitmap.height, 160, Math.max(1, Math.round((bitmap.height / bitmap.width) * 160)));
function faceThumb(bitmap, f) {
    const side = Math.max(f.x2 - f.x1, f.y2 - f.y1) * 1.5;
    return shrink(bitmap, (f.x1 + f.x2) / 2 - side / 2, (f.y1 + f.y2) / 2 - side / 2, side, side, 96, 96);
}

// ---------- matching ----------
function matchPhoto(p, refs) {
    let changed = false;
    for (const f of p.faces) {
        if (f.manual) continue;
        let best = null;
        let top = MATCH;
        if (f.fp) {
            for (const r of refs) {
                const sim = cosine(f.fp, r.fp);
                if (sim >= top) {
                    top = sim;
                    best = r.person;
                }
            }
        }
        const match = best?.id ?? null;
        const blurred = f.userSet ? f.blurred : !best;
        if (match !== f.match || blurred !== f.blurred) changed = true;
        f.match = match;
        f.blurred = blurred;
    }
    return changed;
}
function rematchAll() {
    const refs = fingerprints();
    for (const p of state.photos) if (p.status === "ready" && matchPhoto(p, refs)) p.saved = false;
}

// ---------- adding ----------
// "Trip/day 1/IMG_1.jpg" → top "Trip", dir "day 1/"
function placeOf(file, path) {
    const rel = (path || file.webkitRelativePath || "").replace(/^\/+/, "");
    const folders = rel ? rel.split("/").slice(0, -1) : [];
    return { top: folders[0] ?? null, dir: folders.slice(1).map((d) => `${d}/`).join("") };
}

function addPhotos(items) {
    const known = new Set(state.photos.map((p) => p.key));
    const all = items.filter(({ file }) => !file.name.startsWith("."));
    const fresh = all
        .filter(({ file }) => isPhotoFile(file))
        .map(({ file, path }) => {
            const place = placeOf(file, path);
            return { file, ...place, key: `${place.top ?? ""}/${place.dir}${file.name}|${file.size}|${file.lastModified}` };
        })
        .filter((x) => !known.has(x.key) && known.add(x.key))
        .sort((a, b) => a.key.localeCompare(b.key, undefined, { numeric: true }));
    const skipped = all.length - all.filter(({ file }) => isPhotoFile(file)).length;
    if (skipped) toast(`Skipped ${skipped} file${skipped === 1 ? "" : "s"} that ${skipped === 1 ? "isn't a photo" : "aren't photos"}.`);
    else if (!fresh.length && all.length) toast("Those photos are already in the list.");
    else if (!all.length) toast("No photos found there.");
    for (const x of fresh) {
        state.photos.push({
            id: nextId(),
            key: x.key,
            name: x.file.name,
            file: x.file,
            top: x.top,
            dir: x.dir,
            blob: null,
            width: 0,
            height: 0,
            faces: [],
            status: "queued",
            error: "",
            thumb: "",
            saved: false,
            fpTried: false,
            history: [],
        });
    }
    if (!current() && state.photos.length) select(state.photos[0].id);
    render();
    void pump();
}

function cameraName(name) {
    const base = name.replace(/\.[^.]+$/, "").trim();
    return /\d{3,}/.test(base) || /^(img|dsc|pxl|photo|image|screenshot)\b/i.test(base) ? "" : base.slice(0, 40);
}

function addPeople(files, person = null) {
    const photos = [...files].filter(isPhotoFile);
    if (!photos.length) return toast("Pick a photo of the person's face.");
    for (const file of photos) {
        const ref = { id: nextId(), file, status: "queued", fp: null, thumb: "" };
        if (person && state.people.includes(person)) {
            person.refs.push(ref);
            continue;
        }
        state.people.push({
            id: nextId(),
            label: `Person ${++state.personCount}`,
            name: cameraName(file.name),
            refs: [ref],
        });
    }
    render();
    void pump();
}

function removePerson(person) {
    state.people = state.people.filter((x) => x !== person);
    for (const r of person.refs) if (r.thumb) URL.revokeObjectURL(r.thumb);
    rematchAll();
    render();
}

// Dropped folders arrive as directory entries: walk them for files. Entries
// must be read during the drop event itself, so that part stays synchronous.
function filesFromDrop(data) {
    const entries = [...data.items].filter((i) => i.kind === "file").map((i) => i.webkitGetAsEntry?.() ?? null);
    if (!entries.length || entries.some((e) => !e)) return Promise.resolve([...data.files].map((file) => ({ file })));
    const out = [];
    const walk = async (entry) => {
        if (entry.isFile) {
            out.push({ file: await new Promise((res, rej) => entry.file(res, rej)), path: entry.fullPath });
        } else if (entry.isDirectory) {
            const reader = entry.createReader();
            for (;;) {
                const batch = await new Promise((res, rej) => reader.readEntries(res, rej));
                if (!batch.length) break;
                for (const e of batch) await walk(e);
            }
        }
    };
    return Promise.all(entries.map(walk)).then(() => out);
}

// ---------- the worker: one job at a time, people first ----------
async function processRef(person, ref) {
    ref.status = "working";
    render();
    try {
        const { bitmap } = await decodePhoto(ref.file).catch((error) => {
            throw new Error(`${ref.file.name}: ${message(error)}`);
        });
        try {
            const faces = (await detectFaces(bitmap, { thorough: false })).sort((a, b) => area(b) - area(a));
            const face = faces[0];
            if (!face) throw new Error(`No face found in ${ref.file.name}. Try a clearer, front-on photo.`);
            if (Math.min(face.x2 - face.x1, face.y2 - face.y1) < MIN_REFERENCE)
                throw new Error(`The face in ${ref.file.name} is too small. Try a closer photo.`);
            ref.fp = await embedFace(bitmap, face);
            ref.thumb = await faceThumb(bitmap, face);
            ref.status = "ready";
            if (faces.length > 1) toast(`${ref.file.name} has ${faces.length} faces, so the biggest one was used.`);
        } finally {
            bitmap.close();
        }
    } catch (error) {
        console.warn(error);
        if (state.people.includes(person)) toast(message(error));
        person.refs = person.refs.filter((r) => r !== ref);
        if (!person.refs.length) state.people = state.people.filter((x) => x !== person);
    }
    rematchAll();
    render();
}

async function embedAll(p, bitmap) {
    for (const f of p.faces) {
        if (!recognisable(f) || f.fp) continue;
        try {
            f.fp = await embedFace(bitmap, f);
        } catch (error) {
            console.warn(error);
        }
    }
    p.fpTried = true;
}

async function processPhoto(p) {
    const gone = () => !state.photos.includes(p);
    state.pinned.add(p.id);
    try {
        p.status = "decoding";
        render();
        const { blob, bitmap } = await decodePhoto(p.file);
        if (gone()) return bitmap.close();
        Object.assign(p, { blob, width: bitmap.width, height: bitmap.height, status: "detecting" });
        remember(p.id, bitmap);
        p.thumb = await photoThumb(bitmap);
        render();
        let id = 1;
        p.faces = (await detectFaces(bitmap)).map((found) => ({
            ...found,
            id: id++,
            blurred: true,
            match: null,
            manual: false,
            userSet: false,
            fp: null,
        }));
        if (fingerprints().length) await embedAll(p, bitmap);
        matchPhoto(p, fingerprints());
        p.status = "ready";
    } catch (error) {
        console.error(error);
        p.status = "error";
        p.error = message(error);
    } finally {
        state.pinned.delete(p.id);
    }
    if (!gone()) render();
}

// A person was added after this photo was done: fingerprint its faces now
async function fingerprintPhoto(p) {
    state.pinned.add(p.id);
    try {
        await embedAll(p, await getBitmap(p));
    } catch (error) {
        console.warn(error);
        p.fpTried = true;
    } finally {
        state.pinned.delete(p.id);
    }
    if (matchPhoto(p, fingerprints())) p.saved = false;
    render();
}

async function pump() {
    if (state.working) return;
    state.working = true;
    render();
    try {
        const ok = await state.ready;
        for (;;) {
            if (!ok) {
                for (const p of state.photos)
                    if (p.status === "queued") Object.assign(p, { status: "error", error: "The face models didn't load" });
                state.people = [];
                break;
            }
            const person = state.people.find((x) => x.refs.some((r) => r.status === "queued"));
            if (person) {
                await processRef(
                    person,
                    person.refs.find((r) => r.status === "queued"),
                );
                continue;
            }
            const photo = state.photos.find((x) => x.status === "queued");
            if (photo) {
                await processPhoto(photo);
                continue;
            }
            const behind =
                fingerprints().length && state.photos.find((x) => x.status === "ready" && !x.fpTried && x.faces.some(recognisable));
            if (behind) {
                await fingerprintPhoto(behind);
                continue;
            }
            break;
        }
    } finally {
        state.working = false;
        render();
    }
}

// ---------- editing ----------
function changed(p) {
    p.saved = false;
    render();
}
function toggleFace(p, id) {
    const f = p.faces.find((x) => x.id === id);
    if (!f) return;
    if (f.manual) {
        p.faces = p.faces.filter((x) => x !== f);
        p.history.push({ type: "remove", face: f });
    } else {
        p.history.push({ type: "toggle", id, before: { blurred: f.blurred, userSet: f.userSet } });
        f.blurred = !f.blurred;
        f.userSet = true;
    }
    changed(p);
}
function undo() {
    const p = current();
    const last = p?.history.pop();
    if (!p || !last) return;
    if (last.type === "toggle") Object.assign(p.faces.find((x) => x.id === last.id), last.before);
    if (last.type === "add") p.faces = p.faces.filter((x) => x.id !== last.id);
    if (last.type === "remove") p.faces.push(last.face);
    changed(p);
}

// ---------- saving ----------
function stamp() {
    const d = new Date();
    const pad = (n) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}.${pad(d.getMinutes())}`;
}

// One folder out, laid out like the folder that went in. PNGs stay PNG, the rest become JPEG.
function plan(photos) {
    const tops = new Set(photos.map((p) => p.top));
    const single = tops.size === 1 ? [...tops][0] : undefined;
    const root = single ? `${single} (blurred)` : `Blurred photos ${stamp()}`;
    const used = new Set();
    const files = photos.map((p) => {
        const png = /\.png$/i.test(p.name) || p.file.type === "image/png";
        const ext = png ? "png" : "jpg";
        const dir = single !== undefined ? p.dir : `${p.top ? `${p.top}/` : ""}${p.dir}`;
        const base = p.name.replace(/\.[^.]+$/, "") || "photo";
        let path = `${dir}${base}.${ext}`;
        for (let n = 2; used.has(path.toLowerCase()); n++) path = `${dir}${base} (${n}).${ext}`;
        used.add(path.toLowerCase());
        return { photo: p, path, type: png ? "image/png" : "image/jpeg" };
    });
    return { root, files };
}

async function finished(photo, type) {
    state.pinned.add(photo.id);
    try {
        return await exportPhoto(await getBitmap(photo), photo.faces, state.size, type);
    } finally {
        state.pinned.delete(photo.id);
    }
}

const readyPhotos = () => state.photos.filter((p) => p.status === "ready");
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

async function saveToFolder() {
    const ready = readyPhotos();
    if (!ready.length || state.saving) return;
    let parent;
    try {
        parent = await window.showDirectoryPicker({ id: "face-blur-save", mode: "readwrite", startIn: "pictures" });
    } catch (error) {
        if (error.name !== "AbortError") toast(`Couldn't use that folder: ${message(error)}`);
        return;
    }
    const { root, files } = plan(ready);
    let name = root;
    try {
        for (let n = 2; await exists(parent, name); n++) name = `${root} ${n}`;
        const out = await parent.getDirectoryHandle(name, { create: true });
        const dirs = new Map([["", out]]);
        for (let i = 0; i < files.length; i++) {
            Object.assign(state, { saving: `Saving ${i + 1} of ${files.length}`, savingKind: "folder", saveProgress: i / files.length });
            render();
            const { photo, path, type } = files[i];
            const blob = await finished(photo, type);
            const folders = path.split("/");
            const file = folders.pop();
            let dir = out;
            for (let j = 0; j < folders.length; j++) {
                const key = folders.slice(0, j + 1).join("/");
                if (!dirs.has(key)) dirs.set(key, await dir.getDirectoryHandle(folders[j], { create: true }));
                dir = dirs.get(key);
            }
            const writable = await (await dir.getFileHandle(file, { create: true })).createWritable();
            await writable.write(blob);
            await writable.close();
            photo.saved = true;
        }
        toast(`Saved ${plural(files.length, "photo")} to the “${name}” folder.`);
    } catch (error) {
        console.error(error);
        toast(`Saving stopped: ${message(error)}`);
    } finally {
        Object.assign(state, { saving: "", savingKind: "", saveProgress: 0 });
        render();
    }
}

async function exists(parent, name) {
    try {
        await parent.getDirectoryHandle(name);
        return true;
    } catch (error) {
        if (error.name === "NotFoundError") return false;
        if (error.name === "TypeMismatchError") return true;
        throw error;
    }
}

function download(blob, name) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = name;
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 120_000);
}

async function downloadZip() {
    const ready = readyPhotos();
    if (!ready.length || state.saving) return;
    const { root, files } = plan(ready);
    let part = [];
    let bytes = 0;
    let parts = 0;
    const flush = async (last) => {
        parts++;
        Object.assign(state, { saving: "Making the ZIP", saveProgress: 1 });
        render();
        const zip = await makeZip(part);
        download(zip, last && parts === 1 ? `${root}.zip` : `${root} (part ${parts}).zip`);
        for (const entry of part) entry.photo.saved = true;
        part = [];
        bytes = 0;
    };
    try {
        for (let i = 0; i < files.length; i++) {
            Object.assign(state, { saving: `Preparing ${i + 1} of ${files.length}`, savingKind: "zip", saveProgress: i / files.length });
            render();
            const { photo, path, type } = files[i];
            const blob = await finished(photo, type);
            if (part.length && bytes + blob.size > ZIP_PART_BYTES) await flush(false);
            part.push({ name: `${root}/${path}`, blob, photo });
            bytes += blob.size;
        }
        await flush(true);
        if (parts > 1) toast(`That was a big batch, so it came as ${parts} ZIP files.`);
    } catch (error) {
        console.error(error);
        toast(`Making the ZIP stopped: ${message(error)}`);
    } finally {
        Object.assign(state, { saving: "", savingKind: "", saveProgress: 0 });
        render();
    }
}

function clearAll() {
    if (state.photos.some((p) => p.status === "ready" && !p.saved) && !confirm("Clear these photos? Some haven't been saved yet."))
        return;
    for (const p of state.photos) {
        if (p.thumb) URL.revokeObjectURL(p.thumb);
        forget(p.id);
    }
    state.photos = [];
    state.currentId = null;
    render();
}

// ---------- viewer ----------
function select(id) {
    state.currentId = id;
    state.zoom = 1;
    render();
    els.list.querySelector('[aria-current="true"]')?.scrollIntoView({ block: "nearest", inline: "nearest" });
}
function step(by) {
    const next = state.photos[state.photos.findIndex((p) => p.id === state.currentId) + by];
    if (next) select(next.id);
}

function fitScale(p) {
    const pad = touchFirst ? 16 : 32;
    return Math.min((els.viewer.clientWidth - pad) / p.width, (els.viewer.clientHeight - pad) / p.height);
}

function layout() {
    const p = current();
    if (!p?.width) return;
    const s = fitScale(p) * state.zoom;
    const w = `${Math.max(1, Math.round(p.width * s))}px`;
    const h = `${Math.max(1, Math.round(p.height * s))}px`;
    for (const el of [els.view, els.overlay, els.labels]) {
        el.style.width = w;
        el.style.height = h;
    }
    els.overlay.setAttribute("viewBox", `0 0 ${p.width} ${p.height}`);
    // A finger pans a zoomed-in photo; at full view it draws
    els.overlay.style.touchAction = state.zoom > 1 ? "pan-x pan-y" : "none";
    settleTags();
}

function zoomTo(zoom, anchor) {
    const v = els.viewer;
    const old = state.zoom;
    state.zoom = Math.min(8, Math.max(1, zoom));
    const ax = anchor?.x ?? v.clientWidth / 2;
    const ay = anchor?.y ?? v.clientHeight / 2;
    const fx = (v.scrollLeft + ax) / old;
    const fy = (v.scrollTop + ay) / old;
    layout();
    v.scrollLeft = fx * state.zoom - ax;
    v.scrollTop = fy * state.zoom - ay;
}

let paintedKey = "";
async function paint(p) {
    const key = `${p.id}|${p.faces.map((f) => `${f.id}${f.blurred ? 1 : 0}`).join(",")}|${state.size}|${state.peek}`;
    if (key === paintedKey) return;
    paintedKey = key;
    let bitmap;
    try {
        bitmap = await getBitmap(p);
    } catch (error) {
        console.error(error);
        paintedKey = "";
        return;
    }
    if (current() !== p || paintedKey !== key) return;
    if (els.view.width !== bitmap.width || els.view.height !== bitmap.height) {
        els.view.width = bitmap.width;
        els.view.height = bitmap.height;
    }
    drawPhoto(els.view.getContext("2d"), bitmap, p.faces, state.size, !state.peek);
}

let overlayKey = "";
function drawOverlay(p) {
    const names = new Map(state.people.map((person) => [person.id, personName(person)]));
    const key = `${p.id}|${p.status}|${p.faces.map((f) => `${f.id}${f.blurred ? 1 : 0}${f.match ?? ""}`).join(",")}|${state.size}|${[...names.values()].join(",")}`;
    if (key === overlayKey) return;
    overlayKey = key;
    const nodes = [];
    const labels = [];
    for (const f of p.faces) {
        const s = blurShape(f, state.size);
        const g = document.createElementNS(SVG, "g");
        g.classList.add("face", f.blurred ? "blurred" : "clear");
        if (f.match && !f.blurred) g.classList.add("known");
        g.dataset.id = f.id;
        const e = document.createElementNS(SVG, "ellipse");
        e.setAttribute("cx", s.cx);
        e.setAttribute("cy", s.cy);
        e.setAttribute("rx", s.rx);
        e.setAttribute("ry", s.ry);
        e.setAttribute("transform", `rotate(${(s.angle * 180) / Math.PI} ${s.cx} ${s.cy})`);
        const title = document.createElementNS(SVG, "title");
        title.textContent = f.manual ? "Click to remove this blur" : f.blurred ? "Click to unblur" : "Click to blur";
        e.append(title);
        g.append(e);
        nodes.push(g);
        if (!f.blurred) {
            const tag = document.createElement("span");
            const known = f.match && names.get(f.match);
            tag.className = `tag${known ? " known" : ""}`;
            tag.textContent = known || "Not blurred";
            tag.title = known ? `${known} is left clear` : "Not blurred";
            tag.style.left = `${(s.cx / p.width) * 100}%`;
            tag.dataset.above = `${((s.cy - s.ry) / p.height) * 100}%`;
            tag.dataset.below = `${((s.cy + s.ry) / p.height) * 100}%`;
            tag.style.top = tag.dataset.above;
            labels.push(tag);
        }
    }
    els.overlay.replaceChildren(...nodes);
    els.labels.replaceChildren(...labels);
    settleTags();
}

// Two faces side by side would stack their labels: drop the later one below its face
function settleTags() {
    const top = els.viewer.getBoundingClientRect().top;
    const placed = [];
    for (const tag of els.labels.children) {
        tag.classList.remove("below");
        tag.style.top = tag.dataset.above;
        let box = tag.getBoundingClientRect();
        const hits = (b) => placed.some((p) => b.left < p.right + 4 && b.right > p.left - 4 && b.top < p.bottom + 2 && b.bottom > p.top - 2);
        if (box.top < top || hits(box)) {
            tag.classList.add("below");
            tag.style.top = tag.dataset.below;
            box = tag.getBoundingClientRect();
        }
        placed.push(box);
    }
}

function renderViewer() {
    const p = current();
    els.empty.hidden = state.photos.length > 0;
    els.stage.hidden = !p;
    if (!p) {
        paintedKey = overlayKey = "";
        return;
    }
    const waiting = ["queued", "decoding", "detecting"].includes(p.status);
    els.busy.hidden = !waiting && p.status !== "error";
    els.busy.classList.toggle("working", waiting);
    els.busy.classList.toggle("error", p.status === "error");
    els.busy.textContent =
        p.status === "error"
            ? p.error
            : p.status === "queued"
              ? "Waiting its turn…"
              : p.status === "decoding"
                ? "Opening the photo…"
                : "Finding faces…";
    els.stage.classList.toggle("peek", state.peek);
    els.stage.classList.toggle("no-photo", !p.width);
    if (!p.width) return;
    layout();
    drawOverlay(p);
    void paint(p);
}

// ---------- the rest of the page ----------
const listItems = new Map();
function summary(p) {
    if (p.status === "queued") return "Waiting…";
    if (p.status === "decoding") return "Opening…";
    if (p.status === "detecting") return "Finding faces…";
    if (p.status === "error") return p.error;
    if (!p.faces.length) return "No faces found";
    const blurred = p.faces.filter((f) => f.blurred).length;
    const known = [...new Set(p.faces.filter((f) => !f.blurred && f.match).map((f) => names.get(f.match)).filter(Boolean))];
    const others = p.faces.filter((f) => !f.blurred && !(f.match && names.has(f.match))).length;
    return [blurred && `${blurred} blurred`, known.length && `${known.join(", ")} left clear`, others && `${others} not blurred`]
        .filter(Boolean)
        .join(" · ");
}
let names = new Map();

function renderList() {
    names = new Map(state.people.map((person) => [person.id, personName(person)]));
    let prev = null;
    const seen = new Set();
    for (const p of state.photos) {
        let item = listItems.get(p.id);
        if (!item) {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "item";
            button.addEventListener("click", () => select(p.id));
            const img = document.createElement("img");
            img.alt = "";
            img.loading = "lazy";
            const meta = document.createElement("span");
            meta.className = "meta";
            const name = document.createElement("span");
            name.className = "name";
            name.textContent = `${p.dir}${p.name}`;
            name.title = name.textContent;
            const sub = document.createElement("span");
            sub.className = "sub";
            meta.append(name, sub);
            button.append(img, meta);
            item = { button, img, sub };
            listItems.set(p.id, item);
        }
        if (p.thumb && item.img.getAttribute("src") !== p.thumb) item.img.src = p.thumb;
        const text = summary(p);
        if (item.sub.textContent !== text) {
            item.sub.textContent = text;
            item.sub.title = text;
        }
        const ready = p.status === "ready";
        // Amber when a face nobody asked to leave clear is unblurred, so it gets a second look
        const warn = ready && p.faces.some((f) => !f.blurred && !(f.match && names.has(f.match)));
        item.sub.classList.toggle("err", p.status === "error");
        item.sub.classList.toggle("warn", warn);
        item.sub.classList.toggle("known", ready && !warn && p.faces.some((f) => !f.blurred));
        item.sub.classList.toggle("working", p.status === "decoding" || p.status === "detecting");
        item.button.setAttribute("aria-current", String(p.id === state.currentId));
        if (item.button.previousElementSibling !== prev || item.button.parentNode !== els.list) {
            if (prev) prev.after(item.button);
            else els.list.prepend(item.button);
        }
        prev = item.button;
        seen.add(p.id);
    }
    for (const [id, item] of listItems) {
        if (seen.has(id)) continue;
        item.button.remove();
        listItems.delete(id);
    }
}

function iconButton(icon, label) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "icon-btn";
    button.title = label;
    button.setAttribute("aria-label", label);
    button.innerHTML = `<svg class="icon" aria-hidden="true"><use href="#${icon}"/></svg>`;
    return button;
}

const chipItems = new Map();
function renderPeople() {
    const seen = new Set();
    for (const person of state.people) {
        let chip = chipItems.get(person.id);
        if (!chip) {
            const root = document.createElement("div");
            root.className = "chip";
            const faces = document.createElement("span");
            faces.className = "faces";
            const name = document.createElement("input");
            name.className = "name";
            name.value = person.name;
            name.placeholder = person.label;
            name.maxLength = 40;
            name.setAttribute("aria-label", "Name (only shown to you)");
            name.addEventListener("input", () => {
                person.name = name.value;
                render();
            });
            const more = iconButton("i-image-plus", "Add another photo of this person");
            more.addEventListener("click", () => {
                state.addingTo = person;
                els.personInput.click();
            });
            const remove = iconButton("i-x", "Remove this person");
            remove.addEventListener("click", () => removePerson(person));
            const actions = document.createElement("span");
            actions.className = "person-actions";
            actions.append(more, remove);
            root.append(faces, name, actions);
            els.chips.append(root);
            chip = { root, faces, keys: "" };
            chipItems.set(person.id, chip);
        }
        const keys = person.refs.map((r) => `${r.id}${r.status}`).join(",");
        if (keys !== chip.keys) {
            chip.keys = keys;
            chip.faces.replaceChildren(
                ...person.refs.map((r) => {
                    if (r.thumb) return Object.assign(document.createElement("img"), { src: r.thumb, alt: "", title: r.file.name });
                    return Object.assign(document.createElement("span"), { className: "ph spin", title: "Finding the face…" });
                }),
            );
        }
        seen.add(person.id);
    }
    for (const [id, chip] of chipItems) {
        if (seen.has(id)) continue;
        chip.root.remove();
        chipItems.delete(id);
    }
    els.peopleHint.hidden = state.people.length > 0 && state.photos.length > 0;
}

const setLabel = (button, text) => {
    const span = button.querySelector("span");
    if (span.textContent !== text) span.textContent = text;
};

function renderBar() {
    els.status.textContent = state.status;
    const total = state.photos.length;
    const done = state.photos.filter((p) => p.status === "ready" || p.status === "error").length;
    const failed = state.photos.filter((p) => p.status === "error").length;
    const ready = done - failed;
    const busy = state.working || done < total || !!state.saving;
    document.body.classList.toggle("has-photos", total > 0);

    // A hairline under the top bar while photos are being blurred or saved
    els.progress.hidden = !total || !(done < total || state.saving);
    els.progressBar.style.transform = `scaleX(${state.saving ? state.saveProgress : done / Math.max(1, total)})`;
    els.count.textContent = !total
        ? ""
        : done < total
          ? `${done} of ${total}`
          : state.working
            ? `${total} · matching`
            : failed
              ? `${total} · ${failed} couldn't open`
              : String(total);

    els.addFolder.hidden = touchFirst;
    els.zip.hidden = !total;
    els.save.hidden = !total || !canSaveFolder;
    els.zip.classList.toggle("primary", !canSaveFolder);
    els.zip.classList.toggle("secondary", canSaveFolder);
    els.save.disabled = els.zip.disabled = busy || !ready;
    setLabel(els.save, state.savingKind === "folder" ? `${state.saving}…` : "Save to folder");
    setLabel(els.zip, state.savingKind === "zip" ? `${state.saving}…` : "Download ZIP");
    els.clear.disabled = !!state.saving;

    const at = state.photos.findIndex((p) => p.id === state.currentId);
    els.position.textContent = total && at >= 0 ? `${at + 1} / ${total}` : "";
    els.prev.disabled = at <= 0;
    els.next.disabled = at < 0 || at >= total - 1;
    els.undo.disabled = !current()?.history.length;
}

function render() {
    renderBar();
    renderPeople();
    renderList();
    renderViewer();
}

// ---------- events ----------
els.pickFolder.hidden = touchFirst;
if (touchFirst) {
    els.hint.innerHTML = "<b>Tap a face</b> to blur or unblur it · <b>Drag</b> to blur one it missed";
    els.pickPhotos.classList.replace("secondary", "primary");
}
els.prev.addEventListener("click", () => step(-1));
els.next.addEventListener("click", () => step(1));

const pick = (input) => () => input.click();
els.pickFolder.addEventListener("click", pick(els.folderInput));
els.addFolder.addEventListener("click", pick(els.folderInput));
els.pickPhotos.addEventListener("click", pick(els.photosInput));
els.addPhotos.addEventListener("click", pick(els.photosInput));
els.addPerson.addEventListener("click", () => {
    state.addingTo = null;
    els.personInput.click();
});
for (const input of [els.folderInput, els.photosInput]) {
    input.addEventListener("change", () => {
        addPhotos([...input.files].map((file) => ({ file })));
        input.value = "";
    });
}
els.personInput.addEventListener("change", () => {
    addPeople(els.personInput.files, state.addingTo);
    state.addingTo = null;
    els.personInput.value = "";
});
els.save.addEventListener("click", saveToFolder);
els.zip.addEventListener("click", downloadZip);
els.clear.addEventListener("click", clearAll);
els.undo.addEventListener("click", undo);
els.size.addEventListener("input", () => {
    state.size = Number(els.size.value);
    for (const p of state.photos) p.saved = false;
    render();
});

const setPeek = (on) => {
    if (state.peek === on) return;
    state.peek = on;
    els.peek.setAttribute("aria-pressed", String(on));
    render();
};
els.peek.addEventListener("pointerdown", () => setPeek(true));
for (const type of ["pointerup", "pointerleave", "pointercancel"]) els.peek.addEventListener(type, () => setPeek(false));

$("zoom-in").addEventListener("click", () => zoomTo(state.zoom * 1.5));
$("zoom-out").addEventListener("click", () => zoomTo(state.zoom / 1.5));
$("zoom-fit").addEventListener("click", () => zoomTo(1));
els.viewer.addEventListener(
    "wheel",
    (e) => {
        if (!e.ctrlKey && !e.metaKey) return;
        e.preventDefault();
        const r = els.viewer.getBoundingClientRect();
        zoomTo(state.zoom * Math.exp(-e.deltaY * 0.01), { x: e.clientX - r.left, y: e.clientY - r.top });
    },
    { passive: false },
);
new ResizeObserver(() => layout()).observe(els.viewer);

// Click a face to blur or unblur it; drag anywhere else to blur a face it missed
const svgPoint = (e) => {
    const pt = new DOMPoint(e.clientX, e.clientY).matrixTransform(els.overlay.getScreenCTM().inverse());
    return { x: pt.x, y: pt.y };
};
let draft = null;
els.overlay.addEventListener("click", (e) => {
    const p = current();
    const hit = e.target.closest?.(".face");
    if (p?.status === "ready" && hit) toggleFace(p, Number(hit.dataset.id));
});
els.overlay.addEventListener("pointerdown", (e) => {
    const p = current();
    if (!p || p.status !== "ready" || e.button !== 0 || e.target.closest?.(".face")) return;
    if (e.pointerType === "touch" && state.zoom > 1) return;
    const start = svgPoint(e);
    const el = document.createElementNS(SVG, "ellipse");
    el.setAttribute("class", "draft");
    els.overlay.append(el);
    draft = { start, end: start, el };
    els.overlay.setPointerCapture(e.pointerId);
});
els.overlay.addEventListener("pointermove", (e) => {
    if (!draft) return;
    draft.end = svgPoint(e);
    const { start, end, el } = draft;
    el.setAttribute("cx", (start.x + end.x) / 2);
    el.setAttribute("cy", (start.y + end.y) / 2);
    el.setAttribute("rx", Math.abs(end.x - start.x) / 2);
    el.setAttribute("ry", Math.abs(end.y - start.y) / 2);
});
const endDraft = (keep) => {
    if (!draft) return;
    const { start, end, el } = draft;
    draft = null;
    el.remove();
    const p = current();
    if (!keep || !p?.width) return;
    const min = 8 / (fitScale(p) * state.zoom);
    if (Math.abs(end.x - start.x) < min || Math.abs(end.y - start.y) < min) return;
    const id = Math.max(0, ...p.faces.map((f) => f.id)) + 1;
    p.faces.push({
        id,
        manual: true,
        blurred: true,
        userSet: true,
        match: null,
        fp: null,
        score: 1,
        angle: 0,
        kps: [],
        x1: Math.min(start.x, end.x),
        y1: Math.min(start.y, end.y),
        x2: Math.max(start.x, end.x),
        y2: Math.max(start.y, end.y),
    });
    p.history.push({ type: "add", id });
    overlayKey = "";
    changed(p);
};
els.overlay.addEventListener("pointerup", () => endDraft(true));
els.overlay.addEventListener("pointercancel", () => endDraft(false));

const typing = (t) => t instanceof HTMLTextAreaElement || (t instanceof HTMLInputElement && t.type !== "checkbox");
window.addEventListener("keydown", (e) => {
    if (typing(e.target) || !state.photos.length) return;
    if (e.code === "Space") {
        e.preventDefault();
        setPeek(true);
    } else if (e.key === "ArrowRight" || e.key === "ArrowDown") {
        e.preventDefault();
        step(1);
    } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
        e.preventDefault();
        step(-1);
    } else if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "z") {
        e.preventDefault();
        undo();
    } else if (e.metaKey || e.ctrlKey || e.altKey) return;
    else if (e.key === "+" || e.key === "=") zoomTo(state.zoom * 1.5);
    else if (e.key === "-") zoomTo(state.zoom / 1.5);
    else if (e.key === "0") zoomTo(1);
});
window.addEventListener("keyup", (e) => {
    if (e.code === "Space") setPeek(false);
});
window.addEventListener("blur", () => setPeek(false));

// Drop photos anywhere, or a person's photo onto "Leave clear"
let dragDepth = 0;
const hasFiles = (e) => !!e.dataTransfer && [...e.dataTransfer.types].includes("Files");
const onPeople = (e) => !!e.target?.closest?.(".people");
window.addEventListener("dragenter", (e) => {
    if (!hasFiles(e)) return;
    dragDepth++;
    els.dropzone.hidden = false;
});
window.addEventListener("dragleave", (e) => {
    if (!hasFiles(e)) return;
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) els.dropzone.hidden = true;
});
window.addEventListener("dragover", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    const people = onPeople(e);
    els.dropzone.classList.toggle("for-people", people);
    els.dropText.textContent = people ? "Drop to leave this person clear" : "Drop to blur these photos";
});
window.addEventListener("drop", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    dragDepth = 0;
    els.dropzone.hidden = true;
    const people = onPeople(e);
    void filesFromDrop(e.dataTransfer).then((items) => (people ? addPeople(items.map((x) => x.file)) : addPhotos(items)));
});

window.addEventListener("beforeunload", (e) => {
    if (!state.photos.some((p) => p.status === "ready" && !p.saved) && !state.saving) return;
    e.preventDefault();
    e.returnValue = "";
});

// ---------- start ----------
state.ready = (async () => {
    try {
        const where = await loadModels((fraction) => {
            state.status = `Downloading the face models (first visit only) · ${Math.round(fraction * 100)}%`;
            els.status.textContent = state.status;
        });
        state.status = `Ready · runs on your ${where} · nothing leaves this device`;
        return true;
    } catch (error) {
        console.error(error);
        state.status = `Couldn't load the face models (${message(error)}). Reload the page to try again.`;
        return false;
    } finally {
        render();
    }
})();
render();
void pump();
