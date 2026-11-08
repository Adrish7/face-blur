// Drawing the blur. A face is shrunk until it is about 6 blocks wide (no one
// is recognisable at that size), then smoothed back up inside an oval.

/** The oval that gets blurred. `size` is the Blur size slider (1 = default). */
export function blurShape(face, size) {
    if (face.manual) {
        return {
            cx: (face.x1 + face.x2) / 2,
            cy: (face.y1 + face.y2) / 2,
            rx: Math.abs(face.x2 - face.x1) / 2,
            ry: Math.abs(face.y2 - face.y1) / 2,
            angle: 0,
        };
    }
    const w = face.x2 - face.x1;
    const h = face.y2 - face.y1;
    const lift = 0.07 * h; // nudged up toward the forehead
    return {
        cx: (face.x1 + face.x2) / 2 + Math.sin(face.angle) * lift,
        cy: (face.y1 + face.y2) / 2 - Math.cos(face.angle) * lift,
        rx: w * 0.62 * size,
        ry: h * 0.66 * size,
        angle: face.angle,
    };
}

function blurOval(ctx, photo, s) {
    const r = Math.max(s.rx, s.ry);
    const pad = r * 0.7;
    const x = Math.max(0, Math.floor(s.cx - r - pad));
    const y = Math.max(0, Math.floor(s.cy - r - pad));
    const w = Math.min(photo.width, Math.ceil(s.cx + r + pad)) - x;
    const h = Math.min(photo.height, Math.ceil(s.cy + r + pad)) - y;
    if (w < 2 || h < 2) return;
    const cell = Math.max(2, (2 * Math.min(s.rx, s.ry)) / 6);
    const sw = Math.max(2, Math.round(w / cell));
    const sh = Math.max(2, Math.round(h / cell));
    const small = new OffscreenCanvas(sw, sh);
    const sctx = small.getContext("2d");
    sctx.imageSmoothingQuality = "high";
    sctx.drawImage(photo, x, y, w, h, 0, 0, sw, sh);
    const big = new OffscreenCanvas(w, h);
    const bctx = big.getContext("2d");
    bctx.imageSmoothingEnabled = true;
    bctx.imageSmoothingQuality = "high";
    if ("filter" in bctx) bctx.filter = `blur(${Math.max(1, Math.round(cell * 0.55))}px)`;
    bctx.drawImage(small, 0, 0, sw, sh, 0, 0, w, h);
    ctx.save();
    ctx.beginPath();
    ctx.ellipse(s.cx, s.cy, s.rx, s.ry, s.angle, 0, Math.PI * 2);
    ctx.clip();
    ctx.drawImage(big, x, y);
    ctx.restore();
}

export function drawPhoto(ctx, photo, faces, size, withBlur = true) {
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, photo.width, photo.height);
    ctx.drawImage(photo, 0, 0);
    if (!withBlur) return;
    for (const face of faces) if (face.blurred) blurOval(ctx, photo, blurShape(face, size));
}

/**
 * The finished photo, as a PNG for a PNG and a JPEG for anything else. A fresh
 * canvas carries no EXIF, so the GPS location and phone details are gone.
 */
export function exportPhoto(photo, faces, size, type = "image/jpeg") {
    const canvas = new OffscreenCanvas(photo.width, photo.height);
    drawPhoto(canvas.getContext("2d"), photo, faces, size, true);
    return canvas.convertToBlob(type === "image/png" ? { type } : { type, quality: 0.92 });
}
