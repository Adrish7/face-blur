// A plain static server for trying the site locally: `npm start`, then open
// http://localhost:5177. It sends no special headers, like GitHub Pages, so
// what works here works there.
import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { extname, join, normalize, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(fileURLToPath(new URL(".", import.meta.url)));
const port = Number(process.env.PORT || 5177);

const types = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".onnx": "application/octet-stream",
    ".svg": "image/svg+xml",
    ".mp4": "video/mp4",
    ".webp": "image/webp",
    ".woff2": "font/woff2",
};

createServer(async (req, res) => {
    try {
        let path = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
        if (path.endsWith("/")) path += "index.html";
        const file = normalize(join(root, path));
        if (!file.startsWith(root + sep) || file.includes(`${sep}node_modules${sep}`) || !(await stat(file)).isFile()) throw 404;
        const body = await readFile(file);
        const headers = {
            "Content-Type": types[extname(file).toLowerCase()] || "application/octet-stream",
            "Cache-Control": "max-age=600",
            "Accept-Ranges": "bytes",
        };
        // Byte ranges, so videos can seek (GitHub Pages does the same)
        const range = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range || "");
        if (range && (range[1] || range[2])) {
            const size = body.length;
            const start = range[1] ? Number(range[1]) : Math.max(0, size - Number(range[2]));
            const end = range[1] && range[2] ? Math.min(Number(range[2]), size - 1) : size - 1;
            res.writeHead(206, { ...headers, "Content-Range": `bytes ${start}-${end}/${size}` });
            return res.end(body.subarray(start, end + 1));
        }
        res.writeHead(200, headers);
        res.end(body);
    } catch {
        res.writeHead(404, { "Content-Type": "text/plain" });
        res.end("Not found");
    }
}).listen(port, "127.0.0.1", () => console.log(`Face Blur is running at http://localhost:${port}  (Ctrl+C to stop)`));
