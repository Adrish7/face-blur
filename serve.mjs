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
};

createServer(async (req, res) => {
    try {
        let path = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
        if (path.endsWith("/")) path += "index.html";
        const file = normalize(join(root, path));
        if (!file.startsWith(root + sep) || file.includes(`${sep}node_modules${sep}`) || !(await stat(file)).isFile()) throw 404;
        const body = await readFile(file);
        res.writeHead(200, {
            "Content-Type": types[extname(file).toLowerCase()] || "application/octet-stream",
            "Cache-Control": "max-age=600",
        });
        res.end(body);
    } catch {
        res.writeHead(404, { "Content-Type": "text/plain" });
        res.end("Not found");
    }
}).listen(port, "127.0.0.1", () => console.log(`Face Blur is running at http://localhost:${port}  (Ctrl+C to stop)`));
