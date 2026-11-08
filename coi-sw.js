// Lets the face models use several CPU threads. That needs a "cross-origin
// isolated" page, which takes two response headers that GitHub Pages can't
// send, so this service worker adds them to the site's own files. It caches
// nothing and sees nothing but this site's own files.

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

self.addEventListener("fetch", (event) => {
    const request = event.request;
    if (new URL(request.url).origin !== self.location.origin) return;
    if (request.cache === "only-if-cached" && request.mode !== "same-origin") return;
    event.respondWith(
        fetch(request).then((response) => {
            if (response.status === 0) return response;
            const headers = new Headers(response.headers);
            headers.set("Cross-Origin-Embedder-Policy", "require-corp");
            headers.set("Cross-Origin-Opener-Policy", "same-origin");
            return new Response(response.body, {
                status: response.status,
                statusText: response.statusText,
                headers,
            });
        }),
    );
});
