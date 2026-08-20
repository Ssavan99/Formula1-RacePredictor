# Vendored dependencies

`three.module.min.js` — [three.js](https://threejs.org/) r160, MIT licensed.

Vendored rather than loaded from a CDN so the site has no external runtime
dependency: it cannot break because someone else's CDN changed, and it works
offline. ~654 KB raw, ~160 KB over the wire once GitHub Pages gzips it, and it
is loaded lazily so it never blocks the data on the page.
