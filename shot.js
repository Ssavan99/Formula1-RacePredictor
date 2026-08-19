/**
 * Self-contained headless screenshot.
 *
 * Serves a directory from inside this process and screenshots a page in it.
 * Everything lives in one process because separately-backgrounded servers get
 * suspended in this environment, which made every earlier attempt time out.
 *
 *   node shot.js <dir> <page> <out.png> [width] [height] [waitMs] [scrollY]
 */
const http = require("http");
const fs = require("fs");
const path = require("path");
const puppeteer = require("puppeteer");

const TYPES = { ".html":"text/html", ".js":"application/javascript", ".mjs":"application/javascript",
  ".css":"text/css", ".json":"application/json", ".glb":"model/gltf-binary", ".wasm":"application/wasm",
  ".png":"image/png", ".jpg":"image/jpeg", ".webp":"image/webp", ".svg":"image/svg+xml" };

(async () => {
  const [dir, page, out, w = 1440, h = 900, wait = 8000, scrollY = 0] = process.argv.slice(2);
  const root = path.resolve(dir);

  const server = http.createServer((req, res) => {
    const rel = decodeURIComponent(req.url.split("?")[0]).replace(/^\/+/, "") || "index.html";
    const file = path.join(root, rel);
    if (!file.startsWith(root) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      res.writeHead(404); return res.end("not found");
    }
    res.writeHead(200, { "Content-Type": TYPES[path.extname(file)] || "application/octet-stream" });
    fs.createReadStream(file).pipe(res);
  });
  await new Promise(r => server.listen(0, r));
  const port = server.address().port;

  const browser = await puppeteer.launch({
    args: ["--no-sandbox", "--enable-webgl", "--use-gl=swiftshader", "--enable-unsafe-swiftshader"],
  });
  const tab = await browser.newPage();
  await tab.setViewport({ width: +w, height: +h, deviceScaleFactor: 1 });

  const errors = [];
  tab.on("console", m => m.type() === "error" && errors.push(m.text().slice(0, 150)));
  tab.on("pageerror", e => errors.push("PAGEERROR " + e.message.slice(0, 150)));

  await tab.goto(`http://127.0.0.1:${port}/${page}`, { waitUntil: "domcontentloaded", timeout: 60000 });
  if (+scrollY) await tab.evaluate(y => scrollTo(0, y), +scrollY);
  await new Promise(r => setTimeout(r, +wait));
  await tab.screenshot({ path: out });

  await browser.close();
  server.close();
  console.log("wrote " + out + (errors.length ? "\nerrors:\n  " + errors.join("\n  ") : "\nno console errors"));
})();
