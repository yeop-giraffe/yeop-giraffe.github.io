import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('./site/', import.meta.url));
const port = Number(process.env.PORT || 4173);
let revision = 0;
const watcher = fs.watch(root, { recursive: true }, () => revision++);
const types = { '.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.pdf': 'application/pdf', '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', '.png': 'image/png', '.jpg': 'image/jpeg', '.svg': 'image/svg+xml', '.mp4': 'video/mp4', '.webm': 'video/webm' };
const server = http.createServer(async (request, response) => {
  response.setHeader('Cache-Control', 'no-store');
  if (request.method !== 'GET' && request.method !== 'HEAD') {
    response.writeHead(405).end();
    return;
  }
  try {
    const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
    if (pathname === '/__preview_revision') {
      response.setHeader('Content-Type', 'application/json');
      response.end(JSON.stringify({ revision }));
      return;
    }
    const filename = path.resolve(root, '.' + (pathname.endsWith('/') ? pathname + 'index.html' : pathname));
    const relative = path.relative(root, filename);
    if (relative.startsWith('..') || path.isAbsolute(relative)) {
      response.writeHead(403).end('Forbidden');
      return;
    }
    const extension = path.extname(filename).toLowerCase();
    response.setHeader('Content-Type', types[extension] || 'application/octet-stream');
    if (extension === '.mp4' || extension === '.webm') {
      const { size } = await fs.promises.stat(filename);
      response.setHeader('Accept-Ranges', 'bytes');
      let start = 0;
      let end = size - 1;
      const range = request.headers.range;
      if (range) {
        const match = /^bytes=(\d*)-(\d*)$/.exec(range);
        if (!match || (!match[1] && !match[2])) {
          response.writeHead(416, { 'Content-Range': `bytes */${size}` }).end();
          return;
        }
        if (match[1]) {
          start = Number(match[1]);
          if (match[2]) end = Math.min(Number(match[2]), size - 1);
        } else {
          const suffix = Number(match[2]);
          start = suffix > 0 ? Math.max(0, size - suffix) : size;
        }
        if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start > end || start >= size) {
          response.writeHead(416, { 'Content-Range': `bytes */${size}` }).end();
          return;
        }
        response.setHeader('Content-Range', `bytes ${start}-${end}/${size}`);
      }
      response.setHeader('Content-Length', size ? end - start + 1 : 0);
      response.writeHead(range ? 206 : 200);
      if (request.method === 'HEAD' || !size) {
        response.end();
        return;
      }
      const stream = fs.createReadStream(filename, { start, end });
      stream.on('error', () => response.destroy());
      response.on('close', () => stream.destroy());
      stream.pipe(response);
      return;
    }
    let data = await fs.promises.readFile(filename);
    if (extension === '.html') {
      const reload = `<script>let previewRevision=${revision};setInterval(async()=>{try{const response=await fetch('/__preview_revision');if(!response.ok)return;const next=await response.json();if(next.revision!==previewRevision)location.reload();}catch{}},1000);</script>`;
      data = data.toString().replace('</body>', reload + '</body>');
    }
    response.end(request.method === 'HEAD' ? undefined : data);
  } catch (error) {
    response.writeHead(error.code === 'ENOENT' || error.code === 'EISDIR' ? 404 : 400).end('Page unavailable');
  }
});
server.on('error', error => { console.error(error.message); process.exitCode = 1; watcher.close(); });
server.listen(port, '127.0.0.1', () => console.log(`Live preview: http://127.0.0.1:${port}`));
process.on('SIGINT', () => { watcher.close(); server.close(); });
