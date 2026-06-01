import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { mkdir, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import path from 'node:path';

function sanitizeFileName(fileName: string) {
  return fileName.replace(/[\\/:*?"<>|]/g, '_').replace(/\s+/g, ' ').trim() || 'ansim-photo';
}

function sanitizeFolderName(folderName: string) {
  return sanitizeFileName(folderName).replace(/\.+$/g, '').trim() || '안심사진관_결과';
}

function readBody(request: import('node:http').IncomingMessage) {
  return new Promise<Buffer>((resolve, reject) => {
    const chunks: Buffer[] = [];
    request.on('data', (chunk) => chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)));
    request.on('end', () => resolve(Buffer.concat(chunks)));
    request.on('error', reject);
  });
}

function ansimLocalSavePlugin() {
  const saveHandler = async (
    request: import('node:http').IncomingMessage,
    response: import('node:http').ServerResponse,
    next: () => void
  ) => {
    if (request.method !== 'POST' || request.url?.split('?')[0] !== '/__ansim_save') {
      next();
      return;
    }

    try {
      const encodedName = Array.isArray(request.headers['x-ansim-filename'])
        ? request.headers['x-ansim-filename'][0]
        : request.headers['x-ansim-filename'];
      const encodedFolder = Array.isArray(request.headers['x-ansim-folder'])
        ? request.headers['x-ansim-folder'][0]
        : request.headers['x-ansim-folder'];
      const fileName = sanitizeFileName(decodeURIComponent(encodedName || 'ansim-photo.png'));
      const baseDir = path.join(homedir(), 'Downloads', '안심사진관');
      const outputDir = encodedFolder ? path.join(baseDir, sanitizeFolderName(decodeURIComponent(encodedFolder))) : baseDir;
      const outputPath = path.join(outputDir, fileName);
      const body = await readBody(request);

      await mkdir(outputDir, { recursive: true });
      await writeFile(outputPath, body);

      response.statusCode = 200;
      response.setHeader('content-type', 'application/json; charset=utf-8');
      response.end(JSON.stringify({ ok: true, path: outputPath }));
    } catch (error) {
      response.statusCode = 500;
      response.setHeader('content-type', 'application/json; charset=utf-8');
      response.end(JSON.stringify({ ok: false, message: error instanceof Error ? error.message : 'save failed' }));
    }
  };

  return {
    name: 'ansim-local-save',
    configureServer(server: import('vite').ViteDevServer) {
      server.middlewares.use(saveHandler);
    },
    configurePreviewServer(server: import('vite').PreviewServer) {
      server.middlewares.use(saveHandler);
    }
  };
}

export default defineConfig({
  base: './',
  plugins: [react(), ansimLocalSavePlugin()],
  server: {
    port: 5173,
    strictPort: false
  }
});
