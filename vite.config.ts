import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { execFile } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import path from 'node:path';

function sanitizeFileName(fileName: string) {
  return fileName.replace(/[\\/:*?"<>|]/g, '_').replace(/\s+/g, ' ').trim() || 'ansim-photo';
}

function sanitizeFolderName(folderName: string) {
  return sanitizeFileName(folderName).replace(/\.+$/g, '').trim() || '안심사진관_결과';
}

function decodeHeaderValue(value: string | string[] | undefined) {
  const raw = Array.isArray(value) ? value[0] : value;
  return raw ? decodeURIComponent(raw).trim() : '';
}

function pickFolderWithWindowsDialog() {
  return new Promise<string | null>((resolve, reject) => {
    if (process.platform !== 'win32') {
      reject(new Error('현재는 Windows 로컬 실행에서만 폴더 선택 저장을 지원합니다.'));
      return;
    }

    const script = `
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = 'PNG를 저장할 폴더를 선택하세요'
$dialog.ShowNewFolderButton = $true
$result = $dialog.ShowDialog()
if ($result -eq [System.Windows.Forms.DialogResult]::OK) {
  Write-Output $dialog.SelectedPath
}
`;
    const encodedCommand = Buffer.from(script, 'utf16le').toString('base64');
    execFile(
      'powershell.exe',
      ['-NoProfile', '-STA', '-ExecutionPolicy', 'Bypass', '-EncodedCommand', encodedCommand],
      { windowsHide: false },
      (error, stdout, stderr) => {
        if (error) {
          reject(new Error(stderr.trim() || error.message));
          return;
        }
        resolve(stdout.trim() || null);
      }
    );
  });
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
  const pickFolderHandler = async (
    request: import('node:http').IncomingMessage,
    response: import('node:http').ServerResponse,
    next: () => void
  ) => {
    if (request.method !== 'POST' || request.url?.split('?')[0] !== '/__ansim_pick_folder') {
      next();
      return;
    }

    try {
      const folderPath = await pickFolderWithWindowsDialog();
      response.statusCode = 200;
      response.setHeader('content-type', 'application/json; charset=utf-8');
      response.end(
        JSON.stringify({
          ok: true,
          cancelled: !folderPath,
          path: folderPath,
          name: folderPath ? path.basename(folderPath) : ''
        })
      );
    } catch (error) {
      response.statusCode = 500;
      response.setHeader('content-type', 'application/json; charset=utf-8');
      response.end(JSON.stringify({ ok: false, message: error instanceof Error ? error.message : 'folder picker failed' }));
    }
  };

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
      const fileName = sanitizeFileName(decodeHeaderValue(request.headers['x-ansim-filename']) || 'ansim-photo.png');
      const folderName = decodeHeaderValue(request.headers['x-ansim-folder']);
      const customOutputDir = decodeHeaderValue(request.headers['x-ansim-output-dir']);
      const baseDir = customOutputDir || path.join(homedir(), 'Downloads', '안심사진관');

      if (customOutputDir && !path.isAbsolute(customOutputDir)) {
        response.statusCode = 400;
        response.setHeader('content-type', 'application/json; charset=utf-8');
        response.end(JSON.stringify({ ok: false, message: '저장 폴더 경로를 확인하지 못했습니다.' }));
        return;
      }

      const outputDir = folderName ? path.join(baseDir, sanitizeFolderName(folderName)) : baseDir;
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
      server.middlewares.use(pickFolderHandler);
      server.middlewares.use(saveHandler);
    },
    configurePreviewServer(server: import('vite').PreviewServer) {
      server.middlewares.use(pickFolderHandler);
      server.middlewares.use(saveHandler);
    }
  };
}

export default defineConfig({
  base: process.env.GITHUB_PAGES === 'true' ? '/ansim-photo-studio/' : './',
  plugins: [react(), ansimLocalSavePlugin()],
  server: {
    port: 5173,
    strictPort: false
  }
});
