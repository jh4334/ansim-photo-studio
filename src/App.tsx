import { useEffect, useMemo, useRef, useState } from 'react';
import { Download, Eye, EyeOff, ImageUp, Loader2, ScanFace, ShieldCheck, Trash2 } from 'lucide-react';
import { FaceDetector, FilesetResolver } from '@mediapipe/tasks-vision';
import type { Detection } from '@mediapipe/tasks-vision';

type MaskMode = 'blur' | 'pixelate' | 'black';
type ToolMode = 'view' | 'manual';
type SourceKind = 'auto' | 'manual';

type FaceBox = {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  score: number;
  source: SourceKind;
};

const WASM_PATH = '/models/wasm';
const SHORT_RANGE_MODEL = '/models/blaze_face_short_range.tflite';
const FULL_RANGE_MODEL = '/models/blaze_face_full_range.tflite';

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function expandBox(box: FaceBox, imageWidth: number, imageHeight: number, ratio = 0.18): FaceBox {
  const padX = box.width * ratio;
  const padY = box.height * ratio;
  const x = clamp(box.x - padX, 0, imageWidth);
  const y = clamp(box.y - padY, 0, imageHeight);
  const right = clamp(box.x + box.width + padX, 0, imageWidth);
  const bottom = clamp(box.y + box.height + padY, 0, imageHeight);

  return {
    ...box,
    x,
    y,
    width: Math.max(1, right - x),
    height: Math.max(1, bottom - y)
  };
}

function overlap(first: FaceBox, second: FaceBox) {
  const x1 = Math.max(first.x, second.x);
  const y1 = Math.max(first.y, second.y);
  const x2 = Math.min(first.x + first.width, second.x + second.width);
  const y2 = Math.min(first.y + first.height, second.y + second.height);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const firstArea = first.width * first.height;
  const secondArea = second.width * second.height;
  return intersection / Math.max(1, firstArea + secondArea - intersection);
}

function mergeBoxes(boxes: FaceBox[]) {
  const ordered = [...boxes].sort((a, b) => b.score - a.score || b.width * b.height - a.width * a.height);
  const merged: FaceBox[] = [];

  for (const box of ordered) {
    if (merged.every((existing) => overlap(box, existing) < 0.34)) {
      merged.push(box);
    }
  }

  return merged.sort((a, b) => a.y - b.y || a.x - b.x);
}

function boxFromDetection(detection: Detection, index: number, modelName: string, imageWidth: number, imageHeight: number): FaceBox | null {
  const box = detection.boundingBox;
  if (!box) return null;

  const score = detection.categories[0]?.score ?? 1;
  const raw: FaceBox = {
    id: `${modelName}-${index}-${Math.round(box.originX)}-${Math.round(box.originY)}`,
    x: clamp(box.originX, 0, imageWidth),
    y: clamp(box.originY, 0, imageHeight),
    width: clamp(box.width, 1, imageWidth),
    height: clamp(box.height, 1, imageHeight),
    score,
    source: 'auto'
  };

  return expandBox(raw, imageWidth, imageHeight);
}

function drawMask(ctx: CanvasRenderingContext2D, image: HTMLImageElement, box: FaceBox, mode: MaskMode, strength: number, scale: number) {
  const sx = Math.round(box.x);
  const sy = Math.round(box.y);
  const sw = Math.round(box.width);
  const sh = Math.round(box.height);
  const dx = Math.round(box.x * scale);
  const dy = Math.round(box.y * scale);
  const dw = Math.round(box.width * scale);
  const dh = Math.round(box.height * scale);

  ctx.save();
  ctx.beginPath();
  ctx.rect(dx, dy, dw, dh);
  ctx.clip();

  if (mode === 'black') {
    ctx.fillStyle = '#050505';
    ctx.fillRect(dx, dy, dw, dh);
  } else if (mode === 'pixelate') {
    const pixelSize = Math.max(4, Math.round(strength));
    const smallWidth = Math.max(1, Math.round(dw / pixelSize));
    const smallHeight = Math.max(1, Math.round(dh / pixelSize));
    const buffer = document.createElement('canvas');
    buffer.width = smallWidth;
    buffer.height = smallHeight;
    const bufferCtx = buffer.getContext('2d');
    if (bufferCtx) {
      bufferCtx.imageSmoothingEnabled = true;
      bufferCtx.drawImage(image, sx, sy, sw, sh, 0, 0, smallWidth, smallHeight);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(buffer, 0, 0, smallWidth, smallHeight, dx, dy, dw, dh);
      ctx.imageSmoothingEnabled = true;
    }
  } else {
    ctx.filter = `blur(${Math.max(4, strength)}px)`;
    ctx.drawImage(image, sx, sy, sw, sh, dx, dy, dw, dh);
    ctx.filter = 'none';
  }

  ctx.restore();
}

function drawEditorCanvas(
  canvas: HTMLCanvasElement,
  image: HTMLImageElement,
  boxes: FaceBox[],
  mode: MaskMode,
  strength: number,
  showBoxes: boolean,
  draftBox: FaceBox | null
) {
  const scale = canvas.width / image.naturalWidth;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

  for (const box of boxes) {
    drawMask(ctx, image, box, mode, strength, scale);
  }

  if (showBoxes) {
    const visibleBoxes = draftBox ? [...boxes, draftBox] : boxes;
    for (const box of visibleBoxes) {
      const x = box.x * scale;
      const y = box.y * scale;
      const width = box.width * scale;
      const height = box.height * scale;
      ctx.save();
      ctx.strokeStyle = box.source === 'manual' ? '#f97316' : '#2563eb';
      ctx.lineWidth = 3;
      ctx.setLineDash(box === draftBox ? [8, 6] : []);
      ctx.strokeRect(x, y, width, height);
      ctx.fillStyle = box.source === 'manual' ? '#f97316' : '#2563eb';
      ctx.font = 'bold 13px system-ui';
      ctx.fillRect(x, Math.max(0, y - 24), box.source === 'manual' ? 76 : 62, 22);
      ctx.fillStyle = '#ffffff';
      ctx.fillText(box.source === 'manual' ? '수동 영역' : '자동 감지', x + 7, Math.max(15, y - 8));
      ctx.restore();
    }
  }
}

function App() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const objectUrlRef = useRef<string | null>(null);
  const [fileName, setFileName] = useState('');
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [boxes, setBoxes] = useState<FaceBox[]>([]);
  const [maskMode, setMaskMode] = useState<MaskMode>('blur');
  const [strength, setStrength] = useState(18);
  const [confidence, setConfidence] = useState(0.35);
  const [toolMode, setToolMode] = useState<ToolMode>('view');
  const [showBoxes, setShowBoxes] = useState(true);
  const [status, setStatus] = useState('사진을 선택하면 브라우저 안에서 바로 처리합니다.');
  const [isDetecting, setIsDetecting] = useState(false);
  const [draftBox, setDraftBox] = useState<FaceBox | null>(null);
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null);

  const autoCount = useMemo(() => boxes.filter((box) => box.source === 'auto').length, [boxes]);
  const manualCount = boxes.length - autoCount;

  useEffect(() => {
    return () => {
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    if (!canvas || !image) return;
    drawEditorCanvas(canvas, image, boxes, maskMode, strength, showBoxes, draftBox);
  }, [boxes, maskMode, strength, showBoxes, draftBox]);

  function loadFile(file: File) {
    if (!file.type.match(/^image\/(jpeg|png)$/)) {
      setStatus('JPG 또는 PNG 파일만 사용할 수 있습니다.');
      return;
    }

    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    const nextUrl = URL.createObjectURL(file);
    objectUrlRef.current = nextUrl;

    const image = new Image();
    image.onload = () => {
      imageRef.current = image;
      setFileName(file.name);
      setImageSize({ width: image.naturalWidth, height: image.naturalHeight });
      setBoxes([]);
      setDraftBox(null);
      setToolMode('view');

      const canvas = canvasRef.current;
      if (canvas) {
        const displayScale = Math.min(1, 1120 / image.naturalWidth);
        canvas.width = Math.max(1, Math.round(image.naturalWidth * displayScale));
        canvas.height = Math.max(1, Math.round(image.naturalHeight * displayScale));
        drawEditorCanvas(canvas, image, [], maskMode, strength, showBoxes, null);
      }

      setStatus('사진을 불러왔습니다. 얼굴 자동 감지를 실행하거나 수동 영역을 추가하세요.');
    };
    image.onerror = () => setStatus('이미지를 불러오지 못했습니다. 다른 파일로 시도해 주세요.');
    image.src = nextUrl;
  }

  async function runFaceDetection() {
    const image = imageRef.current;
    if (!image) {
      setStatus('먼저 사진을 선택해 주세요.');
      return;
    }

    setIsDetecting(true);
    setStatus('MediaPipe 얼굴 위치 감지 모델을 실행하는 중입니다.');
    try {
      const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
      const modelRuns = [
        { name: 'short', path: SHORT_RANGE_MODEL },
        { name: 'full', path: FULL_RANGE_MODEL }
      ];
      const detected: FaceBox[] = [];

      for (const run of modelRuns) {
        const detector = await FaceDetector.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: run.path,
            delegate: 'CPU'
          },
          runningMode: 'IMAGE',
          minDetectionConfidence: confidence,
          minSuppressionThreshold: 0.25
        });
        const result = detector.detect(image);
        result.detections.forEach((detection, index) => {
          const box = boxFromDetection(detection, index, run.name, image.naturalWidth, image.naturalHeight);
          if (box) detected.push(box);
        });
        detector.close();
      }

      const manualBoxes = boxes.filter((box) => box.source === 'manual');
      const merged = mergeBoxes([...detected, ...manualBoxes]);
      setBoxes(merged);
      setStatus(
        merged.length > 0
          ? `자동 감지가 끝났습니다. 감지된 얼굴 ${merged.filter((box) => box.source === 'auto').length}개를 확인해 주세요.`
          : '자동 감지된 얼굴이 없습니다. 감도를 낮추거나 수동 영역 추가를 사용하세요.'
      );
    } catch (error) {
      setStatus(`얼굴 감지를 실행하지 못했습니다. 모델 파일 위치를 확인해 주세요. ${error instanceof Error ? error.message : ''}`);
    } finally {
      setIsDetecting(false);
    }
  }

  function pointerToImagePoint(event: React.PointerEvent<HTMLCanvasElement>) {
    const canvas = event.currentTarget;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const canvasX = (event.clientX - rect.left) * scaleX;
    const canvasY = (event.clientY - rect.top) * scaleY;
    const image = imageRef.current;
    if (!image) return null;
    const imageScale = image.naturalWidth / canvas.width;
    return {
      x: clamp(canvasX * imageScale, 0, image.naturalWidth),
      y: clamp(canvasY * imageScale, 0, image.naturalHeight)
    };
  }

  function handlePointerDown(event: React.PointerEvent<HTMLCanvasElement>) {
    if (toolMode !== 'manual' || !imageRef.current) return;
    const point = pointerToImagePoint(event);
    if (!point) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    setDragStart(point);
    setDraftBox({
      id: 'draft',
      x: point.x,
      y: point.y,
      width: 1,
      height: 1,
      score: 1,
      source: 'manual'
    });
  }

  function handlePointerMove(event: React.PointerEvent<HTMLCanvasElement>) {
    if (!dragStart || toolMode !== 'manual' || !imageRef.current) return;
    const point = pointerToImagePoint(event);
    if (!point) return;
    setDraftBox({
      id: 'draft',
      x: Math.min(dragStart.x, point.x),
      y: Math.min(dragStart.y, point.y),
      width: Math.abs(point.x - dragStart.x),
      height: Math.abs(point.y - dragStart.y),
      score: 1,
      source: 'manual'
    });
  }

  function handlePointerUp() {
    if (!draftBox || !imageRef.current) {
      setDragStart(null);
      return;
    }

    if (draftBox.width > 12 && draftBox.height > 12) {
      setBoxes((current) => [
        ...current,
        {
          ...draftBox,
          id: `manual-${Date.now()}`,
          source: 'manual'
        }
      ]);
      setStatus('수동 얼굴 영역을 추가했습니다. 필요하면 아래 목록에서 삭제할 수 있습니다.');
    }
    setDraftBox(null);
    setDragStart(null);
  }

  function drawDownloadCanvas() {
    const image = imageRef.current;
    if (!image) return null;

    const canvas = document.createElement('canvas');
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    for (const box of boxes) {
      drawMask(ctx, image, box, maskMode, strength, 1);
    }
    return canvas;
  }

  function downloadPng() {
    const canvas = drawDownloadCanvas();
    if (!canvas) {
      setStatus('다운로드할 처리 결과가 없습니다.');
      return;
    }

    const link = document.createElement('a');
    const safeName = fileName.replace(/\.[^.]+$/, '').replace(/[\\/:*?"<>|]/g, '_') || 'masked_photo';
    link.download = `안심사진관_${safeName}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
    setStatus('처리 결과 PNG를 다운로드했습니다. 원본은 수정하지 않았습니다.');
  }

  function removeBox(id: string) {
    setBoxes((current) => current.filter((box) => box.id !== id));
    setStatus('선택한 얼굴 영역을 삭제했습니다.');
  }

  function resetAll() {
    setBoxes([]);
    setStatus('모든 얼굴 영역을 지웠습니다. 원본 사진은 그대로 유지됩니다.');
  }

  return (
    <main className="min-h-screen bg-slate-100 text-slate-900">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-7xl flex-col gap-4 px-5 py-5 md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-3xl font-black tracking-tight">안심사진관</h1>
            <p className="mt-1 text-sm font-semibold text-slate-600">학교 사진 얼굴 자동 가림 도구</p>
          </div>
          <div className="rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm font-bold text-emerald-800">
            사진은 서버에 저장되지 않고, 사용자의 PC 브라우저에서만 처리됩니다.
          </div>
        </div>
      </header>

      <section className="mx-auto grid max-w-7xl gap-5 px-5 py-5 lg:grid-cols-[320px_minmax(0,1fr)]">
        <aside className="grid content-start gap-4">
          <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
            <h2 className="mb-3 flex items-center gap-2 text-base font-black">
              <ImageUp size={19} />
              사진 불러오기
            </h2>
            <label className="grid cursor-pointer place-items-center rounded-lg border-2 border-dashed border-slate-300 bg-slate-50 px-4 py-7 text-center hover:border-blue-400 hover:bg-blue-50">
              <input
                className="sr-only"
                type="file"
                accept="image/jpeg,image/png"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) loadFile(file);
                  event.currentTarget.value = '';
                }}
              />
              <ImageUp className="mb-3 text-blue-600" size={30} />
              <strong className="text-sm">JPG 또는 PNG 선택</strong>
              <span className="mt-1 text-xs text-slate-500">파일은 브라우저 메모리에서만 사용됩니다.</span>
            </label>
            {fileName && (
              <dl className="mt-4 grid gap-2 text-sm">
                <div className="flex justify-between gap-3">
                  <dt className="text-slate-500">파일명</dt>
                  <dd className="truncate font-bold">{fileName}</dd>
                </div>
                <div className="flex justify-between gap-3">
                  <dt className="text-slate-500">크기</dt>
                  <dd className="font-bold">
                    {imageSize.width} x {imageSize.height}
                  </dd>
                </div>
              </dl>
            )}
          </section>

          <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
            <h2 className="mb-3 flex items-center gap-2 text-base font-black">
              <ScanFace size={19} />
              얼굴 감지
            </h2>
            <label className="mb-3 grid gap-2 text-sm font-bold">
              감지 감도
              <input
                min="0.15"
                max="0.75"
                step="0.05"
                type="range"
                value={confidence}
                onChange={(event) => setConfidence(Number(event.target.value))}
              />
              <span className="text-xs text-slate-500">값이 낮을수록 더 민감하게 감지합니다. 현재 {confidence.toFixed(2)}</span>
            </label>
            <button
              className="flex min-h-11 w-full items-center justify-center gap-2 rounded-lg bg-slate-950 px-4 text-sm font-black text-white disabled:opacity-50"
              disabled={!imageRef.current || isDetecting}
              onClick={runFaceDetection}
              type="button"
            >
              {isDetecting ? <Loader2 className="animate-spin" size={18} /> : <ScanFace size={18} />}
              얼굴 자동 감지
            </button>
          </section>

          <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
            <h2 className="mb-3 text-base font-black">가림 설정</h2>
            <div className="grid grid-cols-3 gap-2">
              {[
                ['blur', '블러'],
                ['pixelate', '모자이크'],
                ['black', '검은 박스']
              ].map(([value, label]) => (
                <button
                  className={`min-h-10 rounded-lg border px-2 text-sm font-black ${
                    maskMode === value ? 'border-blue-600 bg-blue-600 text-white' : 'border-slate-200 bg-white text-slate-700'
                  }`}
                  key={value}
                  onClick={() => setMaskMode(value as MaskMode)}
                  type="button"
                >
                  {label}
                </button>
              ))}
            </div>
            <label className="mt-4 grid gap-2 text-sm font-bold">
              가림 강도
              <input min="4" max="42" step="1" type="range" value={strength} onChange={(event) => setStrength(Number(event.target.value))} />
              <span className="text-xs text-slate-500">현재 강도 {strength}</span>
            </label>
          </section>

          <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
            <h2 className="mb-3 text-base font-black">영역 편집</h2>
            <div className="grid gap-2">
              <button
                className={`min-h-10 rounded-lg border text-sm font-black ${
                  toolMode === 'manual' ? 'border-orange-500 bg-orange-500 text-white' : 'border-slate-200 bg-slate-50 text-slate-700'
                }`}
                disabled={!imageRef.current}
                onClick={() => setToolMode((current) => (current === 'manual' ? 'view' : 'manual'))}
                type="button"
              >
                {toolMode === 'manual' ? '수동 추가 중' : '수동으로 얼굴 영역 추가'}
              </button>
              <button
                className="flex min-h-10 items-center justify-center gap-2 rounded-lg border border-slate-200 bg-white text-sm font-black text-slate-700"
                disabled={!imageRef.current}
                onClick={() => setShowBoxes((current) => !current)}
                type="button"
              >
                {showBoxes ? <EyeOff size={17} /> : <Eye size={17} />}
                {showBoxes ? '영역 표시 숨기기' : '영역 표시 보기'}
              </button>
              <button
                className="min-h-10 rounded-lg border border-rose-200 bg-rose-50 text-sm font-black text-rose-700"
                disabled={boxes.length === 0}
                onClick={resetAll}
                type="button"
              >
                모든 영역 지우기
              </button>
            </div>
          </section>
        </aside>

        <section className="grid gap-4">
          <div className="grid gap-3 rounded-lg border border-slate-200 bg-white p-4 shadow-sm md:grid-cols-4">
            <div className="rounded-lg bg-slate-100 p-3">
              <span className="text-xs font-bold text-slate-500">전체 영역</span>
              <strong className="block text-2xl">{boxes.length}</strong>
            </div>
            <div className="rounded-lg bg-blue-50 p-3">
              <span className="text-xs font-bold text-blue-700">자동 감지</span>
              <strong className="block text-2xl text-blue-800">{autoCount}</strong>
            </div>
            <div className="rounded-lg bg-orange-50 p-3">
              <span className="text-xs font-bold text-orange-700">수동 추가</span>
              <strong className="block text-2xl text-orange-800">{manualCount}</strong>
            </div>
            <button
              className="flex min-h-16 items-center justify-center gap-2 rounded-lg bg-emerald-600 px-4 font-black text-white disabled:opacity-50"
              disabled={!imageRef.current}
              onClick={downloadPng}
              type="button"
            >
              <Download size={18} />
              PNG 다운로드
            </button>
          </div>

          <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
            <div className="mb-3 flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
              <div>
                <h2 className="text-lg font-black">작업 화면</h2>
                <p className="text-sm text-slate-500">
                  {toolMode === 'manual' ? '사진 위에서 드래그하면 수동 얼굴 영역이 추가됩니다.' : '자동 감지 후 박스를 확인하고 필요 없는 영역은 삭제하세요.'}
                </p>
              </div>
              <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-black text-slate-600">{status}</span>
            </div>
            <div className="overflow-auto rounded-lg border border-slate-200 bg-slate-950 p-2">
              <canvas
                className={`mx-auto block max-w-full rounded bg-white ${toolMode === 'manual' ? 'cursor-crosshair' : 'cursor-default'}`}
                ref={canvasRef}
                onPointerDown={handlePointerDown}
                onPointerMove={handlePointerMove}
                onPointerUp={handlePointerUp}
                onPointerCancel={handlePointerUp}
              />
              {!imageRef.current && (
                <div className="grid min-h-96 place-items-center text-center text-slate-300">
                  <div>
                    <ShieldCheck className="mx-auto mb-3" size={46} />
                    <p className="font-black">사진을 선택하면 이곳에 작업 화면이 표시됩니다.</p>
                    <p className="mt-1 text-sm text-slate-400">로그인, DB, 서버 저장 없이 localhost 브라우저에서만 처리합니다.</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
            <h2 className="mb-3 text-lg font-black">감지/수동 영역 목록</h2>
            {boxes.length === 0 ? (
              <p className="rounded-lg bg-slate-50 p-4 text-sm font-semibold text-slate-500">등록된 얼굴 영역이 없습니다. 자동 감지를 실행하거나 수동으로 영역을 추가하세요.</p>
            ) : (
              <div className="grid gap-2">
                {boxes.map((box, index) => (
                  <article className="grid gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3 md:grid-cols-[64px_1fr_110px]" key={box.id}>
                    <strong className="text-sm">#{index + 1}</strong>
                    <div className="text-sm text-slate-600">
                      <b className={box.source === 'manual' ? 'text-orange-700' : 'text-blue-700'}>{box.source === 'manual' ? '수동 추가' : '자동 감지'}</b>
                      <span className="ml-2">
                        x {Math.round(box.x)}, y {Math.round(box.y)}, {Math.round(box.width)} x {Math.round(box.height)}
                      </span>
                    </div>
                    <button
                      className="flex min-h-9 items-center justify-center gap-2 rounded-lg border border-rose-200 bg-white text-sm font-black text-rose-700"
                      onClick={() => removeBox(box.id)}
                      type="button"
                    >
                      <Trash2 size={15} />
                      삭제
                    </button>
                  </article>
                ))}
              </div>
            )}
          </div>

          <div className="rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm font-semibold leading-6 text-amber-900">
            자동 얼굴 감지는 일부 얼굴을 놓칠 수 있습니다. 게시 전에는 얼굴, 이름표, 작품 이름, 게시판 개인정보, 연락처, QR 코드 노출 여부를 사용자가 반드시 최종 확인해야 합니다.
          </div>
        </section>
      </section>
    </main>
  );
}

export default App;
