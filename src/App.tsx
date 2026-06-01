import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Download,
  Eye,
  EyeOff,
  ImageUp,
  Images,
  Loader2,
  Redo2,
  ScanFace,
  ShieldCheck,
  Trash2,
  Undo2
} from 'lucide-react';
import { FaceDetector, FilesetResolver } from '@mediapipe/tasks-vision';
import type { Detection } from '@mediapipe/tasks-vision';

type MaskMode = 'blur' | 'pixelate' | 'black';
type MaskShape = 'rect' | 'ellipse';
type ToolMode = 'view' | 'manual' | 'brush';
type SourceKind = 'auto' | 'manual';
type DetectionPreset = 'sensitive' | 'balanced' | 'strict';
type ResizeHandle = 'nw' | 'ne' | 'sw' | 'se';

type Point = {
  x: number;
  y: number;
};

type FaceBox = {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  score: number;
  source: SourceKind;
  maskMode?: MaskMode;
  shape?: MaskShape;
};

type PhotoItem = {
  id: string;
  name: string;
  url: string;
  size: number;
  lastModified: number;
  width: number;
  height: number;
};

type ReadyDownload = {
  url: string;
  fileName: string;
  label: string;
};

type SaveFileHandle = {
  createWritable: () => Promise<{
    write: (data: Blob) => Promise<void>;
    close: () => Promise<void>;
  }>;
};

type SavePickerResult = SaveFileHandle | 'cancelled' | null;

type SaveOutcome = {
  mode: 'picker' | 'local' | 'browser';
  path?: string;
};

type EditInteraction =
  | { kind: 'draw'; start: Point }
  | { kind: 'brush'; start: Point }
  | { kind: 'move'; id: string; start: Point; originalBox: FaceBox; originalBoxes: FaceBox[] }
  | { kind: 'resize'; id: string; handle: ResizeHandle; start: Point; originalBox: FaceBox; originalBoxes: FaceBox[] };

const WASM_PATH = '/models/wasm';
const SHORT_RANGE_MODEL = '/models/blaze_face_short_range.tflite';
const FULL_RANGE_MODEL = '/models/blaze_face_full_range.tflite';
const MIN_BOX_SIZE = 16;
const LARGE_TOTAL_BYTES = 180 * 1024 * 1024;
const LARGE_SINGLE_BYTES = 35 * 1024 * 1024;

const DETECTION_PRESETS: Record<DetectionPreset, { label: string; confidence: number; expansion: number; note: string }> = {
  sensitive: { label: '민감', confidence: 0.2, expansion: 0.28, note: '작은 얼굴까지 넓게 찾기' },
  balanced: { label: '균형', confidence: 0.35, expansion: 0.2, note: '기본 권장값' },
  strict: { label: '엄격', confidence: 0.55, expansion: 0.14, note: '오탐 줄이기' }
};

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function safeBaseName(name: string) {
  return name.replace(/\.[^.]+$/, '').replace(/[\\/:*?"<>|]/g, '_') || 'masked_photo';
}

function formatBytes(bytes: number) {
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} B`;
}

function getFileTimestamp() {
  const now = new Date();
  const pad = (value: number) => String(value).padStart(2, '0');
  return `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}`;
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

function boxFromDetection(
  detection: Detection,
  index: number,
  modelName: string,
  imageWidth: number,
  imageHeight: number,
  expansion: number
): FaceBox | null {
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

  return expandBox(raw, imageWidth, imageHeight, expansion);
}

function drawMask(ctx: CanvasRenderingContext2D, image: HTMLImageElement, box: FaceBox, mode: MaskMode, strength: number, scale: number) {
  const imageWidth = image.naturalWidth;
  const imageHeight = image.naturalHeight;
  const sx = Math.max(0, Math.floor(box.x));
  const sy = Math.max(0, Math.floor(box.y));
  const sw = Math.max(1, Math.min(imageWidth - sx, Math.ceil(box.width)));
  const sh = Math.max(1, Math.min(imageHeight - sy, Math.ceil(box.height)));
  const dx = Math.round(box.x * scale);
  const dy = Math.round(box.y * scale);
  const dw = Math.round(box.width * scale);
  const dh = Math.round(box.height * scale);

  if (dw <= 0 || dh <= 0) return;

  ctx.save();
  ctx.beginPath();
  if (box.shape === 'ellipse') {
    ctx.ellipse(dx + dw / 2, dy + dh / 2, dw / 2, dh / 2, 0, 0, Math.PI * 2);
  } else {
    ctx.rect(dx, dy, dw, dh);
  }
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
    const blurRadius = Math.max(4, Math.round(strength * 1.4));
    const padding = Math.max(blurRadius * 3, 24);
    const cropX = Math.max(0, sx - padding);
    const cropY = Math.max(0, sy - padding);
    const cropRight = Math.min(imageWidth, sx + sw + padding);
    const cropBottom = Math.min(imageHeight, sy + sh + padding);
    const cropWidth = Math.max(1, cropRight - cropX);
    const cropHeight = Math.max(1, cropBottom - cropY);
    const buffer = document.createElement('canvas');
    buffer.width = cropWidth;
    buffer.height = cropHeight;
    const bufferCtx = buffer.getContext('2d');

    if (bufferCtx) {
      bufferCtx.filter = `blur(${blurRadius}px)`;
      bufferCtx.drawImage(image, cropX, cropY, cropWidth, cropHeight, 0, 0, cropWidth, cropHeight);
      ctx.drawImage(
        buffer,
        0,
        0,
        cropWidth,
        cropHeight,
        Math.round(cropX * scale),
        Math.round(cropY * scale),
        Math.round(cropWidth * scale),
        Math.round(cropHeight * scale)
      );
    }
  }

  ctx.restore();
}

function drawBoxOverlay(ctx: CanvasRenderingContext2D, box: FaceBox, scale: number, isDraft: boolean, isSelected: boolean) {
  const x = box.x * scale;
  const y = box.y * scale;
  const width = box.width * scale;
  const height = box.height * scale;
  const color = isSelected ? '#eab308' : box.source === 'manual' ? '#f97316' : '#2563eb';

  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = isSelected ? 4 : 3;
  ctx.setLineDash(isDraft ? [8, 6] : []);
  ctx.strokeRect(x, y, width, height);
  ctx.fillStyle = color;
  ctx.font = 'bold 13px system-ui';
  ctx.fillRect(x, Math.max(0, y - 24), box.source === 'manual' ? 76 : 62, 22);
  ctx.fillStyle = '#ffffff';
  ctx.fillText(box.source === 'manual' ? '수동 영역' : '자동 감지', x + 7, Math.max(15, y - 8));

  if (isSelected && !isDraft) {
    const handles = [
      [x, y],
      [x + width, y],
      [x, y + height],
      [x + width, y + height]
    ];
    ctx.fillStyle = '#ffffff';
    ctx.strokeStyle = '#a16207';
    ctx.lineWidth = 2;
    for (const [hx, hy] of handles) {
      ctx.fillRect(hx - 5, hy - 5, 10, 10);
      ctx.strokeRect(hx - 5, hy - 5, 10, 10);
    }
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
  draftBox: FaceBox | null,
  selectedBoxId: string,
  pendingAutoBoxes: FaceBox[],
  draftBrushBoxes: FaceBox[]
) {
  const scale = canvas.width / image.naturalWidth;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

  for (const box of boxes) {
    drawMask(ctx, image, box, box.maskMode ?? mode, strength, scale);
  }
  for (const box of pendingAutoBoxes) {
    drawMask(ctx, image, box, mode, strength, scale);
  }
  for (const box of draftBrushBoxes) {
    drawMask(ctx, image, box, box.maskMode ?? mode, strength, scale);
  }

  if (showBoxes) {
    for (const box of boxes) {
      drawBoxOverlay(ctx, box, scale, false, box.id === selectedBoxId);
    }
    for (const box of pendingAutoBoxes) {
      drawBoxOverlay(ctx, box, scale, true, false);
    }
    for (const box of draftBrushBoxes) {
      drawBoxOverlay(ctx, box, scale, true, false);
    }
    if (draftBox) drawBoxOverlay(ctx, draftBox, scale, true, false);
  }
}

function drawMaskedCanvas(image: HTMLImageElement, boxes: FaceBox[], mode: MaskMode, strength: number) {
  const canvas = document.createElement('canvas');
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;

  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  for (const box of boxes) {
    drawMask(ctx, image, box, box.maskMode ?? mode, strength, 1);
  }
  return canvas;
}

function canvasToPngBytes(canvas: HTMLCanvasElement) {
  return new Promise<Uint8Array>((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!blob) {
        reject(new Error('PNG 변환에 실패했습니다.'));
        return;
      }
      blob.arrayBuffer().then((buffer) => resolve(new Uint8Array(buffer)), reject);
    }, 'image/png');
  });
}

function loadImage(url: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('이미지를 불러오지 못했습니다.'));
    image.src = url;
  });
}

function bytesToArrayBuffer(bytes: Uint8Array) {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return copy.buffer;
}

function App() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const objectUrlsRef = useRef<string[]>([]);
  const loadRequestRef = useRef(0);
  const folderSaveCancelRef = useRef(false);
  const [photos, setPhotos] = useState<PhotoItem[]>([]);
  const [currentPhotoId, setCurrentPhotoId] = useState('');
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [boxes, setBoxes] = useState<FaceBox[]>([]);
  const [boxesByPhoto, setBoxesByPhoto] = useState<Record<string, FaceBox[]>>({});
  const [maskMode, setMaskMode] = useState<MaskMode>('blur');
  const [strength, setStrength] = useState(18);
  const [confidence, setConfidence] = useState(0.35);
  const [boxExpansion, setBoxExpansion] = useState(DETECTION_PRESETS.balanced.expansion);
  const [manualShape, setManualShape] = useState<MaskShape>('rect');
  const [brushSize, setBrushSize] = useState(42);
  const [detectionPreset, setDetectionPreset] = useState<DetectionPreset>('balanced');
  const [toolMode, setToolMode] = useState<ToolMode>('view');
  const [showBoxes, setShowBoxes] = useState(true);
  const [status, setStatus] = useState('사진을 선택하면 브라우저 안에서 바로 처리합니다.');
  const [memoryWarning, setMemoryWarning] = useState('');
  const [operationProgress, setOperationProgress] = useState('');
  const [readyDownload, setReadyDownload] = useState<ReadyDownload | null>(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [isBatchDetecting, setIsBatchDetecting] = useState(false);
  const [isSavingFolder, setIsSavingFolder] = useState(false);
  const [draftBox, setDraftBox] = useState<FaceBox | null>(null);
  const [draftBrushBoxes, setDraftBrushBoxes] = useState<FaceBox[]>([]);
  const [pendingAutoBoxes, setPendingAutoBoxes] = useState<FaceBox[]>([]);
  const [interaction, setInteraction] = useState<EditInteraction | null>(null);
  const [selectedBoxId, setSelectedBoxId] = useState('');
  const [historyPast, setHistoryPast] = useState<FaceBox[][]>([]);
  const [historyFuture, setHistoryFuture] = useState<FaceBox[][]>([]);

  const currentPhoto = useMemo(() => photos.find((photo) => photo.id === currentPhotoId) ?? null, [photos, currentPhotoId]);
  const autoCount = useMemo(() => boxes.filter((box) => box.source === 'auto').length, [boxes]);
  const manualCount = boxes.length - autoCount;
  const selectedBox = useMemo(() => boxes.find((box) => box.id === selectedBoxId) ?? null, [boxes, selectedBoxId]);
  const batchSummary = useMemo(() => {
    const snapshot = currentPhotoId ? { ...boxesByPhoto, [currentPhotoId]: boxes } : boxesByPhoto;
    return {
      processed: photos.filter((photo) => (snapshot[photo.id] ?? []).length > 0).length,
      totalBoxes: photos.reduce((sum, photo) => sum + (snapshot[photo.id] ?? []).length, 0),
      autoBoxes: photos.reduce((sum, photo) => sum + (snapshot[photo.id] ?? []).filter((box) => box.source === 'auto').length, 0),
      manualBoxes: photos.reduce((sum, photo) => sum + (snapshot[photo.id] ?? []).filter((box) => box.source === 'manual').length, 0)
    };
  }, [boxes, boxesByPhoto, currentPhotoId, photos]);

  useEffect(() => {
    return () => {
      for (const url of objectUrlsRef.current) URL.revokeObjectURL(url);
    };
  }, []);

  useEffect(() => {
    return () => {
      if (readyDownload) URL.revokeObjectURL(readyDownload.url);
    };
  }, [readyDownload]);

  useEffect(() => {
    if (!currentPhotoId) return;
    setBoxesByPhoto((current) => (current[currentPhotoId] === boxes ? current : { ...current, [currentPhotoId]: boxes }));
  }, [boxes, currentPhotoId]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    if (!canvas || !image) return;
    drawEditorCanvas(canvas, image, boxes, maskMode, strength, showBoxes, draftBox, selectedBoxId, pendingAutoBoxes, draftBrushBoxes);
  }, [boxes, maskMode, strength, showBoxes, draftBox, selectedBoxId, pendingAutoBoxes, draftBrushBoxes]);

  function resetHistory() {
    setHistoryPast([]);
    setHistoryFuture([]);
  }

  function applyBoxes(nextBoxes: FaceBox[], message: string, nextSelectedId = selectedBoxId) {
    setHistoryPast((current) => [...current.slice(-19), boxes]);
    setHistoryFuture([]);
    setBoxes(nextBoxes);
    setPendingAutoBoxes([]);
    setSelectedBoxId(nextSelectedId);
    setStatus(message);
  }

  function loadPhoto(photo: PhotoItem, savedBoxes = boxesByPhoto) {
    const requestId = ++loadRequestRef.current;
    const image = new Image();
    image.onload = () => {
      if (requestId !== loadRequestRef.current) return;
      imageRef.current = image;
      setCurrentPhotoId(photo.id);
      setImageSize({ width: image.naturalWidth, height: image.naturalHeight });
      setPhotos((current) =>
        current.map((item) => (item.id === photo.id ? { ...item, width: image.naturalWidth, height: image.naturalHeight } : item))
      );
      setBoxes(savedBoxes[photo.id] ?? []);
      setDraftBox(null);
      setDraftBrushBoxes([]);
      setPendingAutoBoxes([]);
      setInteraction(null);
      setSelectedBoxId('');
      setToolMode('view');
      resetHistory();

      const canvas = canvasRef.current;
      if (canvas) {
        const displayScale = Math.min(1, 1120 / image.naturalWidth);
        canvas.width = Math.max(1, Math.round(image.naturalWidth * displayScale));
        canvas.height = Math.max(1, Math.round(image.naturalHeight * displayScale));
        drawEditorCanvas(canvas, image, savedBoxes[photo.id] ?? [], maskMode, strength, showBoxes, null, '', [], []);
      }

      setStatus('사진을 불러왔습니다. 자동 감지 또는 수동 영역 편집을 진행하세요.');
    };
    image.onerror = () => {
      if (requestId === loadRequestRef.current) setStatus('이미지를 불러오지 못했습니다. 다른 파일로 시도해 주세요.');
    };
    image.src = photo.url;
  }

  function loadFiles(files: FileList | File[]) {
    const fileList = Array.from(files);
    const validFiles = fileList.filter((file) => file.type.match(/^image\/(jpeg|png)$/));
    if (validFiles.length === 0) {
      setStatus('JPG 또는 PNG 파일만 사용할 수 있습니다.');
      return;
    }

    const totalBytes = validFiles.reduce((sum, file) => sum + file.size, 0);
    const largestFile = validFiles.reduce((largest, file) => Math.max(largest, file.size), 0);
    if (totalBytes > LARGE_TOTAL_BYTES || largestFile > LARGE_SINGLE_BYTES) {
      setMemoryWarning(
        `큰 사진 묶음입니다. 총 ${formatBytes(totalBytes)}이며 가장 큰 파일은 ${formatBytes(largestFile)}입니다. 일괄 PNG 저장 중 브라우저가 느려질 수 있습니다.`
      );
    } else {
      setMemoryWarning('');
    }

    for (const url of objectUrlsRef.current) URL.revokeObjectURL(url);
    objectUrlsRef.current = [];

    const nextPhotos = validFiles.map((file, index) => {
      const url = URL.createObjectURL(file);
      objectUrlsRef.current.push(url);
      return {
        id: `${file.name}-${file.size}-${file.lastModified}-${index}`,
        name: file.name,
        url,
        size: file.size,
        lastModified: file.lastModified,
        width: 0,
        height: 0
      };
    });

    setPhotos(nextPhotos);
    setBoxesByPhoto({});
    setPendingAutoBoxes([]);
    setDraftBrushBoxes([]);
    setOperationProgress('');
    resetHistory();
    loadPhoto(nextPhotos[0], {});

    setStatus(
      validFiles.length === fileList.length
        ? `${validFiles.length}장의 사진을 작업 목록에 추가했습니다.`
        : `${validFiles.length}장의 JPG/PNG만 추가했습니다. 지원하지 않는 파일은 제외했습니다.`
    );
  }

  function switchPhoto(photo: PhotoItem) {
    const nextBoxesByPhoto = currentPhotoId ? { ...boxesByPhoto, [currentPhotoId]: boxes } : boxesByPhoto;
    setBoxesByPhoto(nextBoxesByPhoto);
    loadPhoto(photo, nextBoxesByPhoto);
  }

  async function runFaceDetection() {
    const image = imageRef.current;
    if (!image) {
      setStatus('먼저 사진을 선택해 주세요.');
      return;
    }

    const preset = DETECTION_PRESETS[detectionPreset];
    setIsDetecting(true);
    setPendingAutoBoxes([]);
    setOperationProgress('감지 준비 중');
    setStatus(`${preset.label} 프리셋으로 MediaPipe 얼굴 위치 감지 모델을 실행하는 중입니다.`);
    try {
      const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
      const modelRuns = [
        { name: 'short', path: SHORT_RANGE_MODEL },
        { name: 'full', path: FULL_RANGE_MODEL }
      ];
      const detected: FaceBox[] = [];

      for (const [runIndex, run] of modelRuns.entries()) {
        setOperationProgress(`얼굴 감지 모델 ${runIndex + 1}/${modelRuns.length} 실행 중`);
        const detector = await FaceDetector.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: run.path,
            delegate: 'CPU'
          },
          runningMode: 'IMAGE',
          minDetectionConfidence: confidence,
          minSuppressionThreshold: detectionPreset === 'sensitive' ? 0.18 : 0.25
        });
        const result = detector.detect(image);
        result.detections.forEach((detection, index) => {
          const box = boxFromDetection(detection, index, run.name, image.naturalWidth, image.naturalHeight, boxExpansion);
          if (box) detected.push(box);
        });
        detector.close();
      }

      const detectedBoxes = mergeBoxes(detected).map((box) => ({ ...box, shape: 'rect' as const }));
      setPendingAutoBoxes(detectedBoxes);
      setOperationProgress('');
      setStatus(
        detectedBoxes.length > 0
          ? `자동 감지 후보 ${detectedBoxes.length}개를 찾았습니다. 박스를 확인한 뒤 감지 결과 적용을 눌러 주세요.`
          : '자동 감지된 얼굴이 없습니다. 민감 프리셋을 선택하거나 수동 영역 추가를 사용하세요.'
      );
    } catch (error) {
      setStatus(`얼굴 감지를 실행하지 못했습니다. 모델 파일 위치를 확인해 주세요. ${error instanceof Error ? error.message : ''}`);
    } finally {
      setOperationProgress('');
      setIsDetecting(false);
    }
  }

  async function detectBoxesForImage(image: HTMLImageElement, preset: (typeof DETECTION_PRESETS)[DetectionPreset], vision: Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>) {
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
        minSuppressionThreshold: detectionPreset === 'sensitive' ? 0.18 : 0.25
      });
      const result = detector.detect(image);
      result.detections.forEach((detection, index) => {
        const box = boxFromDetection(detection, index, run.name, image.naturalWidth, image.naturalHeight, boxExpansion);
        if (box) detected.push(box);
      });
      detector.close();
    }

    return mergeBoxes(detected).map((box) => ({ ...box, shape: 'rect' as const }));
  }

  async function runBatchFaceDetection() {
    if (photos.length === 0) {
      setStatus('먼저 사진을 선택해 주세요.');
      return;
    }

    const preset = DETECTION_PRESETS[detectionPreset];
    setIsBatchDetecting(true);
    setPendingAutoBoxes([]);
    setOperationProgress(`전체 자동 감지 준비 중 0/${photos.length}`);
    setStatus('작업 목록 전체 사진에 자동 감지를 실행합니다.');

    try {
      const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
      const snapshotBoxesByPhoto = currentPhotoId ? { ...boxesByPhoto, [currentPhotoId]: boxes } : boxesByPhoto;
      const nextBoxesByPhoto: Record<string, FaceBox[]> = { ...snapshotBoxesByPhoto };
      let totalDetected = 0;

      for (const [index, photo] of photos.entries()) {
        setOperationProgress(`전체 자동 감지 중 ${index + 1}/${photos.length}: ${photo.name}`);
        const image = photo.id === currentPhotoId && imageRef.current ? imageRef.current : await loadImage(photo.url);
        const detectedBoxes = await detectBoxesForImage(image, preset, vision);
        const manualBoxes = (snapshotBoxesByPhoto[photo.id] ?? []).filter((box) => box.source === 'manual');
        nextBoxesByPhoto[photo.id] = mergeBoxes([...detectedBoxes, ...manualBoxes]);
        totalDetected += detectedBoxes.length;
      }

      setBoxesByPhoto(nextBoxesByPhoto);
      if (currentPhotoId) setBoxes(nextBoxesByPhoto[currentPhotoId] ?? []);
      setStatus(`전체 자동 감지가 끝났습니다. ${photos.length}장 기준 자동 후보 ${totalDetected}개를 바로 적용했습니다. 현재 가림 설정으로 PNG 저장에 반영됩니다.`);
    } catch (error) {
      setStatus(`전체 자동 감지를 완료하지 못했습니다. ${error instanceof Error ? error.message : ''}`);
    } finally {
      setOperationProgress('');
      setIsBatchDetecting(false);
    }
  }

  function applyPendingAutoBoxes() {
    if (pendingAutoBoxes.length === 0) {
      setStatus('적용할 자동 감지 후보가 없습니다.');
      return;
    }
    const manualBoxes = boxes.filter((box) => box.source === 'manual');
    const merged = mergeBoxes([...pendingAutoBoxes, ...manualBoxes]);
    applyBoxes(merged, `자동 감지 후보 ${pendingAutoBoxes.length}개를 적용했습니다. 필요 없는 영역은 삭제하거나 조정하세요.`, '');
  }

  function cancelPendingAutoBoxes() {
    setPendingAutoBoxes([]);
    setStatus('자동 감지 후보 적용을 취소했습니다. 기존 영역은 유지됩니다.');
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

  function handleAtPoint(point: Point, box: FaceBox, canvas: HTMLCanvasElement): ResizeHandle | null {
    const image = imageRef.current;
    if (!image) return null;
    const threshold = Math.max(8, 12 * (image.naturalWidth / canvas.width));
    const corners: Array<[ResizeHandle, Point]> = [
      ['nw', { x: box.x, y: box.y }],
      ['ne', { x: box.x + box.width, y: box.y }],
      ['sw', { x: box.x, y: box.y + box.height }],
      ['se', { x: box.x + box.width, y: box.y + box.height }]
    ];

    for (const [handle, corner] of corners) {
      if (Math.abs(point.x - corner.x) <= threshold && Math.abs(point.y - corner.y) <= threshold) {
        return handle;
      }
    }
    return null;
  }

  function boxAtPoint(point: Point) {
    return [...boxes].reverse().find((box) => point.x >= box.x && point.x <= box.x + box.width && point.y >= box.y && point.y <= box.y + box.height) ?? null;
  }

  function resizeBox(box: FaceBox, handle: ResizeHandle, dx: number, dy: number, imageWidth: number, imageHeight: number): FaceBox {
    let left = box.x;
    let top = box.y;
    let right = box.x + box.width;
    let bottom = box.y + box.height;

    if (handle.includes('w')) left = clamp(left + dx, 0, right - MIN_BOX_SIZE);
    if (handle.includes('e')) right = clamp(right + dx, left + MIN_BOX_SIZE, imageWidth);
    if (handle.includes('n')) top = clamp(top + dy, 0, bottom - MIN_BOX_SIZE);
    if (handle.includes('s')) bottom = clamp(bottom + dy, top + MIN_BOX_SIZE, imageHeight);

    return {
      ...box,
      x: left,
      y: top,
      width: right - left,
      height: bottom - top
    };
  }

  function makeBrushBox(point: Point): FaceBox {
    const image = imageRef.current;
    const size = brushSize;
    const x = image ? clamp(point.x - size / 2, 0, Math.max(0, image.naturalWidth - size)) : point.x - size / 2;
    const y = image ? clamp(point.y - size / 2, 0, Math.max(0, image.naturalHeight - size)) : point.y - size / 2;
    return {
      id: `brush-${Date.now()}-${Math.round(point.x)}-${Math.round(point.y)}`,
      x,
      y,
      width: size,
      height: size,
      score: 1,
      source: 'manual',
      shape: 'ellipse'
    };
  }

  function handlePointerDown(event: React.PointerEvent<HTMLCanvasElement>) {
    if (!imageRef.current) return;
    const point = pointerToImagePoint(event);
    if (!point) return;
    event.currentTarget.setPointerCapture(event.pointerId);

    if (toolMode === 'manual') {
      setInteraction({ kind: 'draw', start: point });
      setDraftBox({
        id: 'draft',
        x: point.x,
        y: point.y,
        width: 1,
        height: 1,
        score: 1,
        source: 'manual',
        shape: manualShape
      });
      return;
    }

    if (toolMode === 'brush') {
      const firstBox = makeBrushBox(point);
      setInteraction({ kind: 'brush', start: point });
      setDraftBrushBoxes([firstBox]);
      setStatus('브러시로 가릴 영역을 칠하는 중입니다.');
      return;
    }

    for (const box of [...boxes].reverse()) {
      const handle = handleAtPoint(point, box, event.currentTarget);
      if (handle) {
        setSelectedBoxId(box.id);
        setInteraction({ kind: 'resize', id: box.id, handle, start: point, originalBox: box, originalBoxes: boxes });
        setStatus('선택한 영역의 모서리를 드래그해 크기를 조정합니다.');
        return;
      }
    }

    const hitBox = boxAtPoint(point);
    if (hitBox) {
      setSelectedBoxId(hitBox.id);
      setInteraction({ kind: 'move', id: hitBox.id, start: point, originalBox: hitBox, originalBoxes: boxes });
      setStatus('선택한 영역을 드래그해 위치를 조정합니다.');
    } else {
      setSelectedBoxId('');
    }
  }

  function handlePointerMove(event: React.PointerEvent<HTMLCanvasElement>) {
    if (!interaction || !imageRef.current) return;
    const point = pointerToImagePoint(event);
    if (!point) return;

    if (interaction.kind === 'draw') {
      setDraftBox({
        id: 'draft',
        x: Math.min(interaction.start.x, point.x),
        y: Math.min(interaction.start.y, point.y),
        width: Math.abs(point.x - interaction.start.x),
        height: Math.abs(point.y - interaction.start.y),
        score: 1,
        source: 'manual',
        shape: manualShape
      });
      return;
    }

    if (interaction.kind === 'brush') {
      setDraftBrushBoxes((current) => {
        const last = current[current.length - 1];
        if (last) {
          const centerX = last.x + last.width / 2;
          const centerY = last.y + last.height / 2;
          const distance = Math.hypot(point.x - centerX, point.y - centerY);
          if (distance < brushSize * 0.45) return current;
        }
        return [...current, makeBrushBox(point)];
      });
      return;
    }

    const dx = point.x - interaction.start.x;
    const dy = point.y - interaction.start.y;
    const nextBox =
      interaction.kind === 'move'
        ? {
            ...interaction.originalBox,
            x: clamp(interaction.originalBox.x + dx, 0, imageRef.current.naturalWidth - interaction.originalBox.width),
            y: clamp(interaction.originalBox.y + dy, 0, imageRef.current.naturalHeight - interaction.originalBox.height)
          }
        : resizeBox(interaction.originalBox, interaction.handle, dx, dy, imageRef.current.naturalWidth, imageRef.current.naturalHeight);

    setBoxes(interaction.originalBoxes.map((box) => (box.id === interaction.id ? nextBox : box)));
  }

  function handlePointerUp() {
    if (!interaction || !imageRef.current) {
      setInteraction(null);
      setDraftBox(null);
      setDraftBrushBoxes([]);
      return;
    }

    if (interaction.kind === 'draw') {
      if (draftBox && draftBox.width > 12 && draftBox.height > 12) {
        const nextBox = {
          ...draftBox,
          id: `manual-${Date.now()}`,
          source: 'manual' as const
        };
        applyBoxes([...boxes, nextBox], '수동 얼굴 영역을 추가했습니다. 노란 테두리를 드래그해 수정할 수 있습니다.', nextBox.id);
      }
    } else if (interaction.kind === 'brush') {
      if (draftBrushBoxes.length > 0) {
        applyBoxes([...boxes, ...draftBrushBoxes], `브러시 마스킹 영역 ${draftBrushBoxes.length}개를 추가했습니다.`, '');
      }
    } else {
      setHistoryPast((current) => [...current.slice(-19), interaction.originalBoxes]);
      setHistoryFuture([]);
      setStatus('영역 편집을 반영했습니다.');
    }

    setInteraction(null);
    setDraftBox(null);
    setDraftBrushBoxes([]);
  }

  function undoBoxes() {
    const previous = historyPast[historyPast.length - 1];
    if (!previous) return;
    setHistoryPast((current) => current.slice(0, -1));
    setHistoryFuture((current) => [boxes, ...current].slice(0, 20));
    setBoxes(previous);
    setSelectedBoxId('');
    setStatus('이전 영역 상태로 되돌렸습니다.');
  }

  function redoBoxes() {
    const next = historyFuture[0];
    if (!next) return;
    setHistoryFuture((current) => current.slice(1));
    setHistoryPast((current) => [...current.slice(-19), boxes]);
    setBoxes(next);
    setSelectedBoxId('');
    setStatus('되돌린 영역 상태를 다시 적용했습니다.');
  }

  function drawDownloadCanvas() {
    const image = imageRef.current;
    if (!image) return null;
    return drawMaskedCanvas(image, boxes, maskMode, strength);
  }

  function prepareDownload(blob: Blob, fileName: string, label: string) {
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.download = fileName;
    link.href = url;
    link.rel = 'noopener';
    link.style.display = 'none';
    document.body.appendChild(link);
    link.click();
    link.remove();
    setReadyDownload({ url, fileName, label });
  }

  async function pickSaveTarget(fileName: string, mimeType: string, extensions: string[]): Promise<SavePickerResult> {
    const picker = (window as Window & {
      showSaveFilePicker?: (options: {
        suggestedName: string;
        types: Array<{ description: string; accept: Record<string, string[]> }>;
      }) => Promise<SaveFileHandle>;
    }).showSaveFilePicker;

    if (!picker) return null;

    try {
      return await picker({
        suggestedName: fileName,
        types: [
          {
            description: 'PNG 이미지',
            accept: { [mimeType]: extensions }
          }
        ]
      });
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') return 'cancelled';
      throw error;
    }
  }

  async function saveViaLocalServer(blob: Blob, fileName: string, folderName?: string) {
    const headers: Record<string, string> = {
      'content-type': 'application/octet-stream',
      'x-ansim-filename': encodeURIComponent(fileName)
    };
    if (folderName) headers['x-ansim-folder'] = encodeURIComponent(folderName);

    const response = await fetch('/__ansim_save', {
      method: 'POST',
      headers,
      body: blob
    });
    const result = await response.json();
    if (!response.ok || !result.ok) {
      throw new Error(result.message || '로컬 저장 실패');
    }
    return String(result.path || '');
  }

  async function savePreparedBlob(blob: Blob, fileName: string, label: string, handle: SavePickerResult): Promise<SaveOutcome> {
    if (handle && handle !== 'cancelled') {
      try {
        const writable = await handle.createWritable();
        await writable.write(blob);
        await writable.close();

        const url = URL.createObjectURL(blob);
        setReadyDownload({ url, fileName, label });
        return { mode: 'picker' };
      } catch {
        // Some embedded browsers expose the picker but block actual writes.
        // Fall through to the localhost save route for local desktop use.
      }
    }

    try {
      const savedPath = await saveViaLocalServer(blob, fileName);
      const url = URL.createObjectURL(blob);
      setReadyDownload({ url, fileName, label });
      return { mode: 'local', path: savedPath };
    } catch {
      prepareDownload(blob, fileName, label);
      return { mode: 'browser' };
    }
  }

  async function downloadPng() {
    const canvas = drawDownloadCanvas();
    if (!canvas || !currentPhoto) {
      setStatus('다운로드할 처리 결과가 없습니다.');
      return;
    }

    const fileName = `안심사진관_${safeBaseName(currentPhoto.name)}.png`;
    const saveTarget = await pickSaveTarget(fileName, 'image/png', ['.png']);
    if (saveTarget === 'cancelled') {
      setStatus('PNG 저장을 취소했습니다.');
      return;
    }

    const bytes = await canvasToPngBytes(canvas);
    const result = await savePreparedBlob(new Blob([bytesToArrayBuffer(bytes)], { type: 'image/png' }), fileName, '현재 사진 PNG', saveTarget);
    setStatus(
      result.mode === 'picker'
        ? '현재 사진 PNG를 선택한 위치에 저장했습니다.'
        : result.mode === 'local'
          ? `현재 사진 PNG를 내 PC에 저장했습니다: ${result.path}`
          : '현재 사진 PNG 다운로드를 준비했습니다. 자동으로 내려받아지지 않으면 화면의 다시 다운로드 버튼을 눌러 주세요.'
    );
  }

  async function saveAllPngFolder() {
    if (photos.length === 0) {
      setStatus('PNG로 저장할 사진이 없습니다.');
      return;
    }

    const exportPhotos = photos;
    const folderName = `안심사진관_${getFileTimestamp()}`;

    folderSaveCancelRef.current = false;
    setIsSavingFolder(true);
    setReadyDownload(null);
    setOperationProgress(`PNG 폴더 저장 준비 중 0/${exportPhotos.length}`);
    setStatus(`작업 목록의 모든 사진을 PNG로 저장하는 중입니다. 저장 폴더: Downloads\\안심사진관\\${folderName}`);
    try {
      const snapshotBoxesByPhoto = currentPhotoId ? { ...boxesByPhoto, [currentPhotoId]: boxes } : boxesByPhoto;
      let savedCount = 0;
      let lastPath = '';

      for (const [index, photo] of exportPhotos.entries()) {
        if (folderSaveCancelRef.current) {
          setStatus('PNG 폴더 저장을 취소했습니다.');
          return;
        }
        setOperationProgress(`PNG 저장 중 ${index + 1}/${exportPhotos.length}: ${photo.name}`);
        const image = await loadImage(photo.url);
        const canvas = drawMaskedCanvas(image, snapshotBoxesByPhoto[photo.id] ?? [], maskMode, strength);
        if (!canvas) continue;
        const bytes = await canvasToPngBytes(canvas);
        lastPath = await saveViaLocalServer(
          new Blob([bytesToArrayBuffer(bytes)], { type: 'image/png' }),
          `안심사진관_${String(index + 1).padStart(3, '0')}_${safeBaseName(photo.name)}.png`,
          folderName
        );
        savedCount += 1;
      }

      if (folderSaveCancelRef.current) {
        setStatus('PNG 폴더 저장을 취소했습니다.');
        return;
      }

      const folderPath = lastPath ? lastPath.replace(/\\[^\\]+$/, '') : `C:\\Users\\USER\\Downloads\\안심사진관\\${folderName}`;
      setStatus(`${savedCount}장의 처리 결과 PNG를 폴더에 저장했습니다: ${folderPath}`);
    } catch (error) {
      setStatus(`PNG 폴더 저장을 완료하지 못했습니다. ${error instanceof Error ? error.message : ''}`);
    } finally {
      setOperationProgress('');
      setIsSavingFolder(false);
      folderSaveCancelRef.current = false;
    }
  }

  function cancelFolderSave() {
    folderSaveCancelRef.current = true;
    setOperationProgress('PNG 폴더 저장 취소 중');
  }

  function removeBox(id: string) {
    applyBoxes(
      boxes.filter((box) => box.id !== id),
      '선택한 얼굴 영역을 삭제했습니다.',
      selectedBoxId === id ? '' : selectedBoxId
    );
  }

  function resetAll() {
    applyBoxes([], '모든 얼굴 영역을 지웠습니다. 원본 사진은 그대로 유지됩니다.', '');
  }

  function applyPreset(preset: DetectionPreset) {
    setDetectionPreset(preset);
    setConfidence(DETECTION_PRESETS[preset].confidence);
    setBoxExpansion(DETECTION_PRESETS[preset].expansion);
    setStatus(`${DETECTION_PRESETS[preset].label} 감지 프리셋을 선택했습니다. 자동 감지를 다시 실행해 주세요.`);
  }

  function updateSelectedBox(patch: Partial<FaceBox>) {
    if (!selectedBoxId) return;
    applyBoxes(
      boxes.map((box) => (box.id === selectedBoxId ? { ...box, ...patch } : box)),
      '선택한 영역의 가림 옵션을 변경했습니다.',
      selectedBoxId
    );
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

      <section className="mx-auto grid max-w-7xl gap-5 px-5 py-5 lg:grid-cols-[340px_minmax(0,1fr)]">
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
                multiple
                onChange={(event) => {
                  const files = event.target.files;
                  if (files) loadFiles(files);
                  event.currentTarget.value = '';
                }}
              />
              <ImageUp className="mb-3 text-blue-600" size={30} />
              <strong className="text-sm">JPG 또는 PNG 여러 장 선택</strong>
              <span className="mt-1 text-xs text-slate-500">파일은 브라우저 메모리에서만 사용됩니다.</span>
            </label>
            {currentPhoto && (
              <dl className="mt-4 grid gap-2 text-sm">
                <div className="flex justify-between gap-3">
                  <dt className="text-slate-500">현재 파일</dt>
                  <dd className="truncate font-bold">{currentPhoto.name}</dd>
                </div>
                <div className="flex justify-between gap-3">
                  <dt className="text-slate-500">크기</dt>
                  <dd className="font-bold">
                    {imageSize.width} x {imageSize.height}
                  </dd>
                </div>
              </dl>
            )}
            {memoryWarning && <p className="mt-3 rounded-lg bg-amber-50 p-3 text-xs font-bold text-amber-800">{memoryWarning}</p>}
          </section>

          {photos.length > 0 && (
            <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
              <h2 className="mb-3 flex items-center gap-2 text-base font-black">
                <Images size={19} />
                작업 목록
              </h2>
              <div className="grid max-h-60 gap-2 overflow-auto">
                {photos.map((photo, index) => {
                  const count = (photo.id === currentPhotoId ? boxes : boxesByPhoto[photo.id] ?? []).length;
                  return (
                    <button
                      className={`grid gap-1 rounded-lg border p-3 text-left text-sm ${
                        photo.id === currentPhotoId ? 'border-blue-500 bg-blue-50' : 'border-slate-200 bg-white'
                      }`}
                      key={photo.id}
                      onClick={() => switchPhoto(photo)}
                      type="button"
                    >
                      <span className="font-black">
                        #{index + 1} {photo.name}
                      </span>
                      <span className="text-xs font-semibold text-slate-500">등록 영역 {count}개</span>
                    </button>
                  );
                })}
              </div>
            </section>
          )}

          <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
            <h2 className="mb-3 flex items-center gap-2 text-base font-black">
              <ScanFace size={19} />
              얼굴 감지
            </h2>
            <div className="mb-3 grid grid-cols-3 gap-2">
              {(Object.keys(DETECTION_PRESETS) as DetectionPreset[]).map((preset) => (
                <button
                  className={`min-h-10 rounded-lg border px-2 text-sm font-black ${
                    detectionPreset === preset ? 'border-blue-600 bg-blue-600 text-white' : 'border-slate-200 bg-white text-slate-700'
                  }`}
                  key={preset}
                  onClick={() => applyPreset(preset)}
                  type="button"
                >
                  {DETECTION_PRESETS[preset].label}
                </button>
              ))}
            </div>
            <p className="mb-3 rounded-lg bg-slate-50 p-3 text-xs font-semibold text-slate-600">{DETECTION_PRESETS[detectionPreset].note}</p>
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
            <label className="mb-3 grid gap-2 text-sm font-bold">
              얼굴 박스 확장률
              <input
                min="0"
                max="0.45"
                step="0.01"
                type="range"
                value={boxExpansion}
                onChange={(event) => setBoxExpansion(Number(event.target.value))}
              />
              <span className="text-xs text-slate-500">얼굴 주변 여유를 {(boxExpansion * 100).toFixed(0)}% 포함합니다.</span>
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
            <button
              className="mt-2 flex min-h-11 w-full items-center justify-center gap-2 rounded-lg bg-blue-600 px-4 text-sm font-black text-white disabled:opacity-50"
              disabled={photos.length === 0 || isDetecting || isBatchDetecting}
              onClick={runBatchFaceDetection}
              type="button"
            >
              {isBatchDetecting ? <Loader2 className="animate-spin" size={18} /> : <Images size={18} />}
              전체 사진 일괄 감지
            </button>
            {pendingAutoBoxes.length > 0 && (
              <div className="mt-3 grid gap-2 rounded-lg border border-blue-200 bg-blue-50 p-3">
                <p className="text-xs font-bold text-blue-800">자동 감지 후보 {pendingAutoBoxes.length}개를 미리보기 중입니다.</p>
                <div className="grid grid-cols-2 gap-2">
                  <button className="min-h-9 rounded-lg bg-blue-600 text-sm font-black text-white" onClick={applyPendingAutoBoxes} type="button">
                    감지 결과 적용
                  </button>
                  <button className="min-h-9 rounded-lg border border-blue-200 bg-white text-sm font-black text-blue-700" onClick={cancelPendingAutoBoxes} type="button">
                    취소
                  </button>
                </div>
              </div>
            )}
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
            <p className="mt-3 rounded-lg bg-slate-50 p-3 text-xs font-bold text-slate-600">가장 안전한 게시용 기본값은 모자이크 또는 검은 박스입니다. 블러는 강도를 충분히 높여 주세요.</p>
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
                className={`min-h-10 rounded-lg border text-sm font-black ${
                  toolMode === 'brush' ? 'border-orange-500 bg-orange-500 text-white' : 'border-slate-200 bg-slate-50 text-slate-700'
                }`}
                disabled={!imageRef.current}
                onClick={() => setToolMode((current) => (current === 'brush' ? 'view' : 'brush'))}
                type="button"
              >
                {toolMode === 'brush' ? '브러시 칠하기 중' : '브러시로 민감정보 칠하기'}
              </button>
              <div className="grid grid-cols-2 gap-2">
                {(['rect', 'ellipse'] as MaskShape[]).map((shape) => (
                  <button
                    className={`min-h-9 rounded-lg border text-xs font-black ${
                      manualShape === shape ? 'border-slate-950 bg-slate-950 text-white' : 'border-slate-200 bg-white text-slate-700'
                    }`}
                    key={shape}
                    onClick={() => setManualShape(shape)}
                    type="button"
                  >
                    {shape === 'rect' ? '사각형' : '타원형'}
                  </button>
                ))}
              </div>
              <label className="grid gap-2 text-sm font-bold">
                브러시 크기
                <input min="16" max="240" step="2" type="range" value={brushSize} onChange={(event) => setBrushSize(Number(event.target.value))} />
                <span className="text-xs text-slate-500">현재 {brushSize}px</span>
              </label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  className="flex min-h-10 items-center justify-center gap-2 rounded-lg border border-slate-200 bg-white text-sm font-black text-slate-700 disabled:opacity-50"
                  disabled={historyPast.length === 0}
                  onClick={undoBoxes}
                  type="button"
                >
                  <Undo2 size={16} />
                  되돌리기
                </button>
                <button
                  className="flex min-h-10 items-center justify-center gap-2 rounded-lg border border-slate-200 bg-white text-sm font-black text-slate-700 disabled:opacity-50"
                  disabled={historyFuture.length === 0}
                  onClick={redoBoxes}
                  type="button"
                >
                  <Redo2 size={16} />
                  다시 적용
                </button>
              </div>
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
          <div className="grid gap-3 rounded-lg border border-slate-200 bg-white p-4 shadow-sm md:grid-cols-6">
            <div className="rounded-lg bg-slate-100 p-3">
              <span className="text-xs font-bold text-slate-500">사진</span>
              <strong className="block text-2xl">{photos.length}</strong>
            </div>
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
            <div className="rounded-lg bg-emerald-50 p-3">
              <span className="text-xs font-bold text-emerald-700">처리된 사진</span>
              <strong className="block text-2xl text-emerald-800">{batchSummary.processed}</strong>
            </div>
            <div className="grid gap-2">
              <button
                className="flex min-h-10 items-center justify-center gap-2 rounded-lg bg-emerald-600 px-4 text-sm font-black text-white disabled:opacity-50"
                disabled={!imageRef.current}
                onClick={downloadPng}
                type="button"
              >
                <Download size={18} />
                PNG
              </button>
              <button
                className="flex min-h-10 items-center justify-center gap-2 rounded-lg bg-slate-950 px-4 text-sm font-black text-white disabled:opacity-50"
                disabled={photos.length === 0 || isSavingFolder}
                onClick={saveAllPngFolder}
                type="button"
              >
                {isSavingFolder ? <Loader2 className="animate-spin" size={18} /> : <Download size={18} />}
                PNG 폴더 저장
              </button>
            </div>
          </div>

          <div className="grid gap-3 rounded-lg border border-slate-200 bg-white p-4 shadow-sm md:grid-cols-4">
            <div className="rounded-lg bg-slate-50 p-3">
              <span className="text-xs font-bold text-slate-500">전체 등록 영역</span>
              <strong className="block text-2xl">{batchSummary.totalBoxes}</strong>
            </div>
            <div className="rounded-lg bg-blue-50 p-3">
              <span className="text-xs font-bold text-blue-700">전체 자동 영역</span>
              <strong className="block text-2xl text-blue-800">{batchSummary.autoBoxes}</strong>
            </div>
            <div className="rounded-lg bg-orange-50 p-3">
              <span className="text-xs font-bold text-orange-700">전체 수동 영역</span>
              <strong className="block text-2xl text-orange-800">{batchSummary.manualBoxes}</strong>
            </div>
            <button
              className="flex min-h-16 items-center justify-center gap-2 rounded-lg bg-blue-600 px-4 font-black text-white disabled:opacity-50"
              disabled={photos.length === 0 || isBatchDetecting}
              onClick={runBatchFaceDetection}
              type="button"
            >
              {isBatchDetecting ? <Loader2 className="animate-spin" size={18} /> : <ScanFace size={18} />}
              전체 사진 일괄 감지
            </button>
          </div>
          <p className="rounded-lg bg-slate-50 p-3 text-xs font-bold text-slate-600">
            일괄 감지 후 가림 방식이나 강도를 바꾸면 모든 사진의 저장 결과에 바로 반영됩니다. 최종 저장은 PNG 폴더 저장 버튼을 누르면 됩니다.
          </p>

          {(operationProgress || isSavingFolder) && (
            <div className="flex items-center justify-between gap-3 rounded-lg border border-blue-200 bg-blue-50 p-4 text-sm font-bold text-blue-800">
              <span>{operationProgress || '작업 처리 중'}</span>
              {isSavingFolder && (
                <button className="min-h-9 rounded-lg border border-blue-200 bg-white px-3 text-blue-700" onClick={cancelFolderSave} type="button">
                  저장 취소
                </button>
              )}
            </div>
          )}

          {readyDownload && (
            <div className="flex flex-col gap-3 rounded-lg border border-emerald-200 bg-emerald-50 p-4 text-sm font-bold text-emerald-900 md:flex-row md:items-center md:justify-between">
              <span>
                다운로드 준비됨: {readyDownload.label}
                <span className="ml-2 font-semibold text-emerald-700">{readyDownload.fileName}</span>
              </span>
              <a
                className="inline-flex min-h-10 items-center justify-center gap-2 rounded-lg bg-emerald-600 px-4 text-white"
                download={readyDownload.fileName}
                href={readyDownload.url}
              >
                <Download size={18} />
                다시 다운로드
              </a>
            </div>
          )}

          <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
            <div className="mb-3 flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
              <div>
                <h2 className="text-lg font-black">작업 화면</h2>
                <p className="text-sm text-slate-500">
                  {toolMode === 'manual'
                    ? '사진 위에서 드래그하면 수동 얼굴 영역이 추가됩니다.'
                    : '박스를 클릭해 선택하고, 드래그로 이동하거나 모서리 핸들로 크기를 조정하세요.'}
                </p>
              </div>
              <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-black text-slate-600">{status}</span>
            </div>
            <div className="overflow-auto rounded-lg border border-slate-200 bg-slate-950 p-2">
              <canvas
                className={`mx-auto block max-w-full rounded bg-white ${toolMode === 'manual' || toolMode === 'brush' ? 'cursor-crosshair' : 'cursor-move'}`}
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
            {selectedBox && (
              <div className="mb-4 grid gap-3 rounded-lg border border-yellow-200 bg-yellow-50 p-3">
                <div>
                  <strong className="text-sm">선택 영역 개별 설정</strong>
                  <p className="mt-1 text-xs font-semibold text-yellow-800">선택한 영역만 다른 가림 방식이나 형태로 조정할 수 있습니다.</p>
                </div>
                <div className="grid gap-2 md:grid-cols-2">
                  <div className="grid grid-cols-3 gap-2">
                    {[
                      ['blur', '블러'],
                      ['pixelate', '모자이크'],
                      ['black', '검은 박스']
                    ].map(([value, label]) => (
                      <button
                        className={`min-h-9 rounded-lg border px-2 text-xs font-black ${
                          (selectedBox.maskMode ?? maskMode) === value ? 'border-yellow-600 bg-yellow-500 text-white' : 'border-yellow-200 bg-white text-yellow-900'
                        }`}
                        key={value}
                        onClick={() => updateSelectedBox({ maskMode: value as MaskMode })}
                        type="button"
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    {(['rect', 'ellipse'] as MaskShape[]).map((shape) => (
                      <button
                        className={`min-h-9 rounded-lg border px-2 text-xs font-black ${
                          (selectedBox.shape ?? 'rect') === shape ? 'border-yellow-600 bg-yellow-500 text-white' : 'border-yellow-200 bg-white text-yellow-900'
                        }`}
                        key={shape}
                        onClick={() => updateSelectedBox({ shape })}
                        type="button"
                      >
                        {shape === 'rect' ? '사각형' : '타원형'}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}
            {boxes.length === 0 ? (
              <p className="rounded-lg bg-slate-50 p-4 text-sm font-semibold text-slate-500">
                등록된 얼굴 영역이 없습니다. 자동 감지를 실행하거나 수동으로 영역을 추가하세요.
              </p>
            ) : (
              <div className="grid gap-2">
                {boxes.map((box, index) => (
                  <article
                    className={`grid gap-3 rounded-lg border p-3 md:grid-cols-[64px_1fr_110px] ${
                      selectedBoxId === box.id ? 'border-yellow-400 bg-yellow-50' : 'border-slate-200 bg-slate-50'
                    }`}
                    key={box.id}
                  >
                    <strong className="text-sm">#{index + 1}</strong>
                    <button className="text-left text-sm text-slate-600" onClick={() => setSelectedBoxId(box.id)} type="button">
                      <b className={box.source === 'manual' ? 'text-orange-700' : 'text-blue-700'}>{box.source === 'manual' ? '수동 추가' : '자동 감지'}</b>
                      <span className="ml-2">
                        x {Math.round(box.x)}, y {Math.round(box.y)}, {Math.round(box.width)} x {Math.round(box.height)}
                      </span>
                    </button>
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
            자동 얼굴 감지는 일부 얼굴을 놓칠 수 있습니다. 게시 전에는 얼굴, 이름표, 작품 이름, 게시판 개인정보, 연락처, QR 코드 노출 여부를
            사용자가 반드시 최종 확인해야 합니다.
          </div>
        </section>
      </section>
    </main>
  );
}

export default App;
