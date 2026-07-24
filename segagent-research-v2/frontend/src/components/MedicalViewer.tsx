import { useEffect, useRef, useState } from 'react';
import { AlertTriangle, Loader2, RotateCcw, ScanLine } from 'lucide-react';
import { Niivue, NVImage, SLICE_TYPE } from '@niivue/niivue';
import { artifactUrl } from '../api';
import type { ViewerMask } from '../types';

interface Props {
  imageUrl: string | null;
  masks: ViewerMask[];
  requiredMaskIds?: string[];
  onReadyChange?: (ready: boolean) => void;
}

type ViewerState = 'empty' | 'loading' | 'ready' | 'error';

const VIEWS: Array<{ label: string; value: SLICE_TYPE }> = [
  { label: 'Axial', value: SLICE_TYPE.AXIAL },
  { label: 'Coronal', value: SLICE_TYPE.CORONAL },
  { label: 'Sagittal', value: SLICE_TYPE.SAGITTAL },
  { label: 'Multi', value: SLICE_TYPE.MULTIPLANAR },
];

export default function MedicalViewer({
  imageUrl,
  masks,
  requiredMaskIds = [],
  onReadyChange,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const niivueRef = useRef<Niivue | null>(null);
  const initializationRef = useRef<Promise<Niivue> | null>(null);
  const imageLoadRef = useRef<Promise<unknown> | null>(null);
  const readyCallback = useRef(onReadyChange);
  const loadedImageUrl = useRef('');
  const imageGeneration = useRef(0);
  const maskGeneration = useRef(0);
  const [view, setView] = useState<SLICE_TYPE>(SLICE_TYPE.MULTIPLANAR);
  const [state, setState] = useState<ViewerState>(imageUrl ? 'loading' : 'empty');
  const [message, setMessage] = useState('');
  const [retry, setRetry] = useState(0);
  const [maskRetry, setMaskRetry] = useState(0);

  useEffect(() => {
    readyCallback.current = onReadyChange;
  }, [onReadyChange]);

  useEffect(() => {
    if (!canvasRef.current) return;
    let observer: ResizeObserver | null = null;
    let cancelled = false;
    const nv = new Niivue({
      backColor: [0.047, 0.051, 0.059, 1],
      show3Dcrosshair: true,
    });
    niivueRef.current = nv;
    const initialize = nv.attachToCanvas(canvasRef.current, false).then(attached => {
      if (cancelled) return attached;
      attached.setSliceType(SLICE_TYPE.MULTIPLANAR);
      attached.setMultiplanarLayout(2);
      observer = new ResizeObserver(() => attached.resizeListener());
      if (containerRef.current) observer.observe(containerRef.current);
      return attached;
    });
    initializationRef.current = initialize;
    // A scan-loading effect reports initialization errors in the viewer.
    void initialize.catch(() => undefined);

    return () => {
      cancelled = true;
      observer?.disconnect();
      if (niivueRef.current === nv) niivueRef.current = null;
      if (initializationRef.current === initialize) initializationRef.current = null;
      const dispose = () => {
        for (let index = nv.volumes.length - 1; index >= 0; index -= 1) {
          nv.removeVolume(nv.volumes[index]);
        }
        nv.cleanup();
      };
      const pendingImageLoad = imageLoadRef.current;
      void initialize.then(
        () => Promise.resolve(pendingImageLoad).then(dispose, dispose),
        dispose,
      ).catch(() => undefined);
    };
  }, [retry]);

  useEffect(() => {
    const nv = niivueRef.current;
    const initialize = initializationRef.current;
    const generation = ++imageGeneration.current;
    maskGeneration.current += 1;
    loadedImageUrl.current = '';
    readyCallback.current?.(false);

    if (!nv || !initialize || !imageUrl) return;

    const load = async () => {
      await initialize;
      if (generation !== imageGeneration.current) return;
      setState('loading');
      setMessage('Loading scan…');
      for (let index = nv.volumes.length - 1; index >= 0; index -= 1) {
        nv.removeVolume(nv.volumes[index]);
      }
      const imageLoad = nv.loadVolumes([{ url: imageUrl }]);
      imageLoadRef.current = imageLoad;
      await imageLoad;
      if (imageLoadRef.current === imageLoad) imageLoadRef.current = null;
      if (generation !== imageGeneration.current) return;
      nv.setSliceType(SLICE_TYPE.MULTIPLANAR);
      nv.setMultiplanarLayout(2);
      setView(SLICE_TYPE.MULTIPLANAR);
      nv.updateGLVolume();
      loadedImageUrl.current = imageUrl;
      setState('ready');
      setMessage('');
    };

    load().catch(() => {
      if (generation !== imageGeneration.current) return;
      setState('error');
      setMessage('Could not load this scan. Check that it is a valid 3D NIfTI file.');
      readyCallback.current?.(false);
    });
    return () => {
      if (imageGeneration.current === generation) imageGeneration.current += 1;
    };
  }, [imageUrl, retry]);

  useEffect(() => {
    const nv = niivueRef.current;
    if (!nv || state !== 'ready' || loadedImageUrl.current !== imageUrl) return;
    const generation = ++maskGeneration.current;
    let requiredFailed = false;
    const required = new Set(requiredMaskIds);
    readyCallback.current?.(false);

    const sync = async () => {
      await Promise.resolve();
      if (generation !== maskGeneration.current) return;
      setMessage('');
      const masksToLoad = masks
        .filter(mask => mask.visible || required.has(mask.artifact_id))
        .sort((left, right) => Number(required.has(right.artifact_id)) - Number(required.has(left.artifact_id)));
      const wanted = new Set(masksToLoad.map(mask => mask.artifact_id));
      for (let index = nv.volumes.length - 1; index >= 1; index -= 1) {
        const volume = nv.volumes[index];
        if (!wanted.has(volume.name)) nv.removeVolume(volume);
      }

      for (const mask of masksToLoad) {
        if (generation !== maskGeneration.current) return;
        const existing = nv.volumes.find(volume => volume.name === mask.artifact_id);
        if (existing) {
          existing.opacity = mask.visible ? 0.52 : 0;
          continue;
        }
        try {
          const volume = await NVImage.loadFromUrl({
            url: artifactUrl(mask.case_id, mask.artifact_id),
            name: mask.artifact_id,
            colormap: mask.color,
            opacity: mask.visible ? 0.52 : 0,
          });
          if (generation !== maskGeneration.current) return;
          volume.name = mask.artifact_id;
          nv.addVolume(volume);
        } catch {
          if (generation !== maskGeneration.current) return;
          if (required.has(mask.artifact_id)) requiredFailed = true;
          setMessage(`Could not load the mask “${mask.label}”.`);
        }
      }

      if (generation !== maskGeneration.current) return;
      nv.updateGLVolume();
      const requiredReady = requiredMaskIds.every(artifactId => {
        const mask = masks.find(item => item.artifact_id === artifactId);
        return Boolean(
          mask?.visible
          && nv.volumes.some(volume => volume.name === artifactId),
        );
      });
      readyCallback.current?.(!requiredFailed && requiredReady);
    };

    void sync();
    return () => {
      if (maskGeneration.current === generation) maskGeneration.current += 1;
    };
  }, [imageUrl, masks, requiredMaskIds, state, maskRetry]);

  const changeView = (next: SLICE_TYPE) => {
    const nv = niivueRef.current;
    setView(next);
    if (!nv) return;
    nv.setSliceType(next);
    if (next === SLICE_TYPE.MULTIPLANAR) nv.setMultiplanarLayout(2);
    nv.updateGLVolume();
  };

  return (
    <div className="viewer" ref={containerRef}>
      <canvas
        ref={canvasRef}
        role="img"
        aria-label="3D medical scan and segmentation masks"
        aria-describedby="viewer-help"
      >
        3D medical image viewer
      </canvas>
      <p className="visually-hidden" id="viewer-help">Use the view buttons to change the scan direction.</p>

      {state === 'empty' && (
        <div className="viewer-message">
          <span><ScanLine size={28} aria-hidden="true" /></span>
          <strong>Upload a scan to begin</strong>
          <p>Choose a .nii or .nii.gz file.</p>
        </div>
      )}

      {state === 'loading' && (
        <div className="viewer-message" role="status">
          <span><Loader2 className="spin" size={26} aria-hidden="true" /></span>
          <strong>Loading scan…</strong>
          <p>This can take a moment for large files.</p>
        </div>
      )}

      {state === 'error' && (
        <div className="viewer-message viewer-error" role="alert">
          <span><AlertTriangle size={26} aria-hidden="true" /></span>
          <strong>Scan could not be shown</strong>
          <p>{message}</p>
          <button
            type="button"
            onClick={() => {
              setState('loading');
              setMessage('Loading scan…');
              setRetry(value => value + 1);
            }}
          ><RotateCcw size={15} /> Retry</button>
        </div>
      )}

      {state === 'ready' && message && (
        <div className="viewer-warning" role="alert">
          <AlertTriangle size={15} />
          <span>{message}</span>
          <button type="button" onClick={() => setMaskRetry(value => value + 1)}><RotateCcw size={14} /> Retry</button>
        </div>
      )}

      {state === 'ready' && (
        <div className="viewer-toolbar" aria-label="View direction">
          {VIEWS.map(item => (
            <button
              type="button"
              key={item.label}
              className={view === item.value ? 'active' : ''}
              aria-pressed={view === item.value}
              onClick={() => changeView(item.value)}
            >
              {item.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
