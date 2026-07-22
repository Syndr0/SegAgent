import { useEffect, useRef, useState } from 'react';
import { Niivue, NVImage, SLICE_TYPE } from '@niivue/niivue';
import { artifactUrl } from '../api';
import type { ViewerMask } from '../types';

interface Props {
  imageUrl: string | null;
  masks: ViewerMask[];
}

export default function MedicalViewer({ imageUrl, masks }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const niivueRef = useRef<Niivue | null>(null);
  const loadedRef = useRef(new Set<string>());
  const [view, setView] = useState<SLICE_TYPE>(SLICE_TYPE.MULTIPLANAR);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (!canvasRef.current) return;
    const nv = new Niivue({ backColor: [0.015, 0.03, 0.055, 1], show3Dcrosshair: true });
    nv.attachToCanvas(canvasRef.current);
    nv.setSliceType(SLICE_TYPE.MULTIPLANAR);
    nv.setMultiplanarLayout(2);
    niivueRef.current = nv;
    const resize = () => nv.resizeListener();
    const observer = new ResizeObserver(resize);
    if (containerRef.current) observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const nv = niivueRef.current;
    if (!nv || !imageUrl) return;
    let active = true;
    const load = async () => {
      setReady(false);
      nv.volumes = [];
      loadedRef.current.clear();
      await nv.loadVolumes([{ url: imageUrl }]);
      if (active) {
        nv.updateGLVolume();
        setReady(true);
      }
    };
    load().catch(() => setReady(false));
    return () => { active = false; };
  }, [imageUrl]);

  useEffect(() => {
    const nv = niivueRef.current;
    if (!nv || !ready) return;
    let active = true;
    const sync = async () => {
      const wanted = new Set(masks.map(mask => mask.artifact_id));
      for (let index = nv.volumes.length - 1; index >= 1; index -= 1) {
        const volume = nv.volumes[index];
        if (!wanted.has(volume.name)) {
          nv.removeVolume(volume);
          loadedRef.current.delete(volume.name);
        }
      }
      for (const mask of masks) {
        const existing = nv.volumes.find(volume => volume.name === mask.artifact_id);
        if (existing) {
          existing.opacity = mask.visible ? 0.52 : 0;
          continue;
        }
        const volume = await NVImage.loadFromUrl({
          url: artifactUrl(mask.case_id, mask.artifact_id),
          name: mask.artifact_id,
          colormap: mask.color,
          opacity: mask.visible ? 0.52 : 0,
        });
        if (!active) return;
        volume.name = mask.artifact_id;
        nv.addVolume(volume);
        loadedRef.current.add(mask.artifact_id);
      }
      nv.updateGLVolume();
    };
    sync().catch(console.error);
    return () => { active = false; };
  }, [masks, ready]);

  const changeView = (next: SLICE_TYPE) => {
    const nv = niivueRef.current;
    setView(next);
    if (nv) {
      nv.setSliceType(next);
      if (next === SLICE_TYPE.MULTIPLANAR) nv.setMultiplanarLayout(2);
      nv.updateGLVolume();
    }
  };

  return (
    <div className="viewer" ref={containerRef}>
      {!imageUrl && <div className="viewer-empty">Create a case to load a NIfTI volume.</div>}
      <canvas ref={canvasRef} />
      {imageUrl && (
        <div className="viewer-toolbar">
          {[
            ['Axial', SLICE_TYPE.AXIAL],
            ['Coronal', SLICE_TYPE.CORONAL],
            ['Sagittal', SLICE_TYPE.SAGITTAL],
            ['Multi', SLICE_TYPE.MULTIPLANAR],
          ].map(([label, value]) => (
            <button
              key={label as string}
              className={view === value ? 'active' : ''}
              onClick={() => changeView(value as SLICE_TYPE)}
            >
              {label as string}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

