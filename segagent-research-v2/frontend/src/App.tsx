import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import {
  AlertTriangle,
  BrainCircuit,
  Check,
  Copy,
  Eye,
  EyeOff,
  FileCheck2,
  FlaskConical,
  Loader2,
  MessageSquareText,
  ScanLine,
  ShieldCheck,
  Sparkles,
  X,
} from 'lucide-react';
import {
  addContours,
  approvalFromEvent,
  caseImageUrl,
  createCase,
  getCase,
  getRun,
  resumeRun,
  startRun,
  uploadEditedMask,
} from './api';
import MedicalViewer from './components/MedicalViewer';
import TracePanel from './components/TracePanel';
import UploadField from './components/UploadField';
import type { ApprovalRequest, ArtifactRef, CaseRecord, RunEvent, ViewerMask } from './types';

const SAVED_CASE = 'segagent.activeCaseId';
const SAVED_RUN = 'segagent.activeRunId';

const MASK_COLORS = [
  { color: 'red', swatch: '#fb7185' },
  { color: 'green', swatch: '#4ade80' },
  { color: 'blue', swatch: '#60a5fa' },
  { color: 'warm', swatch: '#f59e0b' },
  { color: 'cool', swatch: '#22d3ee' },
  { color: 'violet', swatch: '#a78bfa' },
  { color: 'winter', swatch: '#e2e8f0' },
];

type BusyState =
  | 'idle'
  | 'restoring'
  | 'uploading-image'
  | 'uploading-contours'
  | 'running'
  | 'reviewing';

type StreamOutcome = 'unknown' | 'approval' | 'completed' | 'failed';

function isNifti(file: File): boolean {
  const name = file.name.toLowerCase();
  return name.endsWith('.nii') || name.endsWith('.nii.gz');
}

function isAbortError(value: unknown): boolean {
  return value instanceof DOMException && value.name === 'AbortError';
}

function readSaved(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function save(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch {
    // The app still works when storage is blocked by the browser.
  }
}

function forget(key: string): void {
  try {
    localStorage.removeItem(key);
  } catch {
    // The app still works when storage is blocked by the browser.
  }
}

function viewerMask(artifact: ArtifactRef, index: number): ViewerMask {
  const palette = MASK_COLORS[index % MASK_COLORS.length];
  return { ...artifact, visible: true, ...palette };
}

function masksFromCase(record: CaseRecord): ViewerMask[] {
  return record.artifacts
    .filter(artifact => artifact.kind === 'contour' || artifact.kind === 'mask')
    .map(viewerMask);
}

export default function App() {
  const [caseRecord, setCaseRecord] = useState<CaseRecord | null>(null);
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [masks, setMasks] = useState<ViewerMask[]>([]);
  const [question, setQuestion] = useState('');
  const [runId, setRunId] = useState('');
  const [approval, setApproval] = useState<ApprovalRequest | null>(null);
  const [reviewFeedback, setReviewFeedback] = useState('');
  const [busy, setBusy] = useState<BusyState>(() => readSaved(SAVED_CASE) ? 'restoring' : 'idle');
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [viewerReady, setViewerReady] = useState(false);
  const activeCaseId = useRef('');
  const requestController = useRef<AbortController | null>(null);
  const reviewRef = useRef<HTMLElement>(null);
  const editFileRef = useRef<HTMLInputElement>(null);

  const answer = useMemo(
    () => [...events].reverse().find(event => event.type === 'answer')?.payload.text,
    [events],
  );
  const requiredMaskIds = useMemo(
    () => approval?.artifacts?.map(artifact => artifact.artifact_id) || [],
    [approval],
  );

  useEffect(() => {
    const controller = new AbortController();
    const restore = async () => {
      const savedCaseId = readSaved(SAVED_CASE);
      if (!savedCaseId) return;
      try {
        const restoredCase = await getCase(savedCaseId, controller.signal);
        activeCaseId.current = restoredCase.case_id;
        setCaseRecord(restoredCase);
        setMasks(masksFromCase(restoredCase));

        const savedRunId = readSaved(SAVED_RUN);
        if (savedRunId) {
          try {
            const history = await getRun(savedRunId, controller.signal);
            if (history.run.case_id === restoredCase.case_id) {
              setRunId(history.run.run_id);
              setEvents(history.events);
              if (history.run.status === 'waiting_approval') {
                const pending = [...history.events]
                  .reverse()
                  .find(event => event.type === 'approval_required');
                if (pending) setApproval(approvalFromEvent(pending));
              }
            } else {
              forget(SAVED_RUN);
            }
          } catch (caught) {
            if (isAbortError(caught)) throw caught;
            forget(SAVED_RUN);
          }
        }
        setNotice('Previous case restored.');
      } catch (caught) {
        if (!isAbortError(caught)) {
          forget(SAVED_CASE);
          forget(SAVED_RUN);
          setNotice('');
        }
      } finally {
        if (!controller.signal.aborted) setBusy('idle');
      }
    };
    void restore();
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (approval) reviewRef.current?.focus();
  }, [approval]);

  useEffect(() => () => requestController.current?.abort(), []);

  const addArtifact = (artifact: ArtifactRef) => {
    if (artifact.case_id !== activeCaseId.current) return;
    if (artifact.kind !== 'mask' && artifact.kind !== 'contour') return;
    setMasks(previous => {
      if (previous.some(item => item.artifact_id === artifact.artifact_id)) return previous;
      return [...previous, viewerMask(artifact, previous.length)];
    });
  };

  const acceptEvent = (event: RunEvent, expectedCaseId: string) => {
    if (event.case_id !== expectedCaseId || activeCaseId.current !== expectedCaseId) return;
    setEvents(previous => {
      if (previous.some(item => item.event_id === event.event_id)) return previous;
      return [...previous, event];
    });
    if (event.type === 'artifact' && event.payload.artifact) addArtifact(event.payload.artifact);
    if (event.type === 'approval_required') {
      const request = approvalFromEvent(event);
      setViewerReady(false);
      request.artifacts?.forEach(addArtifact);
      setApproval(request);
    }
    if (event.type === 'error') setError(event.payload.message || 'The run failed.');
  };

  const setNewCase = (created: CaseRecord) => {
    activeCaseId.current = created.case_id;
    setCaseRecord(created);
    setEvents([]);
    setRunId('');
    setApproval(null);
    setReviewFeedback('');
    setQuestion('');
    setViewerReady(false);
    setMasks(masksFromCase(created));
    save(SAVED_CASE, created.case_id);
    forget(SAVED_RUN);
  };

  const uploadImage = async (files: File[]) => {
    const file = files[0];
    if (!file || busy !== 'idle') return;
    if (!isNifti(file)) {
      setError('Choose a NIfTI file ending in .nii or .nii.gz.');
      return;
    }
    const controller = new AbortController();
    requestController.current = controller;
    setBusy('uploading-image');
    setError('');
    setNotice(`Uploading ${file.name}…`);
    try {
      const created = await createCase(file, [], controller.signal);
      setNewCase(created);
      setNotice('Scan uploaded.');
    } catch (caught) {
      if (!isAbortError(caught)) {
        setNotice('');
        setError(caught instanceof Error ? caught.message : String(caught));
      }
    } finally {
      requestController.current = null;
      setBusy('idle');
    }
  };

  const uploadContours = async (files: File[]) => {
    if (!caseRecord || !files.length || busy !== 'idle') return;
    const invalid = files.find(file => !isNifti(file));
    if (invalid) {
      setError(`${invalid.name} is not a .nii or .nii.gz file.`);
      return;
    }
    const expectedCaseId = caseRecord.case_id;
    const controller = new AbortController();
    requestController.current = controller;
    setBusy('uploading-contours');
    setError('');
    setNotice(`Adding ${files.length} contour${files.length === 1 ? '' : 's'}…`);
    try {
      const updated = await addContours(expectedCaseId, files, controller.signal);
      if (activeCaseId.current !== expectedCaseId) return;
      setCaseRecord(updated);
      setMasks(previous => {
        const known = new Map(previous.map(item => [item.artifact_id, item]));
        for (const contour of updated.contours) {
          if (!known.has(contour.artifact_id)) {
            known.set(contour.artifact_id, viewerMask(contour, known.size));
          }
        }
        return [...known.values()];
      });
      setNotice(`${files.length} contour${files.length === 1 ? '' : 's'} added.`);
    } catch (caught) {
      if (!isAbortError(caught)) {
        setNotice('');
        setError(caught instanceof Error ? caught.message : String(caught));
      }
    } finally {
      requestController.current = null;
      setBusy('idle');
    }
  };

  const run = async () => {
    if (!caseRecord || !viewerReady || !question.trim() || busy !== 'idle' || approval) return;
    const expectedCaseId = caseRecord.case_id;
    const controller = new AbortController();
    requestController.current = controller;
    setBusy('running');
    setEvents([]);
    setApproval(null);
    setRunId('');
    setError('');
    setNotice('Running…');
    setMasks(previous => previous.map(mask => (
      mask.kind === 'mask' ? { ...mask, visible: false } : mask
    )));
    forget(SAVED_RUN);
    const stream = { outcome: 'unknown' as StreamOutcome };
    try {
      await startRun(
        expectedCaseId,
        question.trim(),
        event => {
          if (event.type === 'approval_required') stream.outcome = 'approval';
          if (event.type === 'run_completed') stream.outcome = 'completed';
          if (event.type === 'error') stream.outcome = 'failed';
          acceptEvent(event, expectedCaseId);
        },
        id => {
          setRunId(id);
          save(SAVED_RUN, id);
        },
        controller.signal,
      );
      if (stream.outcome === 'approval') setNotice('Review the new masks.');
      else if (stream.outcome === 'completed') setNotice('Run complete.');
      else if (stream.outcome === 'failed') setNotice('');
      else setNotice('Run paused.');
    } catch (caught) {
      if (!isAbortError(caught)) {
        setNotice('');
        setError(caught instanceof Error ? caught.message : String(caught));
      }
    } finally {
      requestController.current = null;
      setBusy('idle');
    }
  };

  const review = async (decision: 'approve' | 'reject' | 'feedback') => {
    if (!runId || !caseRecord || busy !== 'idle' || !approval) return;
    if (decision === 'feedback' && !reviewFeedback.trim()) return;
    const currentApproval = approval;
    const expectedCaseId = caseRecord.case_id;
    const controller = new AbortController();
    requestController.current = controller;
    setBusy('reviewing');
    setApproval(null);
    setError('');
    setNotice('Saving review…');
    const stream: { outcome: StreamOutcome; approval: ApprovalRequest | null } = {
      outcome: 'unknown',
      approval: null,
    };
    try {
      await resumeRun(
        runId,
        decision,
        reviewFeedback.trim(),
        event => {
          if (event.type === 'approval_required') {
            stream.outcome = 'approval';
            stream.approval = approvalFromEvent(event);
          }
          if (event.type === 'run_completed') stream.outcome = 'completed';
          if (event.type === 'error') stream.outcome = 'failed';
          acceptEvent(event, expectedCaseId);
        },
        controller.signal,
      );
      setReviewFeedback('');
      if (stream.outcome === 'approval') setNotice('More masks need review.');
      else if (stream.outcome === 'completed') setNotice('Run complete.');
      else if (stream.outcome === 'failed') setNotice('');
      else setNotice('Review saved.');
    } catch (caught) {
      if (!isAbortError(caught)) {
        setNotice('');
        setApproval(stream.approval || currentApproval);
        setError(caught instanceof Error ? caught.message : String(caught));
      }
    } finally {
      requestController.current = null;
      setBusy('idle');
    }
  };

  const modifyContour = async (file: File) => {
    if (!runId || !caseRecord || busy !== 'idle' || !approval) return;
    const target = approval.artifacts?.[0];
    const currentApproval = approval;
    const expectedCaseId = caseRecord.case_id;
    const controller = new AbortController();
    requestController.current = controller;
    setBusy('reviewing');
    setApproval(null);
    setError('');
    setNotice('Uploading edited contour…');
    const stream: { outcome: StreamOutcome; approval: ApprovalRequest | null } = {
      outcome: 'unknown',
      approval: null,
    };
    try {
      const ref = await uploadEditedMask(
        expectedCaseId,
        target?.label || 'edited contour',
        file,
        target?.artifact_id,
        'unknown',
        controller.signal,
      );
      setNotice('Saving edited contour…');
      await resumeRun(
        runId,
        'modify',
        '',
        event => {
          if (event.type === 'approval_required') {
            stream.outcome = 'approval';
            stream.approval = approvalFromEvent(event);
          }
          if (event.type === 'run_completed') stream.outcome = 'completed';
          if (event.type === 'error') stream.outcome = 'failed';
          acceptEvent(event, expectedCaseId);
        },
        controller.signal,
        ref.artifact_id,
      );
      if (stream.outcome === 'approval') setNotice('More masks need review.');
      else if (stream.outcome === 'completed') setNotice('Run complete.');
      else if (stream.outcome === 'failed') setNotice('');
      else setNotice('Edited contour saved.');
    } catch (caught) {
      if (!isAbortError(caught)) {
        setNotice('');
        setApproval(stream.approval || currentApproval);
        setError(caught instanceof Error ? caught.message : String(caught));
      }
    } finally {
      requestController.current = null;
      setBusy('idle');
    }
  };

  const onQuestionKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
      event.preventDefault();
      void run();
    }
  };

  const toggleMask = (artifactId: string) => {
    if (requiredMaskIds.includes(artifactId)) setViewerReady(false);
    setMasks(items => items.map(item => (
      item.artifact_id === artifactId ? { ...item, visible: !item.visible } : item
    )));
  };

  const setAllMasks = (visible: boolean) => {
    if (requiredMaskIds.length) setViewerReady(false);
    setMasks(items => items.map(item => ({ ...item, visible })));
  };

  const copyCaseId = async () => {
    if (!caseRecord) return;
    try {
      await navigator.clipboard.writeText(caseRecord.case_id);
      setNotice('Case ID copied.');
    } catch {
      setNotice(caseRecord.case_id);
    }
  };

  const locked = busy !== 'idle';
  const reviewActions: ReadonlyArray<'approve' | 'feedback' | 'reject' | 'modify'> = approval?.allowed_decisions?.length
    ? approval.allowed_decisions
    : ['approve', 'feedback', 'reject'];
  const reviewMasks = approval?.artifacts || [];

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-mark"><BrainCircuit size={22} aria-hidden="true" /></div>
        <div className="brand-copy">
          <h1>SegAgent <span>Research v2</span></h1>
          <p>AI-assisted 3D segmentation and contour review</p>
        </div>
        <div className="research-badge"><FlaskConical size={14} aria-hidden="true" /> Research only</div>
      </header>

      <aside className="case-panel" aria-labelledby="case-heading">
        <h2 className="panel-heading" id="case-heading"><ScanLine size={18} aria-hidden="true" /> Case</h2>

        <div className="upload-stack">
          <UploadField
            id="scan-upload"
            title={caseRecord ? 'Start new case' : 'Upload scan'}
            hint="NIfTI file · .nii or .nii.gz"
            value={caseRecord?.source_name}
            icon={ScanLine}
            disabled={locked || Boolean(approval)}
            busy={busy === 'uploading-image'}
            onFiles={uploadImage}
          />
          <UploadField
            id="contour-upload"
            title="Add contours"
            hint={caseRecord ? 'Optional' : 'Upload a scan first'}
            value={caseRecord?.contours.length
              ? `${caseRecord.contours.length} contour${caseRecord.contours.length === 1 ? '' : 's'} added`
              : undefined}
            actionLabel="Add"
            icon={FileCheck2}
            multiple
            disabled={!caseRecord || locked || Boolean(approval)}
            busy={busy === 'uploading-contours'}
            onFiles={uploadContours}
          />
        </div>

        <div className="status-line" aria-live="polite">
          {busy === 'restoring' && <><Loader2 className="spin" size={14} /> Restoring your last case…</>}
          {busy !== 'idle' && busy !== 'restoring' && notice && <><Loader2 className="spin" size={14} /> {notice}</>}
          {busy === 'idle' && notice && <><Check size={14} /> {notice}</>}
        </div>

        {caseRecord && (
          <section className="case-card" aria-label="Current case">
            <div className="case-card-title">
              <span className={`ready-dot${approval ? ' needs-review' : ''}`} />
              <strong>{approval ? 'Review needed' : locked ? 'Working' : !viewerReady ? 'Loading scan' : 'Ready'}</strong>
            </div>
            <dl>
              <div><dt>Scan</dt><dd>{caseRecord.source_name}</dd></div>
              <div><dt>Contours</dt><dd>{caseRecord.contours.length}</dd></div>
              <div>
                <dt>Case ID</dt>
                <dd className="case-id"><code>{caseRecord.case_id}</code><button type="button" onClick={copyCaseId} aria-label="Copy case ID"><Copy size={13} /></button></dd>
              </div>
            </dl>
          </section>
        )}

        {masks.length > 0 && (
          <section className="mask-list" aria-labelledby="mask-heading">
            <div className="section-row">
              <h3 id="mask-heading">Contours and masks <span>{masks.length}</span></h3>
              <div className="mini-actions">
                <button type="button" onClick={() => setAllMasks(true)}>Show all</button>
                <button type="button" onClick={() => setAllMasks(false)}>Hide all</button>
              </div>
            </div>
            <div className="mask-items">
              {masks.map(mask => (
                <button
                  type="button"
                  className="mask-item"
                  key={mask.artifact_id}
                  onClick={() => toggleMask(mask.artifact_id)}
                  aria-pressed={mask.visible}
                  title={`${mask.visible ? 'Hide' : 'Show'} ${mask.label}`}
                >
                  <i style={{ backgroundColor: mask.visible ? mask.swatch : '#64748b' }} />
                  <span><strong>{mask.label}</strong><small>{mask.kind === 'contour' ? 'Uploaded' : 'Generated'}</small></span>
                  {mask.visible ? <Eye size={15} aria-hidden="true" /> : <EyeOff size={15} aria-hidden="true" />}
                </button>
              ))}
            </div>
          </section>
        )}
      </aside>

      <main className="viewer-panel" aria-label="Medical image viewer">
        <MedicalViewer
          key={caseRecord?.case_id || 'empty'}
          imageUrl={caseRecord ? caseImageUrl(caseRecord.case_id) : null}
          masks={masks}
          requiredMaskIds={requiredMaskIds}
          onReadyChange={setViewerReady}
        />
      </main>

      <aside className="agent-panel" aria-labelledby="agent-heading">
        <h2 className="panel-heading" id="agent-heading"><MessageSquareText size={18} aria-hidden="true" /> Ask SegAgent</h2>

        <div className="prompt-suggestions" aria-label="Example requests">
          <button type="button" disabled={!caseRecord || !viewerReady || locked || Boolean(approval)} onClick={() => setQuestion('Segment the liver.')}>Liver</button>
          <button type="button" disabled={!caseRecord || !viewerReady || locked || Boolean(approval)} onClick={() => setQuestion('Segment the left kidney and right kidney.')}>Kidneys</button>
          <button type="button" disabled={!caseRecord?.contours.length || !viewerReady || locked || Boolean(approval)} onClick={() => setQuestion('Check contours.')}>Check contours</button>
        </div>

        <div className="composer">
          <label htmlFor="agent-question">What should the agent do?</label>
          <textarea
            id="agent-question"
            value={question}
            onChange={event => setQuestion(event.target.value)}
            onKeyDown={onQuestionKeyDown}
            disabled={!caseRecord || !viewerReady || locked || Boolean(approval)}
            placeholder={!caseRecord ? 'Upload a scan first' : !viewerReady ? 'Wait for the scan to load' : 'Example: Segment the liver'}
          />
          <div className="composer-footer">
            <span>⌘/Ctrl + Enter</span>
            <button
              type="button"
              className="primary run-button"
              disabled={!caseRecord || !viewerReady || !question.trim() || locked || Boolean(approval)}
              onClick={run}
            >
              {busy === 'running' ? <Loader2 className="spin" size={17} /> : <Sparkles size={17} />}
              {busy === 'running' ? 'Running…' : 'Run'}
            </button>
          </div>
        </div>

        {approval && (
          <section className="approval-card" ref={reviewRef} tabIndex={-1} aria-labelledby="review-heading">
            <div className="approval-title">
              <span><ShieldCheck size={19} aria-hidden="true" /></span>
              <div><h3 id="review-heading">Review results</h3><p>{reviewMasks.length} new mask{reviewMasks.length === 1 ? '' : 's'} ready</p></div>
            </div>
            <p className="review-help">Check the masks in the viewer, then choose what to do.</p>
            {!viewerReady && <p className="review-warning">Show every new mask and wait for it to load before approving.</p>}

            {reviewMasks.length > 0 && (
              <div className="review-mask-list">
                {reviewMasks.map(artifact => {
                  const mask = masks.find(item => item.artifact_id === artifact.artifact_id);
                  return (
                    <button type="button" key={artifact.artifact_id} onClick={() => toggleMask(artifact.artifact_id)} aria-pressed={mask?.visible ?? true}>
                      <i style={{ backgroundColor: mask?.swatch || '#60a5fa' }} />
                      <span>{artifact.label}</span>
                      {mask?.visible === false ? <EyeOff size={14} /> : <Eye size={14} />}
                    </button>
                  );
                })}
              </div>
            )}

            {approval.summary && (
              <details className="review-details">
                <summary>View measurements</summary>
                <p>{approval.summary}</p>
              </details>
            )}

            {reviewActions.includes('feedback') && (
              <>
                <label htmlFor="review-note">What should change?</label>
                <textarea
                  id="review-note"
                  value={reviewFeedback}
                  onChange={event => setReviewFeedback(event.target.value)}
                  placeholder="Add a note to request changes"
                />
              </>
            )}
            <div className="review-actions">
              {reviewActions.includes('approve') && (
                <button type="button" className="approve" disabled={busy !== 'idle' || !viewerReady} onClick={() => review('approve')}><Check size={16} /> Approve</button>
              )}
              {reviewActions.includes('feedback') && (
                <button type="button" className="feedback" disabled={busy !== 'idle' || !reviewFeedback.trim()} onClick={() => review('feedback')}><AlertTriangle size={16} /> Request changes</button>
              )}
              {reviewActions.includes('modify') && (
                <button type="button" className="feedback" disabled={busy !== 'idle'} onClick={() => editFileRef.current?.click()} title="Upload an edited contour (.nii/.nii.gz)"><FileCheck2 size={16} /> Upload edit</button>
              )}
              {reviewActions.includes('reject') && (
                <button type="button" className="reject" disabled={busy !== 'idle'} onClick={() => review('reject')}><X size={16} /> Reject</button>
              )}
            </div>
            <input
              ref={editFileRef}
              type="file"
              accept=".nii,.nii.gz"
              className="visually-hidden"
              onChange={event => {
                const file = event.target.files?.[0];
                event.target.value = '';
                if (file) void modifyContour(file);
              }}
            />
          </section>
        )}

        {busy === 'reviewing' && (
          <div className="inline-progress" role="status"><Loader2 className="spin" size={16} /> Saving review…</div>
        )}

        {answer && <section className="answer-card"><h3>Result</h3><p>{answer}</p></section>}

        {error && (
          <div className="error-banner" role="alert">
            <AlertTriangle size={17} aria-hidden="true" />
            <span>{error}</span>
            <button type="button" onClick={() => setError('')} aria-label="Dismiss error"><X size={15} /></button>
          </div>
        )}

        <section className="trace-section" aria-labelledby="activity-heading">
          <h2 className="trace-heading" id="activity-heading"><ActivityLabel /> Activity</h2>
          <div className="trace-scroll"><TracePanel events={events} /></div>
        </section>
      </aside>
    </div>
  );
}

function ActivityLabel() {
  return <span className="pulse-dot" aria-hidden="true" />;
}
