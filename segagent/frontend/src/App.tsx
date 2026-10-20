import { useEffect, useRef, useState, type KeyboardEvent } from 'react';
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
  Pencil,
  ScanLine,
  SendHorizonal,
  X,
} from 'lucide-react';
import {
  addContours,
  caseImageUrl,
  createCase,
  getCase,
  getConversation,
  getRun,
  sendMessage,
  uploadEditedMask,
} from './api';
import MedicalViewer from './components/MedicalViewer';
import ChatThread from './components/ChatThread';
import UploadField from './components/UploadField';
import type { ArtifactRef, CaseRecord, ChatMessage, RunEvent, ViewerMask } from './types';

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
  | 'editing';

type StreamOutcome = 'unknown' | 'completed' | 'failed';

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

// A tumor-target (GTV) mask is never treated as validated: the agent segments
// what was described, it does not confirm disease, so these are flagged for
// mandatory clinician review rather than shown as final.
function isTargetMask(mask: ViewerMask): boolean {
  return mask.metadata?.target_type === 'gtv';
}

function targetTypeOf(mask: ViewerMask): string {
  return typeof mask.metadata?.target_type === 'string' ? mask.metadata.target_type : 'unknown';
}

export default function App() {
  const [caseRecord, setCaseRecord] = useState<CaseRecord | null>(null);
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [history, setHistory] = useState<ChatMessage[]>([]);
  const [awaitingReply, setAwaitingReply] = useState(false);
  const [masks, setMasks] = useState<ViewerMask[]>([]);
  const [question, setQuestion] = useState('');
  const [runId, setRunId] = useState('');
  const [busy, setBusy] = useState<BusyState>(() => readSaved(SAVED_CASE) ? 'restoring' : 'idle');
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [viewerReady, setViewerReady] = useState(false);
  const activeCaseId = useRef('');
  const requestController = useRef<AbortController | null>(null);
  const sendController = useRef<AbortController | null>(null);
  const editFileRef = useRef<HTMLInputElement>(null);
  const editTarget = useRef<ViewerMask | null>(null);

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

        const conversationRunIds: string[] = [];
        let conversationPendingRunId: string | null = null;
        let conversationLoaded = false;
        try {
          const conversation = await getConversation(restoredCase.case_id, controller.signal);
          setHistory(conversation.messages);
          conversationPendingRunId = conversation.conversation.pending_run_id || null;
          setAwaitingReply(Boolean(conversationPendingRunId));
          conversationLoaded = true;
          for (const message of conversation.messages) {
            if (message.run_id && !conversationRunIds.includes(message.run_id)) {
              conversationRunIds.push(message.run_id);
            }
          }
          if (
            conversationPendingRunId
            && !conversationRunIds.includes(conversationPendingRunId)
          ) {
            conversationRunIds.push(conversationPendingRunId);
          }
        } catch (caught) {
          if (isAbortError(caught)) throw caught;
        }

        const savedRunId = readSaved(SAVED_RUN);
        if (savedRunId && !conversationRunIds.includes(savedRunId)) {
          conversationRunIds.push(savedRunId);
        }
        if (conversationRunIds.length) {
          const histories = await Promise.allSettled(
            conversationRunIds.map(id => getRun(id, controller.signal)),
          );
          if (controller.signal.aborted) return;
          const available = histories.flatMap(result =>
            result.status === 'fulfilled' && result.value.run.case_id === restoredCase.case_id
              ? [result.value]
              : [],
          );
          setEvents(available.flatMap(item => item.events));

          const active = available.find(item => item.run.run_id === savedRunId)
            || available.find(item => item.run.run_id === conversationPendingRunId)
            || available[available.length - 1];
          if (active) {
            setRunId(active.run.run_id);
            if (!conversationLoaded) setAwaitingReply(active.run.status === 'waiting_reply');
          }
          if (savedRunId && !available.some(item => item.run.run_id === savedRunId)) {
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

  useEffect(() => () => {
    const activeRequest = requestController.current;
    const activeSend = sendController.current;
    requestController.current = null;
    sendController.current = null;
    activeSend?.abort();
    if (activeRequest !== activeSend) activeRequest?.abort();
  }, []);

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
    if (event.type === 'question_pending') setAwaitingReply(true);
    if (event.type === 'reply_recorded') setAwaitingReply(false);
    if (event.type === 'error') setError(event.payload.message || 'The run failed.');
  };

  const setNewCase = (created: CaseRecord) => {
    activeCaseId.current = created.case_id;
    setCaseRecord(created);
    setEvents([]);
    setHistory([]);
    setAwaitingReply(false);
    setRunId('');
    setQuestion('');
    setViewerReady(false);
    setMasks(masksFromCase(created));
    save(SAVED_CASE, created.case_id);
    forget(SAVED_RUN);
  };

  const uploadImage = async (files: File[]) => {
    const file = files[0];
    if (!file || busy !== 'idle' || requestController.current) return;
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
      if (requestController.current === controller) {
        requestController.current = null;
        setBusy('idle');
      }
    }
  };

  const uploadContours = async (files: File[]) => {
    if (!caseRecord || !files.length || busy !== 'idle' || requestController.current) return;
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
      if (requestController.current === controller) {
        requestController.current = null;
        setBusy('idle');
      }
    }
  };

  // One send action: the server continues a pending question or opens a new
  // turn, so the composer behaves like an ordinary chat box.
  const send = async () => {
    if (
      sendController.current
      || requestController.current
      || !caseRecord
      || !viewerReady
      || !question.trim()
      || busy !== 'idle'
    ) return;
    const expectedCaseId = caseRecord.case_id;
    const text = question.trim();
    const controller = new AbortController();
    // This ref is the synchronous single-flight lock. Unlike React state, it is
    // visible to a second invocation before the next render commits.
    sendController.current = controller;
    requestController.current = controller;
    setBusy('running');
    setQuestion('');
    setError('');
    setNotice(awaitingReply ? 'Sending reply…' : 'Working…');
    const stream = { outcome: 'unknown' as StreamOutcome };
    const visibleMaskIds = masks
      .filter(mask => mask.visible)
      .map(mask => mask.artifact_id);
    try {
      await sendMessage(
        expectedCaseId,
        text,
        event => {
          if (event.type === 'run_completed') stream.outcome = 'completed';
          if (event.type === 'error') stream.outcome = 'failed';
          acceptEvent(event, expectedCaseId);
        },
        id => {
          setRunId(id);
          save(SAVED_RUN, id);
        },
        controller.signal,
        {
          visible_mask_ids: visibleMaskIds,
          // Visibility is ambient viewer state, not an explicit reference.
          // Keeping these separate prevents a 15–20-mask OAR view from being
          // interpreted as though the user named every contour in the prompt.
          referenced_artifact_ids: [],
        },
      );
      if (stream.outcome === 'completed') setNotice('Done.');
      else if (stream.outcome === 'failed') setNotice('');
      else setNotice('');
    } catch (caught) {
      if (!isAbortError(caught)) {
        setNotice('');
        setQuestion(text);  // keep the text so the message is not lost
        setError(caught instanceof Error ? caught.message : String(caught));
      }
    } finally {
      // A superseded request must never clear a newer request's controller or
      // unlock its UI state.
      if (
        sendController.current === controller
        && requestController.current === controller
      ) {
        sendController.current = null;
        requestController.current = null;
        setBusy('idle');
      }
    }
  };

  const startEdit = (mask: ViewerMask) => {
    if (busy !== 'idle') return;
    editTarget.current = mask;
    editFileRef.current?.click();
  };

  // On-demand edit: the user corrects a contour (e.g. drawn/exported from a
  // viewer) and uploads it. It is registered as a new mask on the same grid —
  // no run pause, no approval gate.
  const uploadEdit = async (file: File) => {
    const target = editTarget.current;
    if (!caseRecord || !target || busy !== 'idle' || requestController.current) return;
    if (!isNifti(file)) {
      setError('The edited contour must be a .nii or .nii.gz file.');
      return;
    }
    const expectedCaseId = caseRecord.case_id;
    const controller = new AbortController();
    requestController.current = controller;
    setBusy('editing');
    setError('');
    setNotice(`Uploading edited ${target.label}…`);
    try {
      const ref = await uploadEditedMask(
        expectedCaseId,
        target.label,
        file,
        target.artifact_id,
        targetTypeOf(target),
        controller.signal,
      );
      if (activeCaseId.current !== expectedCaseId) return;
      addArtifact(ref);
      setNotice(`Edited ${target.label} added to the viewer.`);
    } catch (caught) {
      if (!isAbortError(caught)) {
        setNotice('');
        setError(caught instanceof Error ? caught.message : String(caught));
      }
    } finally {
      if (requestController.current === controller) {
        requestController.current = null;
        editTarget.current = null;
        setBusy('idle');
      }
    }
  };

  const onQuestionKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    // Enter sends; Shift+Enter inserts a newline.
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      void send();
    }
  };

  const toggleMask = (artifactId: string) => {
    setMasks(items => items.map(item => (
      item.artifact_id === artifactId ? { ...item, visible: !item.visible } : item
    )));
  };

  const setAllMasks = (visible: boolean) => {
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

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-mark"><BrainCircuit size={22} aria-hidden="true" /></div>
        <div className="brand-copy">
          <h1>SegAgent</h1>
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
            disabled={locked}
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
            disabled={!caseRecord || locked}
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
              <span className="ready-dot" />
              <strong>{locked ? 'Working' : !viewerReady ? 'Loading scan' : 'Ready'}</strong>
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
                <div className="mask-row" key={mask.artifact_id}>
                  <button
                    type="button"
                    className="mask-item"
                    onClick={() => toggleMask(mask.artifact_id)}
                    aria-pressed={mask.visible}
                    title={`${mask.visible ? 'Hide' : 'Show'} ${mask.label}`}
                  >
                    <i style={{ backgroundColor: mask.visible ? mask.swatch : '#64748b' }} />
                    <span>
                      <strong>{mask.label}</strong>
                      <small>{mask.kind === 'contour' ? 'Uploaded' : 'Generated'}</small>
                    </span>
                    {mask.visible ? <Eye size={15} aria-hidden="true" /> : <EyeOff size={15} aria-hidden="true" />}
                  </button>
                  {isTargetMask(mask) && (
                    <span className="mask-badge" title="Tumor target — not validated; a clinician must review before clinical use">Review</span>
                  )}
                  <button
                    type="button"
                    className="mask-edit"
                    onClick={() => startEdit(mask)}
                    disabled={locked}
                    title={`Upload an edited version of ${mask.label} (.nii/.nii.gz)`}
                    aria-label={`Edit ${mask.label}`}
                  >
                    <Pencil size={14} aria-hidden="true" />
                  </button>
                </div>
              ))}
            </div>
            <input
              ref={editFileRef}
              type="file"
              accept=".nii,.nii.gz"
              className="visually-hidden"
              onChange={event => {
                const file = event.target.files?.[0];
                event.target.value = '';
                if (file) void uploadEdit(file);
              }}
            />
          </section>
        )}
      </aside>

      <main className="viewer-panel" aria-label="Medical image viewer">
        <MedicalViewer
          key={caseRecord?.case_id || 'empty'}
          imageUrl={caseRecord ? caseImageUrl(caseRecord.case_id) : null}
          masks={masks}
          onReadyChange={setViewerReady}
        />
      </main>

      <aside className="agent-panel" aria-labelledby="agent-heading">
        <h2 className="panel-heading" id="agent-heading">
          <MessageSquareText size={18} aria-hidden="true" /> SegAgent
          {busy === 'running' && <span className="pulse-dot" aria-label="Working" />}
        </h2>

        <div className="chat-scroll">
          <ChatThread messages={history} events={events} busy={busy === 'running'} />
        </div>

        {error && (
          <div className="error-banner" role="alert">
            <AlertTriangle size={17} aria-hidden="true" />
            <span>{error}</span>
            <button type="button" onClick={() => setError('')} aria-label="Dismiss error"><X size={15} /></button>
          </div>
        )}

        <div className="composer">
          <label className="visually-hidden" htmlFor="agent-question">Message SegAgent</label>
          <textarea
            id="agent-question"
            rows={2}
            value={question}
            onChange={event => setQuestion(event.target.value)}
            onKeyDown={onQuestionKeyDown}
            disabled={!caseRecord || !viewerReady || locked}
            placeholder={
              !caseRecord
                ? 'Upload a scan first'
                : !viewerReady
                  ? 'Wait for the scan to load'
                  : awaitingReply
                    ? 'Answer the question above…'
                    : 'Ask SegAgent to contour something, or ask about the scan'
            }
          />
          <div className="composer-footer">
            <span>{awaitingReply ? 'Replying to SegAgent' : 'Enter to send · Shift+Enter for a new line'}</span>
            <button
              type="button"
              className="primary run-button"
              disabled={!caseRecord || !viewerReady || !question.trim() || locked}
              onClick={send}
              aria-label="Send message"
            >
              {busy === 'running' ? <Loader2 className="spin" size={17} /> : <SendHorizonal size={17} />}
            </button>
          </div>
        </div>
      </aside>
    </div>
  );
}
