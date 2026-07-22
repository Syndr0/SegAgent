import { useMemo, useRef, useState } from 'react';
import {
  AlertTriangle,
  BrainCircuit,
  Check,
  FlaskConical,
  Loader2,
  MessageSquareText,
  ScanLine,
  ShieldCheck,
  Upload,
  X,
} from 'lucide-react';
import {
  approvalFromEvent,
  caseImageUrl,
  createCase,
  resumeRun,
  startRun,
} from './api';
import MedicalViewer from './components/MedicalViewer';
import TracePanel from './components/TracePanel';
import type { ApprovalRequest, ArtifactRef, CaseRecord, RunEvent, ViewerMask } from './types';

const COLORS = ['red', 'green', 'blue', 'warm', 'cool', 'violet', 'winter'];

export default function App() {
  const [caseRecord, setCaseRecord] = useState<CaseRecord | null>(null);
  const [image, setImage] = useState<File | null>(null);
  const [contours, setContours] = useState<File[]>([]);
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [masks, setMasks] = useState<ViewerMask[]>([]);
  const [question, setQuestion] = useState('');
  const [runId, setRunId] = useState('');
  const [approval, setApproval] = useState<ApprovalRequest | null>(null);
  const [reviewFeedback, setReviewFeedback] = useState('');
  const [uploading, setUploading] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState('');
  const contourRef = useRef<HTMLInputElement>(null);

  const answer = useMemo(
    () => [...events].reverse().find(event => event.type === 'answer')?.payload.text as string | undefined,
    [events],
  );

  const addArtifact = (artifact: ArtifactRef) => {
    if (!['mask', 'contour'].includes(artifact.kind)) return;
    setMasks(previous => {
      if (previous.some(item => item.artifact_id === artifact.artifact_id)) return previous;
      return [
        ...previous,
        { ...artifact, visible: true, color: COLORS[previous.length % COLORS.length] },
      ];
    });
  };

  const acceptEvent = (event: RunEvent) => {
    setEvents(previous => [...previous, event]);
    if (event.type === 'artifact' && event.payload.artifact) addArtifact(event.payload.artifact);
    if (event.type === 'approval_required') setApproval(approvalFromEvent(event));
    if (event.type === 'error') setError(event.payload.message || 'Agent run failed.');
  };

  const uploadCase = async () => {
    if (!image) return;
    setUploading(true);
    setError('');
    try {
      const created = await createCase(image, contours);
      setCaseRecord(created);
      setEvents([]);
      setRunId('');
      setApproval(null);
      setQuestion('');
      setMasks(
        created.contours.map((artifact, index) => ({
          ...artifact,
          visible: true,
          color: COLORS[index % COLORS.length],
        })),
      );
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setUploading(false);
    }
  };

  const run = async () => {
    if (!caseRecord || !question.trim() || running) return;
    setRunning(true);
    setEvents([]);
    setApproval(null);
    setError('');
    try {
      const id = await startRun(caseRecord.case_id, question.trim(), acceptEvent);
      setRunId(id);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setRunning(false);
    }
  };

  const review = async (decision: 'approve' | 'reject' | 'feedback') => {
    if (!runId || running) return;
    setRunning(true);
    setError('');
    try {
      await resumeRun(runId, decision, reviewFeedback, acceptEvent);
      setApproval(null);
      setReviewFeedback('');
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-mark"><BrainCircuit size={22} /></div>
        <div>
          <h1>SegAgent <span>Research v2</span></h1>
          <p>Stateful 3D segmentation agent · evidence before answer</p>
        </div>
        <div className="research-badge"><FlaskConical size={14} /> Research only</div>
      </header>

      <aside className="case-panel">
        <div className="panel-heading"><ScanLine size={17} /><span>Case workspace</span></div>
        <label className="file-drop">
          <Upload size={22} />
          <strong>{image?.name || 'Select a NIfTI image'}</strong>
          <span>.nii or .nii.gz</span>
          <input type="file" accept=".nii,.nii.gz,.gz" onChange={event => setImage(event.target.files?.[0] || null)} />
        </label>
        <label className="secondary-upload">
          <ShieldCheck size={16} />
          <span>{contours.length ? `${contours.length} contour files` : 'Optional QC contours'}</span>
          <input
            ref={contourRef}
            type="file"
            accept=".nii,.nii.gz,.gz"
            multiple
            onChange={event => setContours(Array.from(event.target.files || []))}
          />
        </label>
        <button className="primary" disabled={!image || uploading} onClick={uploadCase}>
          {uploading ? <Loader2 className="spin" size={16} /> : <ScanLine size={16} />}
          {uploading ? 'Creating case…' : 'Create isolated case'}
        </button>

        {caseRecord && (
          <div className="case-card">
            <div><small>CASE ID</small><code>{caseRecord.case_id}</code></div>
            <div><small>IMAGE</small><span>{caseRecord.source_name}</span></div>
            <div><small>CONTOURS</small><span>{caseRecord.contours.length}</span></div>
          </div>
        )}

        {masks.length > 0 && (
          <div className="mask-list">
            <small>CASE-BOUND ARTIFACTS</small>
            {masks.map(mask => (
              <button
                key={mask.artifact_id}
                onClick={() => setMasks(items => items.map(item =>
                  item.artifact_id === mask.artifact_id ? { ...item, visible: !item.visible } : item,
                ))}
              >
                <i style={{ background: mask.visible ? mask.color : '#334155' }} />
                <span>{mask.label}</span>
                <em>{mask.visible ? 'visible' : 'hidden'}</em>
              </button>
            ))}
          </div>
        )}
      </aside>

      <main className="viewer-panel">
        <MedicalViewer
          imageUrl={caseRecord ? caseImageUrl(caseRecord.case_id) : null}
          masks={masks}
        />
      </main>

      <aside className="agent-panel">
        <div className="panel-heading"><MessageSquareText size={17} /><span>Agent experiment</span></div>
        <div className="composer">
          <textarea
            value={question}
            onChange={event => setQuestion(event.target.value)}
            disabled={!caseRecord || running || Boolean(approval)}
            placeholder="Segment both kidneys, retrieve pelvic OARs, or audit the uploaded contours…"
          />
          <button className="primary" disabled={!caseRecord || !question.trim() || running || Boolean(approval)} onClick={run}>
            {running ? <Loader2 className="spin" size={16} /> : <BrainCircuit size={16} />}
            Run typed workflow
          </button>
        </div>

        {approval && (
          <section className="approval-card">
            <div className="approval-title"><ShieldCheck size={18} /><strong>Human review required</strong></div>
            <p>{approval.message}</p>
            <textarea
              value={reviewFeedback}
              onChange={event => setReviewFeedback(event.target.value)}
              placeholder="Optional correction or reviewer note"
            />
            <div className="review-actions">
              <button className="approve" onClick={() => review('approve')}><Check size={15} /> Approve</button>
              <button className="feedback" onClick={() => review('feedback')}><AlertTriangle size={15} /> Feedback</button>
              <button className="reject" onClick={() => review('reject')}><X size={15} /> Reject</button>
            </div>
          </section>
        )}

        {answer && <section className="answer-card"><small>GROUNDED RESPONSE</small><p>{answer}</p></section>}
        {error && <div className="error-banner"><AlertTriangle size={15} />{error}</div>}

        <div className="trace-heading"><ActivityLabel /> Auditable trace</div>
        <div className="trace-scroll"><TracePanel events={events} /></div>
      </aside>
    </div>
  );
}

function ActivityLabel() {
  return <span className="pulse-dot" />;
}

