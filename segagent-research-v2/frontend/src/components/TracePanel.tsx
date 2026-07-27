import { useEffect, useRef } from 'react';
import {
  AlertTriangle,
  BrainCircuit,
  CheckCircle2,
  Database,
  Loader2,
  ScanLine,
  Search,
  Sparkles,
  Target,
  type LucideIcon,
} from 'lucide-react';
import type { RunEvent } from '../types';

// Events that mean the run is no longer thinking. Anything else at the tail of
// the stream keeps the "Working…" indicator alive.
const TERMINAL = new Set(['answer', 'run_completed', 'error']);

const ACTION_LABELS: Record<string, string> = {
  lookup_protocol: 'Looking up the protocol',
  segment: 'Planning the segmentation',
  ask_user: 'Needs more information',
  final: 'Preparing the result',
};

const TOOL_ICON: Record<string, LucideIcon> = {
  lookup_protocol: Search,
  segment: ScanLine,
};

const TOOL_RUNNING: Record<string, string> = {
  lookup_protocol: 'Searching the guideline library',
  segment: 'Creating masks with VoxTell',
};

const INTENT_LABELS: Record<string, string> = {
  organ: 'Organ',
  oar: 'Organs at risk',
  gtv: 'Tumor target',
  protocol: 'Protocol',
  question: 'Needs clarification',
};

interface IntentPayload {
  intent?: string;
  site?: string | null;
  targets?: string[];
  rationale?: string;
}

interface Finding {
  severity?: string;
  message?: string;
}

function Turn({ event, active }: { event: RunEvent; active: boolean }) {
  const { type, payload } = event;

  if (type === 'run_started') {
    const question = typeof payload.question === 'string' ? payload.question : '';
    const intent = payload.intent as IntentPayload | undefined;
    return (
      <div className="chat-group">
        {question && <div className="chat-user">{question}</div>}
        {intent?.intent && (
          <div className="chat-intent">
            <span className="chat-intent-tag">{INTENT_LABELS[intent.intent] || intent.intent}</span>
            {intent.site && <span className="chat-tag">{intent.site}</span>}
            {(intent.targets || []).map(target => (
              <span className="chat-tag" key={target}>{target}</span>
            ))}
            {intent.rationale && <small>{intent.rationale}</small>}
          </div>
        )}
      </div>
    );
  }

  if (type === 'planner_decision') {
    const action = typeof payload.action === 'string' ? payload.action : '';
    const rationale = typeof payload.rationale_summary === 'string' ? payload.rationale_summary : '';
    return (
      <div className={`chat-step${active ? ' chat-step-active' : ''}`}>
        <span className="chat-icon">{active ? <Loader2 className="spin" size={15} /> : <BrainCircuit size={15} />}</span>
        <div className="chat-step-body">
          <strong>{ACTION_LABELS[action] || 'Thinking'}</strong>
          {rationale && <p className="chat-think">{rationale}</p>}
        </div>
      </div>
    );
  }

  if (type === 'tool_started') {
    const tool = typeof payload.tool === 'string' ? payload.tool : '';
    const Icon = TOOL_ICON[tool] || Database;
    const structures = Array.isArray(payload.structures) ? (payload.structures as string[]).join(', ') : '';
    const query = typeof payload.query === 'string' ? payload.query : '';
    return (
      <div className={`chat-step${active ? ' chat-step-active' : ''}`}>
        <span className="chat-icon">{active ? <Loader2 className="spin" size={15} /> : <Icon size={15} />}</span>
        <div className="chat-step-body">
          <strong>{TOOL_RUNNING[tool] || 'Working'}</strong>
          {(structures || query) && <p>{structures || query}</p>}
        </div>
      </div>
    );
  }

  if (type === 'observation') {
    const observation = payload.observation as { summary?: string } | undefined;
    const summary = observation?.summary || 'Step complete.';
    return (
      <div className="chat-note">
        <span className="chat-icon-sm"><CheckCircle2 size={13} /></span>
        <p>{summary}</p>
      </div>
    );
  }

  if (type === 'artifact') {
    const artifact = payload.artifact as { label?: string } | undefined;
    return (
      <div className="chat-artifact">
        <Target size={13} aria-hidden="true" />
        <span>Mask ready · {artifact?.label || 'result'}</span>
      </div>
    );
  }

  if (type === 'answer') {
    const text = typeof payload.text === 'string' ? payload.text : '';
    const findings = (payload.critic_findings as Finding[] | undefined) || [];
    return (
      <div className="chat-answer">
        <div className="chat-answer-head"><Sparkles size={14} aria-hidden="true" /> Result</div>
        <p>{text}</p>
        {findings.length > 0 && (
          <ul className="chat-findings">
            {findings.map((finding, index) => (
              <li key={index} className={finding.severity === 'error' ? 'is-error' : 'is-warn'}>
                {finding.message}
              </li>
            ))}
          </ul>
        )}
      </div>
    );
  }

  if (type === 'error') {
    return (
      <div className="chat-error">
        <AlertTriangle size={14} aria-hidden="true" />
        <span>{typeof payload.message === 'string' ? payload.message : 'The run failed.'}</span>
      </div>
    );
  }

  return null;
}

export default function TracePanel({ events }: { events: RunEvent[] }) {
  const endRef = useRef<HTMLDivElement>(null);
  // run_completed carries no content; it only flips the "working" state off.
  const visible = events.filter(event => event.type !== 'run_completed');

  useEffect(() => {
    const scroll = endRef.current?.closest<HTMLElement>('.trace-scroll');
    if (!scroll) return;
    const distanceFromBottom = scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight;
    if (distanceFromBottom < 200) scroll.scrollTo({ top: scroll.scrollHeight });
  }, [visible.length]);

  if (!visible.length) {
    return <div className="chat-empty">The agent's reasoning will appear here.</div>;
  }

  const last = events[events.length - 1];
  const runActive = Boolean(last) && !TERMINAL.has(last.type);

  return (
    <div className="chat-thread" role="log" aria-live="polite" aria-label="Agent conversation">
      {visible.map(event => (
        <Turn key={event.event_id} event={event} active={runActive && event.event_id === last.event_id} />
      ))}
      {runActive && (
        <div className="chat-typing"><Loader2 className="spin" size={14} aria-hidden="true" /> Working…</div>
      )}
      <div ref={endRef} />
    </div>
  );
}
