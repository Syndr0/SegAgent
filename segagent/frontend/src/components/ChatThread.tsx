import { useEffect, useMemo, useRef } from 'react';
import {
  AlertTriangle,
  Brain,
  CheckCircle2,
  Database,
  GitBranch,
  HelpCircle,
  Loader2,
  ScanLine,
  Sparkles,
  Wrench,
  type LucideIcon,
} from 'lucide-react';
import type { ChatMessage, RunEvent, TaskIntent } from '../types';

const ACTIONS: Record<string, string> = {
  lookup_protocol: 'Looking up the protocol',
  segment: 'Contouring',
  inspect_mask: 'Measuring existing contours',
  run_qc: 'Checking contours',
  answer_question: 'Looking at the image',
  ask_user: 'Asking a question',
  final: 'Writing the answer',
};

const INTENT_LABELS: Record<string, string> = {
  organ: 'Organ',
  oar: 'Organs at risk',
  gtv: 'Tumor target',
  qc: 'Contour check',
  report: 'Report',
  protocol: 'Protocol',
  vqa: 'Image question',
  mask_question: 'Mask question',
  ambiguous: 'Needs clarification',
  missing_input: 'Missing input',
  question: 'Question',
};

interface Step {
  key: string;
  phase?: string;
  title: string;
  detail?: string;
  icon: LucideIcon;
  status: 'running' | 'completed' | 'failed' | 'info';
}

type Block =
  | { kind: 'user'; key: string; text: string }
  | { kind: 'intent'; key: string; intent: TaskIntent }
  | { kind: 'steps'; key: string; steps: Step[] }
  | {
      kind: 'thinking';
      key: string;
      streamKey: string;
      channel: string;
      text: string;
      complete: boolean;
    }
  | { kind: 'question'; key: string; text: string }
  | { kind: 'answer'; key: string; text: string; complete: boolean }
  | { kind: 'error'; key: string; text: string };

interface Turn {
  runId: string;
  blocks: Block[];
  done: boolean;
}

function displayAction(value: unknown, fallback = 'Working'): string {
  const action = typeof value === 'string' ? value : '';
  if (ACTIONS[action]) return ACTIONS[action];
  if (!action) return fallback;
  return action
    .replaceAll('_', ' ')
    .replace(/\b\w/g, character => character.toUpperCase());
}

function phaseIcon(phase: string): LucideIcon {
  if (phase === 'planner' || phase === 'answer_question') return Brain;
  if (phase === 'lookup_protocol' || phase === 'inspect_mask' || phase === 'run_qc') {
    return Database;
  }
  if (phase === 'segment') return ScanLine;
  return Wrench;
}

function detailFrom(payload: RunEvent['payload']): string | undefined {
  if (typeof payload.detail === 'string' && payload.detail) return payload.detail;
  if (typeof payload.summary === 'string' && payload.summary) return payload.summary;
  if (typeof payload.error === 'string' && payload.error) return payload.error;
  if (typeof payload.message === 'string' && payload.message) return payload.message;
  if (Array.isArray(payload.structures) && payload.structures.length) {
    return payload.structures.join(', ');
  }
  if (typeof payload.query === 'string' && payload.query) return payload.query;
  return undefined;
}

/** Group the flat event stream into conversational turns of ordered blocks, so
 *  a mid-turn question and the user's reply interleave in the right order. */
function buildTurns(events: RunEvent[]): Turn[] {
  const order: string[] = [];
  const byRun = new Map<string, Turn>();

  const turnFor = (runId: string): Turn => {
    let turn = byRun.get(runId);
    if (!turn) {
      turn = { runId, blocks: [], done: false };
      byRun.set(runId, turn);
      order.push(runId);
    }
    return turn;
  };

  const addStep = (turn: Turn, step: Step) => {
    const last = turn.blocks[turn.blocks.length - 1];
    if (last?.kind === 'steps') last.steps.push(step);
    else turn.blocks.push({ kind: 'steps', key: step.key, steps: [step] });
  };

  const allSteps = (turn: Turn) =>
    turn.blocks.flatMap(block => block.kind === 'steps' ? block.steps : []);

  const findPhase = (turn: Turn, phase: string, runningOnly = false) =>
    [...allSteps(turn)]
      .reverse()
      .find(step => step.phase === phase && (!runningOnly || step.status === 'running'));

  const updatePhase = (
    turn: Turn,
    event: RunEvent,
    status: Step['status'],
  ) => {
    const { payload } = event;
    const phase = String(payload.phase || payload.tool || '');
    const existing = !phase
      ? undefined
      : status === 'running'
        ? findPhase(turn, phase, true)
        : findPhase(turn, phase, true) || findPhase(turn, phase);
    const detail = detailFrom(payload);
    const title = typeof payload.label === 'string' && payload.label
      ? payload.label
      : displayAction(phase);
    if (existing) {
      existing.status = status;
      if (typeof payload.label === 'string' && payload.label) existing.title = title;
      if (detail) existing.detail = detail;
      return;
    }
    addStep(turn, {
      key: event.event_id,
      phase: phase || undefined,
      title,
      detail,
      icon: phaseIcon(phase),
      status,
    });
  };

  const updateDecision = (turn: Turn, event: RunEvent, summary: string | undefined) => {
    const { payload } = event;
    const stepNumber = typeof payload.step === 'number' ? payload.step : undefined;
    const phase = stepNumber === undefined ? `decision:${event.event_id}` : `decision:${stepNumber}`;
    const existing = findPhase(turn, phase);
    const title = displayAction(payload.action, 'Reasoning summary');
    if (existing) {
      existing.title = title;
      if (summary) existing.detail = summary;
      existing.status = 'completed';
      return;
    }
    addStep(turn, {
      key: event.event_id,
      phase,
      title,
      detail: summary,
      icon: GitBranch,
      status: 'completed',
    });
  };

  const thinkingFor = (turn: Turn, streamKey: string) =>
    turn.blocks.find(
      (block): block is Extract<Block, { kind: 'thinking' }> =>
        block.kind === 'thinking' && block.streamKey === streamKey,
    );

  const answerFor = (turn: Turn) =>
    turn.blocks.find(
      (block): block is Extract<Block, { kind: 'answer' }> => block.kind === 'answer',
    );

  const finishOpenBlocks = (turn: Turn, failed = false) => {
    for (const block of turn.blocks) {
      if (block.kind === 'thinking' || block.kind === 'answer') block.complete = true;
      if (block.kind === 'steps') {
        for (const step of block.steps) {
          if (step.status === 'running') step.status = failed ? 'failed' : 'completed';
        }
      }
    }
  };

  for (const event of events) {
    const turn = turnFor(event.run_id);
    const { payload } = event;
    const key = event.event_id;
    switch (event.type) {
      case 'run_started':
        if (payload.question) {
          turn.blocks.push({ kind: 'user', key, text: String(payload.question) });
        }
        if (payload.intent) {
          turn.blocks.push({ kind: 'intent', key: `${key}-intent`, intent: payload.intent });
        }
        break;
      case 'planner_decision':
        updateDecision(
          turn,
          event,
          typeof payload.rationale_summary === 'string' ? payload.rationale_summary : undefined,
        );
        break;
      case 'thinking_delta': {
        const channel = typeof payload.channel === 'string' ? payload.channel : 'planner';
        const step = typeof payload.step === 'number' ? payload.step : 'current';
        const streamKey = `${channel}:${step}`;
        const delta = String(payload.delta || payload.text || payload.thinking || '');
        const existing = thinkingFor(turn, streamKey);
        if (existing) {
          existing.text += delta;
          existing.complete = false;
        } else {
          turn.blocks.push({
            kind: 'thinking',
            key,
            streamKey,
            channel,
            text: delta,
            complete: false,
          });
        }
        break;
      }
      case 'thinking': {
        const channel = typeof payload.channel === 'string' ? payload.channel : 'planner';
        const step = typeof payload.step === 'number' ? payload.step : 'current';
        const streamKey = `${channel}:${step}`;
        const text = String(payload.thinking || payload.text || '');
        const existing = thinkingFor(turn, streamKey);
        if (existing) {
          // The terminal event is the authoritative full transcript. Replacing
          // streamed deltas prevents the final snapshot from appearing twice.
          if (text) existing.text = text;
          existing.complete = payload.complete !== false;
        } else {
          turn.blocks.push({
            kind: 'thinking',
            key,
            streamKey,
            channel,
            text,
            complete: payload.complete !== false,
          });
        }
        break;
      }
      case 'reasoning_summary':
        updateDecision(
          turn,
          event,
          typeof payload.text === 'string'
            ? payload.text
            : typeof payload.rationale_summary === 'string'
              ? payload.rationale_summary
              : undefined,
        );
        break;
      case 'phase_started':
        updatePhase(turn, event, 'running');
        break;
      case 'phase_completed':
        updatePhase(turn, event, 'completed');
        break;
      case 'phase_failed':
        updatePhase(turn, event, 'failed');
        break;
      case 'tool_started':
        updatePhase(turn, event, 'running');
        break;
      case 'observation': {
        const phase = payload.observation?.tool || '';
        const existing = phase ? findPhase(turn, phase, true) || findPhase(turn, phase) : undefined;
        if (existing) {
          existing.status = 'completed';
          if (payload.observation?.summary) existing.detail = payload.observation.summary;
        } else {
          addStep(turn, {
            key,
            phase: phase || undefined,
            title: 'Result',
            detail: payload.observation?.summary,
            icon: Database,
            status: 'completed',
          });
        }
        break;
      }
      case 'artifact':
        addStep(turn, {
          key,
          title: 'Contour ready',
          detail: payload.artifact?.label,
          icon: ScanLine,
          status: 'completed',
        });
        break;
      case 'question_pending':
        finishOpenBlocks(turn);
        turn.blocks.push({ kind: 'question', key, text: String(payload.question || '') });
        turn.done = true;
        break;
      case 'reply_recorded':
        turn.blocks.push({ kind: 'user', key, text: String(payload.text || '') });
        turn.done = false;
        break;
      case 'answer_delta': {
        const delta = String(payload.delta || payload.text || '');
        const existing = answerFor(turn);
        if (existing) {
          existing.text += delta;
          existing.complete = false;
        } else {
          turn.blocks.push({ kind: 'answer', key, text: delta, complete: false });
        }
        break;
      }
      case 'answer': {
        const text = String(payload.text || '');
        const existing = answerFor(turn);
        if (existing) {
          if (text) existing.text = text;
          existing.complete = true;
        } else {
          turn.blocks.push({ kind: 'answer', key, text, complete: true });
        }
        break;
      }
      case 'error':
        finishOpenBlocks(turn, true);
        turn.blocks.push({
          kind: 'error',
          key,
          text: String(payload.message || 'Something went wrong.'),
        });
        turn.done = true;
        break;
      case 'run_completed':
        finishOpenBlocks(turn);
        turn.done = true;
        break;
      default:
        break;
    }
  }
  return order.map(runId => byRun.get(runId)!);
}

function IntentChip({ intent }: { intent: TaskIntent }) {
  const label = INTENT_LABELS[intent.intent] || intent.intent;
  return (
    <span className={`intent-chip intent-${intent.intent}`} title={intent.rationale || undefined}>
      {label}
      {intent.site ? ` · ${intent.site}` : ''}
    </span>
  );
}

function ThinkingPanel({
  text,
  channel,
  complete,
}: {
  text: string;
  channel: string;
  complete: boolean;
}) {
  const channelLabel = channel === 'vqa' ? 'Image reasoning' : 'Planner reasoning';
  const textRef = useRef<HTMLParagraphElement>(null);
  const followStream = useRef(true);

  useEffect(() => {
    const element = textRef.current;
    if (element && followStream.current) element.scrollTop = element.scrollHeight;
  }, [text]);

  return (
    <section
      className={complete ? 'thinking-panel complete' : 'thinking-panel streaming'}
      aria-label="Model reasoning, not clinical evidence"
    >
      <header className="thinking-header">
        <Brain size={15} aria-hidden="true" />
        <div>
          <strong>Model reasoning</strong>
          <span>{channelLabel} · not clinical evidence</span>
        </div>
        <span className="thinking-state">
          {!complete && <span className="thinking-live-dot" aria-hidden="true" />}
          {complete ? 'Complete' : 'Live'}
        </span>
      </header>
      <p
        className="thinking-text"
        ref={textRef}
        onScroll={event => {
          const element = event.currentTarget;
          followStream.current =
            element.scrollHeight - element.scrollTop - element.clientHeight < 24;
        }}
      >
        {text || (!complete ? 'Waiting for the first reasoning token…' : 'No reasoning text returned.')}
        {!complete && text && <span className="thinking-cursor" aria-hidden="true" />}
      </p>
    </section>
  );
}

function StepList({ steps }: { steps: Step[] }) {
  if (!steps.length) return null;
  return (
    <section className="progress-panel" aria-label="Structured progress">
      <h3>Progress</h3>
      <ol className="step-list">
        {steps.map(step => {
          const running = step.status === 'running';
          const failed = step.status === 'failed';
          const Icon = running ? Loader2 : failed ? AlertTriangle : step.icon;
          return (
            <li key={step.key} className={`step ${step.status}`}>
              <Icon size={14} className={running ? 'spin' : undefined} aria-hidden="true" />
              <div className="step-body">
                <span className="step-title">{step.title}</span>
                {step.detail && <span className="step-detail">{step.detail}</span>}
              </div>
            </li>
          );
        })}
      </ol>
    </section>
  );
}

interface Props {
  messages: ChatMessage[];
  events: RunEvent[];
  busy: boolean;
}

export default function ChatThread({ messages, events, busy }: Props) {
  const endRef = useRef<HTMLDivElement>(null);
  const turns = useMemo(() => buildTurns(events), [events]);

  // A run with a restored event trace renders its full reasoning/progress.
  // Messages without an available trace still fall back to ordinary bubbles.
  const liveRuns = useMemo(() => new Set(turns.map(turn => turn.runId)), [turns]);
  const history = useMemo(
    () => messages.filter(item => !item.run_id || !liveRuns.has(item.run_id)),
    [messages, liveRuns],
  );

  useEffect(() => {
    const scroll = endRef.current?.closest<HTMLElement>('.chat-scroll');
    if (!scroll) return;
    const distance = scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight;
    if (distance < 220) scroll.scrollTo({ top: scroll.scrollHeight, behavior: 'smooth' });
  }, [events.length, messages.length]);

  if (!history.length && !turns.length) {
    return (
      <div className="chat-empty">
        <Sparkles size={22} aria-hidden="true" />
        <p>Ask about this scan — contour a structure, or ask a question about it.</p>
      </div>
    );
  }

  return (
    <div className="chat-thread" role="log" aria-live="polite" aria-label="Conversation">
      {history.map(item => (
        <div className={`bubble bubble-${item.role}`} key={item.message_id}>
          {item.text}
        </div>
      ))}

      {turns.map((turn, index) => {
        const active = busy && index === turns.length - 1 && !turn.done;
        return (
          <article className="turn" key={turn.runId}>
            {turn.blocks.map(block => {
              if (block.kind === 'user') {
                return <div className="bubble bubble-user" key={block.key}>{block.text}</div>;
              }
              if (block.kind === 'intent') {
                return <IntentChip intent={block.intent} key={block.key} />;
              }
              if (block.kind === 'steps') {
                return (
                  <div className="agent-turn" key={block.key}>
                    <StepList steps={block.steps} />
                  </div>
                );
              }
              if (block.kind === 'thinking') {
                return (
                  <ThinkingPanel
                    key={block.key}
                    text={block.text}
                    channel={block.channel}
                    complete={block.complete}
                  />
                );
              }
              if (block.kind === 'question') {
                return (
                  <div className="agent-question" key={block.key}>
                    <HelpCircle size={15} aria-hidden="true" />
                    <p>{block.text}</p>
                  </div>
                );
              }
              if (block.kind === 'answer') {
                return (
                  <div className={`agent-answer${block.complete ? '' : ' streaming'}`} key={block.key}>
                    {block.complete
                      ? <CheckCircle2 size={14} aria-hidden="true" />
                      : <Loader2 className="spin" size={14} aria-hidden="true" />}
                    <div>
                      <strong>Final answer</strong>
                      <p>{block.text}</p>
                    </div>
                  </div>
                );
              }
              return (
                <div className="agent-error" key={block.key}>
                  <AlertTriangle size={14} aria-hidden="true" />
                  <p>{block.text}</p>
                </div>
              );
            })}
            {active
              && !turn.blocks.some(block =>
                (block.kind === 'thinking' && !block.complete)
                || (block.kind === 'steps' && block.steps.some(step => step.status === 'running'))
                || (block.kind === 'answer' && !block.complete)
              )
              && (
              <div className="agent-pending">
                <Loader2 className="spin" size={14} /> Waiting for the next event…
              </div>
            )}
          </article>
        );
      })}
      <div ref={endRef} />
    </div>
  );
}
