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
  question: 'Question',
};

interface Step {
  key: string;
  title: string;
  detail?: string;
  thinking?: string;
  icon: LucideIcon;
}

type Block =
  | { kind: 'user'; key: string; text: string }
  | { kind: 'intent'; key: string; intent: TaskIntent }
  | { kind: 'steps'; key: string; steps: Step[] }
  | { kind: 'question'; key: string; text: string }
  | { kind: 'answer'; key: string; text: string }
  | { kind: 'error'; key: string; text: string };

interface Turn {
  runId: string;
  blocks: Block[];
  done: boolean;
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
        addStep(turn, {
          key,
          title: ACTIONS[String(payload.action || '')] || 'Planning',
          detail: payload.rationale_summary,
          icon: GitBranch,
        });
        break;
      case 'thinking':
        addStep(turn, {
          key,
          title: 'Thinking',
          thinking: String(payload.thinking || payload.text || ''),
          icon: Brain,
        });
        break;
      case 'tool_started':
        addStep(turn, {
          key,
          title: ACTIONS[String(payload.tool || '')] || 'Working',
          detail: payload.structures?.join(', ') || payload.query,
          icon: Wrench,
        });
        break;
      case 'observation':
        addStep(turn, { key, title: 'Result', detail: payload.observation?.summary, icon: Database });
        break;
      case 'artifact':
        addStep(turn, { key, title: 'Contour ready', detail: payload.artifact?.label, icon: ScanLine });
        break;
      case 'question_pending':
        turn.blocks.push({ kind: 'question', key, text: String(payload.question || '') });
        turn.done = true;
        break;
      case 'reply_recorded':
        turn.blocks.push({ kind: 'user', key, text: String(payload.text || '') });
        turn.done = false;
        break;
      case 'answer':
        turn.blocks.push({ kind: 'answer', key, text: String(payload.text || '') });
        break;
      case 'error':
        turn.blocks.push({
          kind: 'error',
          key,
          text: String(payload.message || 'Something went wrong.'),
        });
        turn.done = true;
        break;
      case 'run_completed':
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

function StepList({ steps, active }: { steps: Step[]; active: boolean }) {
  if (!steps.length) return null;
  return (
    <ol className="step-list">
      {steps.map((step, index) => {
        const running = active && index === steps.length - 1;
        const Icon = running ? Loader2 : step.icon;
        return (
          <li key={step.key} className={running ? 'step running' : 'step'}>
            <Icon size={14} className={running ? 'spin' : undefined} aria-hidden="true" />
            <div className="step-body">
              <span className="step-title">{step.title}</span>
              {step.thinking ? (
                <details className="step-thinking">
                  <summary>Show reasoning</summary>
                  <p>{step.thinking}</p>
                </details>
              ) : (
                step.detail && <span className="step-detail">{step.detail}</span>
              )}
            </div>
          </li>
        );
      })}
    </ol>
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

  // Prior turns restored from the server have no events; render them as plain
  // bubbles, skipping any run already represented in this session's stream.
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
        const lastBlock = turn.blocks[turn.blocks.length - 1];
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
                    <StepList steps={block.steps} active={active && block === lastBlock} />
                  </div>
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
                  <div className="agent-answer" key={block.key}>
                    <CheckCircle2 size={14} aria-hidden="true" />
                    <p>{block.text}</p>
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
            {active && lastBlock?.kind !== 'steps' && (
              <div className="agent-pending"><Loader2 className="spin" size={14} /> Thinking…</div>
            )}
          </article>
        );
      })}
      <div ref={endRef} />
    </div>
  );
}
