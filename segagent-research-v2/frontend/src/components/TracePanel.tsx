import { useEffect, useRef } from 'react';
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Database,
  GitBranch,
  ScanLine,
  ShieldCheck,
  Wrench,
  type LucideIcon,
} from 'lucide-react';
import type { RunEvent } from '../types';

const ACTIONS: Record<string, string> = {
  lookup_protocol: 'Find protocol',
  segment: 'Create masks',
  run_qc: 'Check contours',
  ask_user: 'Need more information',
  final: 'Prepare result',
};

const DECISIONS: Record<string, string> = {
  approve: 'Approved',
  feedback: 'Changes requested',
  reject: 'Rejected',
};

interface Item {
  title: string;
  body?: string;
  icon: LucideIcon;
}

function content(event: RunEvent): Item {
  switch (event.type) {
    case 'run_started':
      return { title: 'Run started', body: 'The agent is planning the next step.', icon: Activity };
    case 'planner_decision': {
      const confidence = typeof event.payload.confidence === 'number'
        ? ` · ${Math.round(event.payload.confidence * 100)}% confidence`
        : '';
      return {
        title: ACTIONS[event.payload.action || ''] || 'Plan',
        body: `${event.payload.rationale_summary || 'Next step selected.'}${confidence}`,
        icon: GitBranch,
      };
    }
    case 'tool_started': {
      const structures = event.payload.structures?.join(', ');
      return {
        title: ACTIONS[event.payload.tool || ''] || 'Working',
        body: structures || event.payload.query || 'This tool step was used.',
        icon: Wrench,
      };
    }
    case 'observation':
      return {
        title: 'Step complete',
        body: event.payload.observation?.summary || 'The result was saved.',
        icon: Database,
      };
    case 'artifact':
      return {
        title: 'Mask ready',
        body: event.payload.artifact?.label || 'A new result is ready in the viewer.',
        icon: ScanLine,
      };
    case 'approval_required':
      return {
        title: 'Review needed',
        body: 'Check the new masks before the run continues.',
        icon: ShieldCheck,
      };
    case 'approval_recorded': {
      const decision = event.payload.approval?.decision || '';
      return {
        title: 'Review saved',
        body: DECISIONS[decision] || 'Your decision was recorded.',
        icon: CheckCircle2,
      };
    }
    case 'run_completed':
      return { title: 'Done', body: 'The run is complete.', icon: CheckCircle2 };
    case 'error':
      return { title: 'Run failed', body: event.payload.message || 'Something went wrong.', icon: AlertTriangle };
    default:
      return { title: 'Activity', icon: Activity };
  }
}

export default function TracePanel({ events }: { events: RunEvent[] }) {
  const endRef = useRef<HTMLDivElement>(null);
  const visible = events.filter(event => event.type !== 'answer');

  useEffect(() => {
    const scroll = endRef.current?.closest<HTMLElement>('.trace-scroll');
    if (!scroll) return;
    const distanceFromBottom = scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight;
    if (distanceFromBottom < 160) scroll.scrollTo({ top: scroll.scrollHeight });
  }, [visible.length]);

  if (!visible.length) {
    return <div className="trace-empty">Run steps will appear here.</div>;
  }

  return (
    <div className="trace-list" role="log" aria-live="polite" aria-label="Agent activity">
      {visible.map(event => {
        const item = content(event);
        const Icon = item.icon;
        return (
          <article className={`trace-item trace-${event.type}`} key={event.event_id}>
            <span className="trace-icon"><Icon size={16} aria-hidden="true" /></span>
            <div>
              <div className="trace-title"><strong>{item.title}</strong><small>#{event.sequence}</small></div>
              {item.body && <p>{item.body}</p>}
              <details>
                <summary>Details</summary>
                <pre>{JSON.stringify(event.payload, null, 2)}</pre>
              </details>
            </div>
          </article>
        );
      })}
      <div ref={endRef} />
    </div>
  );
}
