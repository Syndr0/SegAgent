import { Activity, AlertTriangle, CheckCircle2, Database, GitBranch, Wrench } from 'lucide-react';
import type { RunEvent } from '../types';

function content(event: RunEvent): { title: string; body: string; icon: typeof Activity } {
  switch (event.type) {
    case 'planner_decision':
      return {
        title: `Planner · ${event.payload.action}`,
        body: `${event.payload.rationale_summary} (confidence ${event.payload.confidence})`,
        icon: GitBranch,
      };
    case 'tool_started':
      return { title: `Tool · ${event.payload.tool}`, body: JSON.stringify(event.payload), icon: Wrench };
    case 'observation':
      return {
        title: `Evidence · ${event.payload.observation?.tool || 'tool'}`,
        body: event.payload.observation?.summary || 'Observation recorded.',
        icon: Database,
      };
    case 'approval_recorded':
      return {
        title: 'Human review',
        body: `Decision: ${event.payload.approval?.decision}`,
        icon: CheckCircle2,
      };
    case 'error':
      return { title: 'Run error', body: event.payload.message || 'Unknown error', icon: AlertTriangle };
    default:
      return { title: event.type.replaceAll('_', ' '), body: '', icon: Activity };
  }
}

export default function TracePanel({ events }: { events: RunEvent[] }) {
  const visible = events.filter(event =>
    ['planner_decision', 'tool_started', 'observation', 'approval_recorded', 'error'].includes(event.type),
  );
  if (!visible.length) return <div className="trace-empty">Typed action and evidence events will appear here.</div>;
  return (
    <div className="trace-list">
      {visible.map(event => {
        const item = content(event);
        const Icon = item.icon;
        return (
          <article className={`trace-item trace-${event.type}`} key={event.event_id}>
            <Icon size={15} />
            <div>
              <div className="trace-title"><span>{item.title}</span><small>#{event.sequence}</small></div>
              {item.body && <p>{item.body}</p>}
            </div>
          </article>
        );
      })}
    </div>
  );
}

