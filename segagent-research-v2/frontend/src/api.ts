import type {
  ArtifactRef,
  CaseRecord,
  ConversationHistory,
  MessageContext,
  RunEvent,
  RunHistory,
} from './types';

export const API = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

const RUN_EVENT_TYPES = new Set<RunEvent['type']>([
  'run_started',
  'planner_decision',
  'thinking_delta',
  'thinking',
  'reasoning_summary',
  'phase_started',
  'phase_completed',
  'phase_failed',
  'tool_started',
  'observation',
  'artifact',
  'question_pending',
  'reply_recorded',
  'answer_delta',
  'answer',
  'error',
  'run_completed',
]);

function parseRunEvent(line: string): RunEvent {
  const value: unknown = JSON.parse(line);
  if (!value || typeof value !== 'object') throw new Error('The server returned an invalid event.');
  const event = value as Partial<RunEvent>;
  if (
    typeof event.event_id !== 'string'
    || typeof event.run_id !== 'string'
    || typeof event.case_id !== 'string'
    || typeof event.sequence !== 'number'
    || !Number.isInteger(event.sequence)
    || typeof event.type !== 'string'
    || !RUN_EVENT_TYPES.has(event.type as RunEvent['type'])
    || !event.payload
    || typeof event.payload !== 'object'
  ) {
    throw new Error('The server returned an incomplete event.');
  }
  return event as RunEvent;
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const payload: unknown = await response.json();
    if (payload && typeof payload === 'object' && 'detail' in payload) {
      const detail = (payload as { detail: unknown }).detail;
      if (typeof detail === 'string') return detail;
      if (Array.isArray(detail)) {
        const messages = detail
          .map(item => item && typeof item === 'object' && 'msg' in item ? String(item.msg) : '')
          .filter(Boolean);
        if (messages.length) return messages.join(' ');
      }
    }
    return JSON.stringify(payload);
  } catch {
    return `${response.status} ${response.statusText}`;
  }
}

export async function createCase(
  image: File,
  contours: File[] = [],
  signal?: AbortSignal,
): Promise<CaseRecord> {
  const form = new FormData();
  form.append('image', image);
  for (const contour of contours) form.append('contours', contour);
  const response = await fetch(`${API}/api/cases`, { method: 'POST', body: form, signal });
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function addContours(
  caseId: string,
  contours: File[],
  signal?: AbortSignal,
): Promise<CaseRecord> {
  const form = new FormData();
  for (const contour of contours) form.append('contours', contour);
  const response = await fetch(`${API}/api/cases/${caseId}/contours`, {
    method: 'POST',
    body: form,
    signal,
  });
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function readEventStream(
  response: Response,
  onEvent: (event: RunEvent) => void,
): Promise<void> {
  if (!response.ok || !response.body) throw new Error(await errorMessage(response));
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  for (;;) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    let newline = buffer.indexOf('\n');
    while (newline >= 0) {
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      if (line) onEvent(parseRunEvent(line));
      newline = buffer.indexOf('\n');
    }
    if (done) break;
  }
  if (buffer.trim()) onEvent(parseRunEvent(buffer));
}

/** Send one conversational turn. Viewer artifact IDs travel with the message,
 *  so references such as "this contour" resolve to explicit case-owned masks.
 *  The server decides whether this opens a new turn or answers a pending one. */
export async function sendMessage(
  caseId: string,
  question: string,
  onEvent: (event: RunEvent) => void,
  onRunId: (runId: string) => void,
  signal?: AbortSignal,
  context: MessageContext = {},
): Promise<string> {
  const response = await fetch(`${API}/api/cases/${caseId}/messages`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, ...context }),
    signal,
  });
  const runId = response.headers.get('X-Run-Id') || '';
  if (!runId && response.ok) throw new Error('The server did not return a run ID.');
  if (runId) onRunId(runId);
  await readEventStream(response, onEvent);
  return runId;
}

export async function uploadEditedMask(
  caseId: string,
  structure: string,
  file: File,
  derivedFrom?: string,
  targetType: string = 'unknown',
  signal?: AbortSignal,
): Promise<ArtifactRef> {
  const form = new FormData();
  form.append('structure', structure);
  form.append('file', file);
  if (derivedFrom) form.append('derived_from', derivedFrom);
  form.append('target_type', targetType);
  const response = await fetch(`${API}/api/cases/${caseId}/edited-mask`, {
    method: 'POST',
    body: form,
    signal,
  });
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export const caseImageUrl = (caseId: string) => `${API}/api/cases/${caseId}/image`;
export const artifactUrl = (caseId: string, artifactId: string) =>
  `${API}/api/cases/${caseId}/artifacts/${artifactId}`;

export async function getCase(caseId: string, signal?: AbortSignal): Promise<CaseRecord> {
  const response = await fetch(`${API}/api/cases/${caseId}`, { signal });
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function getRun(runId: string, signal?: AbortSignal): Promise<RunHistory> {
  const response = await fetch(`${API}/api/runs/${runId}`, { signal });
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function getConversation(
  caseId: string,
  signal?: AbortSignal,
): Promise<ConversationHistory> {
  const response = await fetch(`${API}/api/cases/${caseId}/conversation`, { signal });
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}
