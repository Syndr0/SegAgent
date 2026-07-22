import type { ApprovalRequest, CaseRecord, RunEvent } from './types';

export const API = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

async function errorMessage(response: Response): Promise<string> {
  try {
    const payload = await response.json();
    return payload.detail || JSON.stringify(payload);
  } catch {
    return `${response.status} ${response.statusText}`;
  }
}

export async function createCase(image: File, contours: File[]): Promise<CaseRecord> {
  const form = new FormData();
  form.append('image', image);
  for (const contour of contours) form.append('contours', contour);
  const response = await fetch(`${API}/api/cases`, { method: 'POST', body: form });
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
      if (line) onEvent(JSON.parse(line) as RunEvent);
      newline = buffer.indexOf('\n');
    }
    if (done) break;
  }
  if (buffer.trim()) onEvent(JSON.parse(buffer) as RunEvent);
}

export async function startRun(
  caseId: string,
  question: string,
  onEvent: (event: RunEvent) => void,
): Promise<string> {
  const response = await fetch(`${API}/api/cases/${caseId}/runs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  });
  const runId = response.headers.get('X-Run-Id') || '';
  await readEventStream(response, onEvent);
  return runId;
}

export async function resumeRun(
  runId: string,
  decision: 'approve' | 'reject' | 'feedback',
  feedback: string,
  onEvent: (event: RunEvent) => void,
): Promise<void> {
  const response = await fetch(`${API}/api/runs/${runId}/resume`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ decision, feedback: feedback || null }),
  });
  await readEventStream(response, onEvent);
}

export const caseImageUrl = (caseId: string) => `${API}/api/cases/${caseId}/image`;
export const artifactUrl = (caseId: string, artifactId: string) =>
  `${API}/api/cases/${caseId}/artifacts/${artifactId}`;

export function approvalFromEvent(event: RunEvent): ApprovalRequest {
  return event.payload as unknown as ApprovalRequest;
}

