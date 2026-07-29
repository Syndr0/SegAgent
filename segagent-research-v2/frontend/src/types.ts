export type ArtifactKind = 'image' | 'contour' | 'mask' | 'overlay' | 'report' | 'export';

export interface ArtifactRef {
  artifact_id: string;
  case_id: string;
  kind: ArtifactKind;
  label: string;
  media_type: string;
  sha256: string;
  metadata: Record<string, unknown>;
}

export interface CaseRecord {
  case_id: string;
  source_name: string;
  image: ArtifactRef;
  contours: ArtifactRef[];
  artifacts: ArtifactRef[];
  metadata: Record<string, unknown>;
}

export interface RunEvent {
  event_id: string;
  run_id: string;
  case_id: string;
  sequence: number;
  type:
    | 'run_started'
    | 'planner_decision'
    | 'thinking'
    | 'tool_started'
    | 'observation'
    | 'artifact'
    | 'question_pending'
    | 'reply_recorded'
    | 'answer'
    | 'error'
    | 'run_completed';
  timestamp: string;
  payload: {
    [key: string]: unknown;
    action?: string;
    artifact?: ArtifactRef;
    intent?: TaskIntent;
    message?: string;
    observation?: { tool?: string; summary?: string };
    query?: string;
    question?: string;
    rationale_summary?: string;
    structures?: string[];
    text?: string;
    thinking?: string;
    tool?: string;
  };
}

export interface TaskIntent {
  intent: 'organ' | 'oar' | 'gtv' | 'protocol' | 'question';
  confidence: number;
  targets: string[];
  site?: string | null;
  rationale?: string;
}

export interface ChatMessage {
  message_id: string;
  conversation_id: string;
  case_id: string;
  role: 'user' | 'agent';
  text: string;
  run_id?: string | null;
  created_at: string;
  metadata: Record<string, unknown>;
}

export interface ConversationRecord {
  conversation_id: string;
  case_id: string;
  turn_count: number;
  pending_question?: string | null;
  pending_run_id?: string | null;
}

export interface ConversationHistory {
  conversation: ConversationRecord;
  messages: ChatMessage[];
}

export interface ViewerMask extends ArtifactRef {
  visible: boolean;
  color: string;
  swatch: string;
}

export interface RunRecord {
  run_id: string;
  case_id: string;
  question: string;
  status: 'created' | 'running' | 'waiting_reply' | 'completed' | 'failed';
  final_answer?: string | null;
}

export interface RunHistory {
  run: RunRecord;
  events: RunEvent[];
}
