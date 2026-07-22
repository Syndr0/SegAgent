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
    | 'tool_started'
    | 'observation'
    | 'artifact'
    | 'approval_required'
    | 'approval_recorded'
    | 'answer'
    | 'error'
    | 'run_completed';
  timestamp: string;
  payload: {
    [key: string]: unknown;
    action?: string;
    approval?: { decision?: string };
    artifact?: ArtifactRef;
    confidence?: number;
    message?: string;
    observation?: { tool?: string; summary?: string };
    query?: string;
    rationale_summary?: string;
    structures?: string[];
    text?: string;
    tool?: string;
  };
}

export interface ViewerMask extends ArtifactRef {
  visible: boolean;
  color: string;
  swatch: string;
}

export interface ApprovalRequest {
  message: string;
  summary: string;
  artifacts: ArtifactRef[];
  allowed_decisions: Array<'approve' | 'reject' | 'feedback'>;
}

export interface RunRecord {
  run_id: string;
  case_id: string;
  question: string;
  status: 'created' | 'running' | 'waiting_approval' | 'completed' | 'failed';
  final_answer?: string | null;
}

export interface RunHistory {
  run: RunRecord;
  events: RunEvent[];
}
