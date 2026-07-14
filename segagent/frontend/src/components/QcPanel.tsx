import { useRef, useState } from 'react';
import { Upload, Loader2, ShieldCheck, AlertTriangle, XCircle, CheckCircle2, Brain, ChevronDown, ChevronRight, ClipboardList, Circle } from 'lucide-react';

const API = 'http://localhost:8000';

interface Finding { check: string; severity: 'ok' | 'warn' | 'error'; message: string; }
interface OrganRow { organ: string; status: 'ok' | 'warn' | 'error'; volume_ml: number | null; dice: number | null; findings: Finding[]; }

interface QcState {
  running: boolean;
  fileName: string | null;
  warnings: string[];
  structures: string[];      // all contours to process (from qc_start)
  phase: string | null;      // current phase text
  organs: OrganRow[];        // completed rows
  thoughts: string[];
  report: string | null;
  error: string | null;
}

const EMPTY: QcState = {
  running: false, fileName: null, warnings: [], structures: [],
  phase: null, organs: [], thoughts: [], report: null, error: null,
};

const STATUS = {
  ok: { icon: CheckCircle2, color: 'text-emerald-400', bg: 'bg-emerald-500/10 border-emerald-500/30' },
  warn: { icon: AlertTriangle, color: 'text-amber-400', bg: 'bg-amber-500/10 border-amber-500/30' },
  error: { icon: XCircle, color: 'text-red-400', bg: 'bg-red-500/10 border-red-500/30' },
};

interface QcPanelProps { onMask?: (maskId: string, prompt: string) => void; }

export default function QcPanel({ onMask }: QcPanelProps) {
  const [st, setSt] = useState<QcState>(EMPTY);
  const fileRef = useRef<HTMLInputElement>(null);

  const patch = (u: Partial<QcState> | ((s: QcState) => QcState)) =>
    setSt(prev => (typeof u === 'function' ? u(prev) : { ...prev, ...u }));

  const onFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setSt({ ...EMPTY, running: true, fileName: file.name, phase: 'Reading structure set…' });

    try {
      const form = new FormData();
      form.append('archive', file);
      form.append('question', '');
      const res = await fetch(`${API}/qc`, { method: 'POST', body: form });
      if (!res.ok || !res.body) throw new Error(`Request failed (${res.status})`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let nl: number;
        while ((nl = buffer.indexOf('\n')) >= 0) {
          const line = buffer.slice(0, nl).trim();
          buffer = buffer.slice(nl + 1);
          if (!line) continue;
          const ev = JSON.parse(line);
          switch (ev.type) {
            case 'qc_start':
              patch({ structures: ev.structures || [], warnings: ev.warnings || [], phase: 'Preparing…' });
              break;
            case 'qc_phase':
              patch({ phase: ev.text });
              break;
            case 'qc_organ':
              patch(s => ({ ...s, organs: [...s.organs, ev as OrganRow] }));
              break;
            case 'mask':
              if (onMask) onMask(ev.mask_id, ev.prompt);
              break;
            case 'thinking':
              patch(s => ({ ...s, thoughts: [...s.thoughts, ev.text] }));
              break;
            case 'answer':
              patch({ report: ev.text, phase: null });
              break;
            case 'error':
              patch(s => ({ ...s, error: ev.text }));
              break;
          }
        }
      }
    } catch (err) {
      patch({ error: err instanceof Error ? err.message : String(err) });
    } finally {
      patch({ running: false, phase: null });
      if (fileRef.current) fileRef.current.value = '';
    }
  };

  const counts = st.organs.reduce(
    (a, o) => ({ ...a, [o.status]: a[o.status] + 1 }),
    { ok: 0, warn: 0, error: 0 } as Record<string, number>,
  );
  const total = st.structures.length;
  const done = st.organs.length;
  const pct = total > 0 ? Math.round((done / total) * 100) : 0;
  const rowByOrgan = new Map(st.organs.map(o => [o.organ, o]));

  return (
    <div className="flex flex-col h-full bg-slate-900/40 border-l border-slate-800">
      <div className="flex items-center gap-2 px-4 py-3 border-b border-slate-800 flex-shrink-0">
        <ShieldCheck className="w-4 h-4 text-indigo-400" />
        <span className="text-sm font-semibold text-slate-200">Contour QC</span>
        <span className="text-[10px] text-slate-500 ml-auto">geometry + expert model</span>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Upload */}
        <div className="relative group">
          <input
            ref={fileRef}
            type="file"
            accept=".zip"
            onChange={onFile}
            disabled={st.running}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20 disabled:cursor-not-allowed"
          />
          <div className="border-2 border-dashed border-slate-700 hover:border-slate-500 bg-slate-800/30 rounded-xl p-4 flex flex-col items-center gap-2 text-center">
            {st.running ? <Loader2 className="w-6 h-6 text-indigo-400 animate-spin" /> : <Upload className="w-6 h-6 text-slate-400" />}
            <p className="text-sm font-medium text-slate-300">
              {st.running ? 'Analyzing…' : st.fileName || 'Upload structure set (.zip)'}
            </p>
            <p className="text-xs text-slate-500">image.nii + left_eye.nii, spinal_cord.nii, …</p>
          </div>
        </div>

        {st.error && (
          <div className="text-xs text-red-400 bg-red-500/10 border border-red-500/30 rounded-lg px-3 py-2">{st.error}</div>
        )}

        {/* Phase + progress */}
        {st.running && (
          <div className="space-y-2">
            {st.phase && (
              <div className="flex items-center gap-2 text-xs text-indigo-300">
                <Loader2 className="w-3.5 h-3.5 animate-spin" /> {st.phase}
              </div>
            )}
            {total > 0 && (
              <div>
                <div className="flex justify-between text-[10px] text-slate-500 mb-1">
                  <span>Contours checked</span><span>{done}/{total}</span>
                </div>
                <div className="h-1.5 rounded-full bg-slate-800 overflow-hidden">
                  <div className="h-full bg-indigo-500 transition-all duration-300" style={{ width: `${pct}%` }} />
                </div>
              </div>
            )}
          </div>
        )}

        {/* Overall counts */}
        {st.organs.length > 0 && (
          <div className="flex items-center gap-3 text-xs">
            <span className="text-emerald-400">{counts.ok} ok</span>
            <span className="text-amber-400">{counts.warn} review</span>
            <span className="text-red-400">{counts.error} problem</span>
          </div>
        )}

        {/* Checklist: every contour, pending until its result arrives */}
        {st.structures.length > 0 && (
          <div className="space-y-1.5">
            <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">Contours ({total})</div>
            {st.structures.map(name => {
              const row = rowByOrgan.get(name);
              return row
                ? <OrganCard key={name} row={row} />
                : (
                  <div key={name} className="flex items-center gap-2 px-2.5 py-1.5 rounded-lg border border-slate-800 bg-slate-800/20 text-slate-500">
                    {st.running ? <Loader2 className="w-3.5 h-3.5 animate-spin flex-shrink-0" /> : <Circle className="w-3.5 h-3.5 flex-shrink-0" />}
                    <span className="text-sm truncate">{name}</span>
                    <span className="text-[10px] ml-auto">pending</span>
                  </div>
                );
            })}
          </div>
        )}

        {/* Narrative report */}
        {st.report && (
          <div className="bg-slate-800 border border-slate-700 rounded-xl p-3">
            <div className="flex items-center gap-2 text-xs font-semibold text-slate-300 mb-2">
              <ClipboardList className="w-3.5 h-3.5 text-indigo-400" /> QC Report
            </div>
            <div className="text-sm text-slate-200 whitespace-pre-wrap leading-relaxed">{st.report}</div>
          </div>
        )}

        {st.thoughts.length > 0 && <Reasoning thoughts={st.thoughts} />}

        {st.warnings.length > 0 && (
          <div className="text-[11px] text-slate-500 space-y-0.5">
            {st.warnings.map((w, i) => <div key={i}>⚠ {w}</div>)}
          </div>
        )}
      </div>
    </div>
  );
}

function OrganCard({ row }: { row: OrganRow }) {
  const [open, setOpen] = useState(row.status !== 'ok');
  const s = STATUS[row.status];
  const Icon = s.icon;
  return (
    <div className={`border rounded-lg ${s.bg}`}>
      <button onClick={() => setOpen(o => !o)} className="w-full flex items-center gap-2 px-2.5 py-1.5 text-left">
        <Icon className={`w-4 h-4 flex-shrink-0 ${s.color}`} />
        <span className="text-sm text-slate-200 flex-1 truncate">{row.organ}</span>
        <span className="text-[10px] text-slate-400 font-mono">
          {row.volume_ml != null ? `${row.volume_ml} mL` : ''}{row.dice != null ? ` · D${row.dice}` : ''}
        </span>
        {row.findings.length > 0 && (open ? <ChevronDown className="w-3.5 h-3.5 text-slate-500" /> : <ChevronRight className="w-3.5 h-3.5 text-slate-500" />)}
      </button>
      {open && row.findings.length > 0 && (
        <div className="px-2.5 pb-2 space-y-1">
          {row.findings.map((f, i) => (
            <div key={i} className={`text-[11px] ${f.severity === 'error' ? 'text-red-300' : f.severity === 'warn' ? 'text-amber-300' : 'text-slate-400'}`}>
              · {f.message}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function Reasoning({ thoughts }: { thoughts: string[] }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="bg-slate-800/30 border border-slate-700/60 rounded-xl overflow-hidden">
      <button onClick={() => setOpen(o => !o)} className="w-full flex items-center gap-2 px-3 py-2 text-[11px] uppercase tracking-wider text-slate-400 hover:bg-slate-800/40">
        <Brain className="w-3.5 h-3.5 text-violet-400" /> Reasoning
        {open ? <ChevronDown className="w-3.5 h-3.5 ml-auto" /> : <ChevronRight className="w-3.5 h-3.5 ml-auto" />}
      </button>
      {open && (
        <div className="px-3 pb-3 space-y-2 text-xs text-slate-300 whitespace-pre-wrap">
          {thoughts.map((t, i) => <div key={i}>{t}</div>)}
        </div>
      )}
    </div>
  );
}
