import { useEffect, useRef, useState } from 'react';
import { Send, Brain, Loader2, Wrench, Activity, MessageSquare, ChevronDown, ChevronRight } from 'lucide-react';

const API = 'http://localhost:8000';

// One streamed reasoning trace produced by SegAgent for a single question.
export interface AgentStep {
  kind: 'thinking' | 'action' | 'observation';
  step: number;
  text?: string;
  prompt?: string;
  tool?: string; // for 'action': "segment" | "lookup_oar"
}

interface ChatTurn {
  id: string;
  question: string;
  steps: AgentStep[];
  answer: string | null;
  error: string | null;
  running: boolean;
}

interface ChatPanelProps {
  image: File | null;
  // Called when the agent produces a mask; App fetches it and adds it to the viewer.
  onMask: (maskId: string, prompt: string) => void;
}

export default function ChatPanel({ image, onMask }: ChatPanelProps) {
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [turns]);

  const send = async () => {
    if (!input.trim() || !image || busy) return;
    const question = input.trim();
    setInput('');
    setBusy(true);

    const turnId = Date.now().toString();
    setTurns(prev => [...prev, { id: turnId, question, steps: [], answer: null, error: null, running: true }]);

    const patch = (fn: (t: ChatTurn) => ChatTurn) =>
      setTurns(prev => prev.map(t => (t.id === turnId ? fn(t) : t)));

    try {
      const form = new FormData();
      form.append('image', image);
      form.append('question', question);

      const res = await fetch(`${API}/chat`, { method: 'POST', body: form });
      if (!res.ok || !res.body) throw new Error(`Request failed (${res.status})`);

      // Read the NDJSON stream line by line.
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
            case 'thinking':
              patch(t => ({ ...t, steps: [...t.steps, { kind: 'thinking', step: ev.step, text: ev.text }] }));
              break;
            case 'action':
              patch(t => ({ ...t, steps: [...t.steps, { kind: 'action', step: ev.step, prompt: ev.prompt, tool: ev.tool }] }));
              break;
            case 'observation':
              patch(t => ({ ...t, steps: [...t.steps, { kind: 'observation', step: ev.step, text: ev.text, prompt: ev.prompt }] }));
              break;
            case 'mask':
              onMask(ev.mask_id, ev.prompt);
              break;
            case 'answer':
              patch(t => ({ ...t, answer: ev.text }));
              break;
            case 'error':
              patch(t => ({ ...t, error: ev.text }));
              break;
          }
        }
      }
    } catch (e) {
      patch(t => ({ ...t, error: e instanceof Error ? e.message : String(e) }));
    } finally {
      patch(t => ({ ...t, running: false }));
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-slate-900/40 border-l border-slate-800">
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-slate-800 flex-shrink-0">
        <MessageSquare className="w-4 h-4 text-indigo-400" />
        <span className="text-sm font-semibold text-slate-200">SegAgent Chat</span>
        <span className="text-[10px] text-slate-500 ml-auto">Qwen2.5-VL + VoxTell</span>
      </div>

      {/* Transcript */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-5">
        {turns.length === 0 && (
          <div className="text-center text-slate-600 text-sm mt-10 px-4">
            Ask a question about the scan, e.g.<br />
            <span className="text-slate-400">"Is the spleen enlarged?"</span> or{' '}
            <span className="text-slate-400">"Compare the two kidneys."</span>
          </div>
        )}
        {turns.map(turn => (
          <div key={turn.id} className="space-y-3">
            {/* User question */}
            <div className="flex justify-end">
              <div className="max-w-[85%] bg-indigo-600/90 text-white rounded-2xl rounded-br-sm px-3.5 py-2 text-sm">
                {turn.question}
              </div>
            </div>

            {/* Chain of thought */}
            {turn.steps.length > 0 && <ChainOfThought steps={turn.steps} running={turn.running} />}

            {/* Final answer */}
            {turn.answer && (
              <div className="flex justify-start">
                <div className="max-w-[90%] bg-slate-800 border border-slate-700 text-slate-100 rounded-2xl rounded-bl-sm px-3.5 py-2.5 text-sm whitespace-pre-wrap">
                  {turn.answer}
                </div>
              </div>
            )}

            {turn.error && (
              <div className="text-xs text-red-400 bg-red-500/10 border border-red-500/30 rounded-lg px-3 py-2">
                {turn.error}
              </div>
            )}

            {turn.running && !turn.answer && (
              <div className="flex items-center gap-2 text-xs text-slate-500">
                <Loader2 className="w-3.5 h-3.5 animate-spin" /> reasoning…
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Composer */}
      <div className="p-3 border-t border-slate-800 flex-shrink-0">
        {!image && (
          <p className="text-[11px] text-amber-400/80 mb-2 text-center">Upload a scan to start chatting.</p>
        )}
        <div className="flex items-end gap-2">
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            placeholder="Ask about the scan…"
            rows={1}
            disabled={!image || busy}
            className="flex-1 resize-none bg-slate-800/60 border border-slate-700 rounded-xl px-3 py-2 text-sm text-slate-200 placeholder:text-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 disabled:opacity-50 max-h-32"
          />
          <button
            onClick={send}
            disabled={!image || busy || !input.trim()}
            className="p-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white disabled:bg-slate-800 disabled:text-slate-600 transition-all flex-shrink-0"
            title="Send"
          >
            {busy ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
          </button>
        </div>
      </div>
    </div>
  );
}

// Collapsible chain-of-thought block rendering thoughts, tool calls and observations.
function ChainOfThought({ steps, running }: { steps: AgentStep[]; running: boolean }) {
  const [open, setOpen] = useState(true);
  return (
    <div className="bg-slate-800/30 border border-slate-700/60 rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center gap-2 px-3 py-2 text-[11px] uppercase tracking-wider text-slate-400 hover:bg-slate-800/40"
      >
        <Brain className="w-3.5 h-3.5 text-violet-400" />
        Chain of Thought
        {running && <Loader2 className="w-3 h-3 animate-spin text-slate-500" />}
        {open ? <ChevronDown className="w-3.5 h-3.5 ml-auto" /> : <ChevronRight className="w-3.5 h-3.5 ml-auto" />}
      </button>
      {open && (
        <div className="px-3 pb-3 space-y-2.5">
          {steps.map((s, i) => {
            if (s.kind === 'thinking') {
              return (
                <div key={i} className="flex gap-2 text-xs text-slate-300 leading-relaxed">
                  <span className="text-violet-400/70 font-mono flex-shrink-0">{s.step}.</span>
                  <span className="whitespace-pre-wrap">{s.text}</span>
                </div>
              );
            }
            if (s.kind === 'action') {
              return (
                <div key={i} className="flex items-center gap-2 text-xs text-cyan-300 bg-cyan-500/10 border border-cyan-500/20 rounded-lg px-2.5 py-1.5">
                  <Wrench className="w-3.5 h-3.5 flex-shrink-0" />
                  <span className="font-mono">{s.tool ?? 'segment'}("{s.prompt}")</span>
                </div>
              );
            }
            return (
              <div key={i} className="flex gap-2 text-xs text-emerald-300/90 bg-emerald-500/5 border border-emerald-500/20 rounded-lg px-2.5 py-1.5">
                <Activity className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
                <span className="whitespace-pre-wrap">{s.text}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
