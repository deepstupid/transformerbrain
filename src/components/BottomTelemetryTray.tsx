import React, { useState } from 'react';
import { ThoughtFrame, TrainStepMetrics } from '../ml/transformer';
import { formatTokenLabel } from '../ml/tokenizer';
import { SelectionTarget } from '../types/observatory';

interface BottomTelemetryTrayProps {
  lossHistory: TrainStepMetrics[];
  frame: ThoughtFrame;
  selection: SelectionTarget | null;
  onSelect: (target: SelectionTarget) => void;
}

export const BottomTelemetryTray: React.FC<BottomTelemetryTrayProps> = ({
  lossHistory,
  frame,
  selection,
  onSelect,
}) => {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  const latest = lossHistory[lossHistory.length - 1] ?? {
    step: frame.step,
    loss: 0,
    perplexity: 1,
    gradNorm: 0,
  };

  const w = 440;
  const h = 84;
  const padL = 32;
  const padR = 10;
  const padT = 8;
  const padB = 14;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const losses = lossHistory.map((m) => m.loss);
  const maxLoss = Math.max(2.5, ...losses);
  const minLoss = Math.max(0, Math.min(...losses, maxLoss - 0.5) * 0.9);

  const points = lossHistory.map((m, i) => {
    const x =
      lossHistory.length <= 1
        ? padL + plotW * 0.5
        : padL + (i / (lossHistory.length - 1)) * plotW;
    const normY = (m.loss - minLoss) / Math.max(1e-4, maxLoss - minLoss);
    const y = padT + (1 - Math.max(0, Math.min(1, normY))) * plotH;
    return { x, y, m };
  });

  const pathD =
    points.length > 0
      ? points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' ')
      : '';

  const hoveredPoint = hoverIdx !== null ? points[hoverIdx] : null;

  let activePos: number | null = null;
  if (selection?.kind === 'pathwayNode') activePos = selection.tokenPos;
  else if (selection?.kind === 'attentionCell') activePos = selection.qIdx;
  else if (selection?.kind === 'mlpNeuron') activePos = selection.tokenPos;
  else if (selection?.kind === 'token' && selection.tokenPos !== undefined) {
    activePos = selection.tokenPos;
  }

  return (
    <section className="bg-[#0C111B] border border-[#1E293B] rounded-lg px-3.5 py-2.5 grid grid-cols-1 lg:grid-cols-12 gap-4 shrink-0">
      {/* Training Loss Sparkline */}
      <div className="lg:col-span-5 flex flex-col gap-1.5 min-w-0">
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs font-semibold text-slate-200">Training Telemetry</span>
          <div className="flex items-center gap-2.5 text-[11px] font-mono tabular-nums text-slate-300">
            <span>Step {(hoveredPoint?.m.step ?? latest.step).toLocaleString()}</span>
            <span aria-hidden="true" className="text-slate-600">·</span>
            <span className="text-cyan-300">
              Loss {(hoveredPoint?.m.loss ?? latest.loss).toFixed(3)}
            </span>
            <span aria-hidden="true" className="text-slate-600">·</span>
            <span className="text-amber-300">
              PPL {(hoveredPoint?.m.perplexity ?? latest.perplexity).toFixed(1)}
            </span>
          </div>
        </div>

        <svg
          viewBox={`0 0 ${w} ${h}`}
          className="w-full h-[84px] bg-[#07090E] border border-[#1E293B] rounded"
          onMouseLeave={() => setHoverIdx(null)}
          onMouseMove={(e) => {
            if (points.length === 0) return;
            const rect = e.currentTarget.getBoundingClientRect();
            const relX = ((e.clientX - rect.left) / rect.width) * w;
            const frac = Math.max(0, Math.min(1, (relX - padL) / plotW));
            setHoverIdx(Math.round(frac * (points.length - 1)));
          }}
        >
          {/* Horizontal hairline grid */}
          {[0, 0.5, 1].map((t) => {
            const gy = padT + t * plotH;
            const val = maxLoss - t * (maxLoss - minLoss);
            return (
              <g key={t}>
                <line
                  x1={padL}
                  y1={gy}
                  x2={w - padR}
                  y2={gy}
                  stroke="#1E293B"
                  strokeWidth={1}
                />
                <text
                  x={padL - 4}
                  y={gy + 3}
                  textAnchor="end"
                  fill="#64748b"
                  fontSize="9"
                  fontFamily="IBM Plex Mono, monospace"
                >
                  {val.toFixed(1)}
                </text>
              </g>
            );
          })}

          {pathD && (
            <path
              d={pathD}
              fill="none"
              stroke="#06B6D4"
              strokeWidth={1.75}
              strokeLinejoin="round"
            />
          )}

          {hoveredPoint && (
            <g>
              <line
                x1={hoveredPoint.x}
                y1={padT}
                x2={hoveredPoint.x}
                y2={h - padB}
                stroke="#F59E0B"
                strokeWidth={1}
                strokeDasharray="2 2"
              />
              <circle
                cx={hoveredPoint.x}
                cy={hoveredPoint.y}
                r={3}
                fill="#F59E0B"
              />
            </g>
          )}
        </svg>
      </div>

      {/* Clickable Token Context Tape */}
      <div className="lg:col-span-7 flex flex-col gap-1.5 min-w-0">
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs font-semibold text-slate-200">
            Active Context Window ({frame.seqLenActual} tokens)
          </span>
          <span className="text-[11px] font-mono tabular-nums text-slate-400">
            Click a token to inspect its attention & residual pathway
          </span>
        </div>

        <div className="flex-1 bg-[#07090E] border border-[#1E293B] rounded p-2 overflow-x-auto flex items-center gap-1">
          {frame.tokens.map((tok, pos) => {
            const tokId = frame.tokenIds[pos];
            const isSelected = activePos === pos;
            const isLast = pos === frame.tokens.length - 1;
            return (
              <button
                key={`${pos}-${tokId}`}
                type="button"
                onClick={() =>
                  onSelect({
                    kind: 'pathwayNode',
                    stage: frame.nLayers,
                    tokenPos: pos,
                  })
                }
                title={`Pos ${pos} · Token ID ${tokId}`}
                className={`flex flex-col items-center justify-center min-w-[28px] h-14 px-1.5 rounded border font-mono transition-colors shrink-0 ${
                  isSelected
                    ? 'bg-amber-500/20 border-amber-400 text-amber-200'
                    : isLast
                    ? 'bg-cyan-500/15 border-cyan-500/50 text-cyan-200'
                    : 'bg-[#0F1522] border-[#1E293B] text-slate-200 hover:border-slate-500'
                }`}
              >
                <span className="text-xs font-medium leading-none">
                  {formatTokenLabel(tok)}
                </span>
                <span className="text-[9px] tabular-nums text-slate-400 mt-1.5 leading-none">
                  {pos}
                </span>
              </button>
            );
          })}
        </div>
      </div>
    </section>
  );
};
