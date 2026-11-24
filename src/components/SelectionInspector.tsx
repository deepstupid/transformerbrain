import React from 'react';
import { X } from 'lucide-react';
import { ThoughtFrame, TinyTransformer } from '../ml/transformer';
import { CharTokenizer, describeToken, formatTokenLabel } from '../ml/tokenizer';
import { vectorL2Norm } from '../ml/pca';
import { SelectionTarget } from '../types/observatory';

interface SelectionInspectorProps {
  selection: SelectionTarget | null;
  onClear: () => void;
  onSelect: (target: SelectionTarget) => void;
  frame: ThoughtFrame;
  model: TinyTransformer;
  tokenizer: CharTokenizer;
  activeLayer: number;
  activeHead: number;
}

export const SelectionInspector: React.FC<SelectionInspectorProps> = ({
  selection,
  onClear,
  onSelect,
  frame,
  model,
  tokenizer,
  activeLayer,
  activeHead,
}) => {
  if (!selection) {
    return (
      <div className="px-3.5 py-2.5 bg-[#0C111B] border border-[#1E293B] rounded-lg flex items-center justify-between gap-4">
        <div className="flex items-center gap-3 min-w-0">
          <span className="text-xs font-medium text-slate-300 whitespace-nowrap">
            Inspector Idle
          </span>
          <span className="text-xs text-slate-400 truncate">
            Click any embedding node, attention matrix cell, pathway node, or logit bar to isolate its connections and tensor values.
          </span>
        </div>
        <div className="flex items-center gap-3 text-[11px] font-mono tabular-nums text-slate-400 shrink-0">
          <span>Entropy {frame.entropy.toFixed(2)} bits</span>
          <span aria-hidden="true">·</span>
          <span>Top Next: {formatTokenLabel(frame.topCandidates[0]?.token ?? ' ')} ({((frame.topCandidates[0]?.prob ?? 0) * 100).toFixed(1)}%)</span>
        </div>
      </div>
    );
  }

  const renderVectorStrip = (vec: Float32Array, offset: number, dim: number) => {
    const slice = Array.from(vec.subarray(offset, offset + dim));
    const maxAbs = Math.max(0.2, ...slice.map((v) => Math.abs(v)));
    return (
      <div className="flex items-center gap-0.5 h-5" title={`${dim}-dim vector components`}>
        {slice.map((val, idx) => {
          const ratio = val / maxAbs;
          const h = Math.max(2, Math.round(Math.abs(ratio) * 18));
          return (
            <div
              key={idx}
              className="w-1.5 flex items-end justify-center h-full bg-[#111827]/60 rounded-[1px]"
              title={`d[${idx}] = ${val.toFixed(3)}`}
            >
              <div
                style={{ height: `${h}px` }}
                className={val >= 0 ? 'w-full bg-cyan-400/90' : 'w-full bg-rose-400/90'}
              />
            </div>
          );
        })}
      </div>
    );
  };

  if (selection.kind === 'token') {
    const id = Math.max(0, Math.min(tokenizer.vocabSize - 1, selection.tokenId));
    const tok = tokenizer.idToToken[id] ?? '?';
    const norm = frame.embeddingNorms[id] ?? 0;
    const x = frame.embeddingCoords3D[id * 3] ?? 0;
    const y = frame.embeddingCoords3D[id * 3 + 1] ?? 0;
    const z = frame.embeddingCoords3D[id * 3 + 2] ?? 0;
    const neighbors = model.getTopCosineNeighbors(id, 5);
    const candIdx = frame.topCandidates.findIndex((c) => c.tokenId === id);
    const cand = candIdx >= 0 ? frame.topCandidates[candIdx] : null;

    return (
      <div className="px-3.5 py-2.5 bg-[#0C111B] border border-cyan-500/40 rounded-lg flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-4 min-w-0">
          <div>
            <div className="text-[11px] font-mono text-amber-300">
              Embedding Vector · ID #{id} · {describeToken(tok)}
            </div>
            <div className="text-xs font-mono tabular-nums text-slate-300 mt-0.5">
              ‖E‖₂ {norm.toFixed(3)} · PCA ({x.toFixed(2)}, {y.toFixed(2)}, {z.toFixed(2)})
              {cand ? ` · Next P ${(cand.prob * 100).toFixed(1)}% (#${candIdx + 1})` : ''}
            </div>
          </div>
          {renderVectorStrip(frame.embeddingVectors, id * frame.dModel, frame.dModel)}
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 text-[11px] font-mono text-slate-300">
            <span className="text-slate-400">Nearest:</span>
            {neighbors.map((nb) => (
              <button
                key={nb.tokenId}
                type="button"
                onClick={() => onSelect({ kind: 'token', tokenId: nb.tokenId })}
                className="px-1.5 py-0.5 bg-[#131A26] hover:bg-cyan-500/20 border border-[#1E293B] rounded text-cyan-200 transition-colors tabular-nums"
              >
                {formatTokenLabel(tokenizer.idToToken[nb.tokenId] ?? '?')} {nb.similarity.toFixed(2)}
              </button>
            ))}
          </div>
          <button
            type="button"
            onClick={onClear}
            className="p-1 text-slate-400 hover:text-slate-100 rounded transition-colors"
            title="Clear selection"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>
    );
  }

  if (selection.kind === 'attentionCell') {
    const layer = Math.max(0, Math.min(frame.nLayers - 1, selection.layer ?? activeLayer));
    const ly = frame.layers[layer];
    const T = frame.seqLenActual;
    const q = Math.min(T - 1, selection.qIdx);
    const k = Math.min(q, selection.kIdx);
    const head = selection.head ?? activeHead;

    let weight = 0;
    let score = 0;
    if (head < 0) {
      for (let h = 0; h < frame.nHeads; h++) {
        weight += ly.attnProbs[h * T * T + q * T + k];
        score += ly.attnScores[h * T * T + q * T + k];
      }
      weight /= frame.nHeads;
      score /= frame.nHeads;
    } else {
      const h = Math.min(frame.nHeads - 1, head);
      weight = ly.attnProbs[h * T * T + q * T + k];
      score = ly.attnScores[h * T * T + q * T + k];
    }

    return (
      <div className="px-3.5 py-2.5 bg-[#0C111B] border border-cyan-500/40 rounded-lg flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <div>
            <div className="text-[11px] font-mono text-amber-300">
              Attention Matrix · Layer {layer} · {head < 0 ? 'Head Avg' : `Head ${head}`}
            </div>
            <div className="text-xs font-mono tabular-nums text-slate-200 mt-0.5">
              Query[{q}] '{formatTokenLabel(frame.tokens[q] ?? '?')}' ← Key[{k}] '{formatTokenLabel(frame.tokens[k] ?? '?')}'
              {' · '}Weight {(weight * 100).toFixed(2)}% · Q·K/√d {score.toFixed(3)}
            </div>
          </div>
          {renderVectorStrip(ly.attnDelta, q * frame.dModel, frame.dModel)}
        </div>
        <button
          type="button"
          onClick={onClear}
          className="p-1 text-slate-400 hover:text-slate-100 rounded transition-colors"
          title="Clear selection"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    );
  }

  if (selection.kind === 'pathwayNode') {
    const T = frame.seqLenActual;
    const pos = Math.min(T - 1, selection.tokenPos);
    const stage = Math.min(frame.nLayers, selection.stage);
    const vecs = stage === 0 ? frame.inputEmbeds : frame.layers[stage - 1].residualOut;
    const resNorm = vectorL2Norm(vecs, pos * frame.dModel, frame.dModel);
    const attnNorm =
      stage > 0 ? vectorL2Norm(frame.layers[stage - 1].attnDelta, pos * frame.dModel, frame.dModel) : 0;
    const mlpNorm =
      stage > 0 ? vectorL2Norm(frame.layers[stage - 1].mlpDelta, pos * frame.dModel, frame.dModel) : 0;

    return (
      <div className="px-3.5 py-2.5 bg-[#0C111B] border border-cyan-500/40 rounded-lg flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <div>
            <div className="text-[11px] font-mono text-amber-300">
              Residual Stream · {stage === 0 ? 'Input Embed+Pos' : `Block ${stage - 1} Output`} · Pos {pos} '{formatTokenLabel(frame.tokens[pos] ?? '?')}'
            </div>
            <div className="text-xs font-mono tabular-nums text-slate-200 mt-0.5">
              ‖x‖₂ {resNorm.toFixed(3)}
              {stage > 0 ? ` · ΔAttn ${attnNorm.toFixed(3)} · ΔMLP ${mlpNorm.toFixed(3)}` : ''}
            </div>
          </div>
          {renderVectorStrip(vecs, pos * frame.dModel, frame.dModel)}
        </div>
        <button
          type="button"
          onClick={onClear}
          className="p-1 text-slate-400 hover:text-slate-100 rounded transition-colors"
          title="Clear selection"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    );
  }

  // mlpNeuron
  const layer = Math.min(frame.nLayers - 1, selection.layer);
  const pos = Math.min(frame.seqLenActual - 1, selection.tokenPos);
  const act = frame.layers[layer].mlpHidden[pos * frame.dMlp + selection.neuronIdx] ?? 0;

  return (
    <div className="px-3.5 py-2.5 bg-[#0C111B] border border-cyan-500/40 rounded-lg flex flex-wrap items-center justify-between gap-4">
      <div>
        <div className="text-[11px] font-mono text-amber-300">
          MLP Hidden Neuron · Block {layer} · Neuron #{selection.neuronIdx} · Pos {pos} '{formatTokenLabel(frame.tokens[pos] ?? '?')}'
        </div>
        <div className="text-xs font-mono tabular-nums text-slate-200 mt-0.5">
          GELU Activation {act.toFixed(4)} · {act > 0 ? 'Excited (+)' : 'Suppressed (−)'}
        </div>
      </div>
      <button
        type="button"
        onClick={onClear}
        className="p-1 text-slate-400 hover:text-slate-100 rounded transition-colors"
        title="Clear selection"
      >
        <X className="w-4 h-4" />
      </button>
    </div>
  );
};
