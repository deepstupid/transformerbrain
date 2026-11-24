import React from 'react';
import { Play, Pause, RotateCcw, SkipForward, Sparkles, StepForward } from 'lucide-react';
import { CORPUS_PRESETS } from '../ml/corpora';

interface ControlDeckProps {
  selectedPresetId: string;
  onSelectPreset: (id: string) => void;
  corpusText: string;
  onCorpusTextChange: (text: string) => void;
  onApplyCustomCorpus: () => void;
  corpusDirty: boolean;
  vocabSize: number;
  corpusLength: number;
  paramCount: number;
  isTraining: boolean;
  onToggleTraining: () => void;
  onStepTrain: () => void;
  onResetModel: () => void;
  dModel: number;
  nLayers: number;
  nHeads: number;
  learningRate: number;
  onArchitectureChange: (dModel: number, nLayers: number, nHeads: number) => void;
  onLearningRateChange: (lr: number) => void;
  promptText: string;
  onPromptChange: (text: string) => void;
  temperature: number;
  onTemperatureChange: (t: number) => void;
  topK: number;
  onTopKChange: (k: number) => void;
  onStepToken: () => void;
  isGenerating: boolean;
  onToggleGenerate: () => void;
  onResetPrompt: () => void;
}

export const ControlDeck: React.FC<ControlDeckProps> = ({
  selectedPresetId,
  onSelectPreset,
  corpusText,
  onCorpusTextChange,
  onApplyCustomCorpus,
  corpusDirty,
  vocabSize,
  corpusLength,
  paramCount,
  isTraining,
  onToggleTraining,
  onStepTrain,
  onResetModel,
  dModel,
  nLayers,
  nHeads,
  learningRate,
  onArchitectureChange,
  onLearningRateChange,
  promptText,
  onPromptChange,
  temperature,
  onTemperatureChange,
  topK,
  onTopKChange,
  onStepToken,
  isGenerating,
  onToggleGenerate,
  onResetPrompt,
}) => {
  return (
    <aside className="w-full lg:w-[336px] shrink-0 bg-[#0C111B] border border-[#1E293B] rounded-lg flex flex-col overflow-y-auto">
      {/* Section 1: Corpus & Tokenizer */}
      <div className="p-3.5 border-b border-[#1E293B] space-y-2.5">
        <div className="flex items-center justify-between">
          <h2 className="text-xs font-semibold text-slate-200 tracking-tight">
            01. Corpus & Tokenizer
          </h2>
          <span className="text-[11px] font-mono tabular-nums text-slate-400">
            V={vocabSize} · {corpusLength.toLocaleString()} ch
          </span>
        </div>

        <div className="grid grid-cols-2 gap-1.5">
          {CORPUS_PRESETS.map((preset) => (
            <button
              key={preset.id}
              type="button"
              onClick={() => onSelectPreset(preset.id)}
              className={`px-2.5 py-1.5 text-left rounded border text-[11px] transition-colors ${
                selectedPresetId === preset.id && !corpusDirty
                  ? 'bg-cyan-500/15 border-cyan-500/50 text-cyan-200 font-medium'
                  : 'bg-[#121926] border-[#1E293B] text-slate-300 hover:border-slate-600'
              }`}
            >
              <div className="truncate">{preset.name}</div>
            </button>
          ))}
        </div>

        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <label htmlFor="corpus-buffer" className="text-[11px] text-slate-400">
              Training Text Buffer (Editable)
            </label>
            {corpusDirty && (
              <button
                type="button"
                onClick={onApplyCustomCorpus}
                className="px-2 py-0.5 text-[11px] font-medium bg-amber-500/20 text-amber-200 border border-amber-500/40 rounded hover:bg-amber-500/30 transition-colors"
              >
                Apply Corpus
              </button>
            )}
          </div>
          <textarea
            id="corpus-buffer"
            value={corpusText}
            onChange={(e) => onCorpusTextChange(e.target.value)}
            rows={4}
            spellCheck={false}
            placeholder="Paste custom training text here..."
            className="w-full px-2.5 py-2 bg-[#07090E] border border-[#1E293B] focus:border-cyan-500/60 focus:outline-none rounded text-[11px] font-mono leading-relaxed text-slate-200 resize-y"
          />
        </div>
      </div>

      {/* Section 2: Training & Architecture */}
      <div className="p-3.5 border-b border-[#1E293B] space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-xs font-semibold text-slate-200 tracking-tight">
            02. Training & Model
          </h2>
          <span className="text-[11px] font-mono tabular-nums text-slate-400">
            {(paramCount / 1000).toFixed(1)}k params
          </span>
        </div>

        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={onToggleTraining}
            className={`flex-1 flex items-center justify-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium transition-colors whitespace-nowrap ${
              isTraining
                ? 'bg-amber-500/20 text-amber-200 border border-amber-500/50 hover:bg-amber-500/30'
                : 'bg-cyan-500 text-slate-950 hover:bg-cyan-400'
            }`}
          >
            {isTraining ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
            <span>{isTraining ? 'Pause Training' : 'Train Model'}</span>
          </button>
          <button
            type="button"
            onClick={onStepTrain}
            disabled={isTraining}
            title="Run 1 optimization step"
            className="px-2.5 py-1.5 bg-[#121926] border border-[#1E293B] hover:border-slate-600 disabled:opacity-40 rounded text-xs text-slate-200 transition-colors flex items-center gap-1"
          >
            <SkipForward className="w-3.5 h-3.5" />
            <span>Step</span>
          </button>
          <button
            type="button"
            onClick={onResetModel}
            title="Re-initialize weights"
            className="p-1.5 bg-[#121926] border border-[#1E293B] hover:border-rose-500/50 hover:text-rose-300 rounded text-slate-400 transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Learning Rate */}
        <div className="space-y-1">
          <div className="flex items-center justify-between text-[11px]">
            <span className="text-slate-400">Learning Rate</span>
            <span className="font-mono tabular-nums text-slate-200">{learningRate.toFixed(4)}</span>
          </div>
          <input
            type="range"
            min={0.0005}
            max={0.015}
            step={0.0005}
            value={learningRate}
            onChange={(e) => onLearningRateChange(parseFloat(e.target.value))}
            className="w-full accent-cyan-400 cursor-pointer h-1.5 bg-[#131A26] rounded"
          />
        </div>

        {/* Architecture Grid */}
        <div className="grid grid-cols-3 gap-2 pt-0.5">
          <div>
            <label className="block text-[10px] text-slate-400 mb-1">Embed Dim</label>
            <select
              value={dModel}
              onChange={(e) => onArchitectureChange(Number(e.target.value), nLayers, nHeads)}
              className="w-full px-2 py-1 bg-[#07090E] border border-[#1E293B] rounded text-[11px] font-mono tabular-nums text-slate-200 focus:border-cyan-500/60 focus:outline-none"
            >
              <option value={24}>24</option>
              <option value={32}>32</option>
              <option value={48}>48</option>
            </select>
          </div>
          <div>
            <label className="block text-[10px] text-slate-400 mb-1">Layers</label>
            <select
              value={nLayers}
              onChange={(e) => onArchitectureChange(dModel, Number(e.target.value), nHeads)}
              className="w-full px-2 py-1 bg-[#07090E] border border-[#1E293B] rounded text-[11px] font-mono tabular-nums text-slate-200 focus:border-cyan-500/60 focus:outline-none"
            >
              <option value={2}>2</option>
              <option value={3}>3</option>
            </select>
          </div>
          <div>
            <label className="block text-[10px] text-slate-400 mb-1">Heads</label>
            <select
              value={nHeads}
              onChange={(e) => onArchitectureChange(dModel, nLayers, Number(e.target.value))}
              className="w-full px-2 py-1 bg-[#07090E] border border-[#1E293B] rounded text-[11px] font-mono tabular-nums text-slate-200 focus:border-cyan-500/60 focus:outline-none"
            >
              <option value={2}>2</option>
              <option value={4}>4</option>
            </select>
          </div>
        </div>
      </div>

      {/* Section 3: Prompt & Thought Generation */}
      <div className="p-3.5 space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-xs font-semibold text-slate-200 tracking-tight">
            03. Prompt & Sampling
          </h2>
          <button
            type="button"
            onClick={onResetPrompt}
            className="text-[11px] text-slate-400 hover:text-cyan-300 transition-colors"
          >
            Reset Prompt
          </button>
        </div>

        <textarea
          value={promptText}
          onChange={(e) => onPromptChange(e.target.value)}
          rows={3}
          spellCheck={false}
          placeholder="Enter seed text for the model to continue..."
          className="w-full px-2.5 py-2 bg-[#07090E] border border-[#1E293B] focus:border-cyan-500/60 focus:outline-none rounded text-xs font-mono leading-relaxed text-cyan-100 resize-y"
        />

        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={onStepToken}
            className="flex-1 flex items-center justify-center gap-1.5 px-3 py-1.5 bg-[#121926] border border-[#1E293B] hover:border-cyan-500/60 rounded text-xs font-medium text-slate-100 transition-colors whitespace-nowrap"
          >
            <StepForward className="w-3.5 h-3.5 text-cyan-400" />
            <span>Step Token</span>
          </button>
          <button
            type="button"
            onClick={onToggleGenerate}
            className={`flex-1 flex items-center justify-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium transition-colors whitespace-nowrap ${
              isGenerating
                ? 'bg-amber-500/20 text-amber-200 border border-amber-500/50'
                : 'bg-[#121926] border border-[#1E293B] hover:border-cyan-500/60 text-slate-100'
            }`}
          >
            <Sparkles className="w-3.5 h-3.5 text-amber-400" />
            <span>{isGenerating ? 'Stop Stream' : 'Stream Tokens'}</span>
          </button>
        </div>

        <div className="space-y-2 pt-0.5">
          <div className="space-y-1">
            <div className="flex items-center justify-between text-[11px]">
              <span className="text-slate-400">Temperature</span>
              <span className="font-mono tabular-nums text-slate-200">{temperature.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min={0.15}
              max={2.0}
              step={0.05}
              value={temperature}
              onChange={(e) => onTemperatureChange(parseFloat(e.target.value))}
              className="w-full accent-cyan-400 cursor-pointer h-1.5 bg-[#131A26] rounded"
            />
          </div>

          <div className="space-y-1">
            <div className="flex items-center justify-between text-[11px]">
              <span className="text-slate-400">Top-K Pool</span>
              <span className="font-mono tabular-nums text-slate-200">{topK} tokens</span>
            </div>
            <input
              type="range"
              min={1}
              max={Math.max(2, Math.min(24, vocabSize))}
              step={1}
              value={Math.min(topK, Math.max(2, vocabSize))}
              onChange={(e) => onTopKChange(parseInt(e.target.value, 10))}
              className="w-full accent-amber-400 cursor-pointer h-1.5 bg-[#131A26] rounded"
            />
          </div>
        </div>
      </div>
    </aside>
  );
};
