import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Play, Pause } from 'lucide-react';
import { CORPUS_PRESETS } from './ml/corpora';
import { buildTokenizer, CharTokenizer, encodeText } from './ml/tokenizer';
import { ThoughtFrame, TinyTransformer, TrainStepMetrics } from './ml/transformer';
import { CameraState, createDefaultCamera } from './gl/camera';
import { LensKind, WebGLLensViewport } from './components/WebGLLensViewport';
import { ControlDeck } from './components/ControlDeck';
import { SelectionInspector } from './components/SelectionInspector';
import { BottomTelemetryTray } from './components/BottomTelemetryTray';
import { SelectionTarget, ViewMode } from './types/observatory';

interface EngineBundle {
  tokenizer: CharTokenizer;
  corpusIds: number[];
  model: TinyTransformer;
}

function buildEngineBundle(
  corpusText: string,
  dModel: number,
  nLayers: number,
  nHeads: number,
  learningRate: number,
  warmSteps = 70
): EngineBundle {
  const tokenizer = buildTokenizer(corpusText);
  const corpusIds = encodeText(tokenizer, corpusText);
  const model = new TinyTransformer({
    vocabSize: tokenizer.vocabSize,
    dModel,
    nLayers,
    nHeads,
    seqLen: 24,
    learningRate,
  });

  if (corpusIds.length >= 8 && warmSteps > 0) {
    for (let i = 0; i < warmSteps; i++) {
      model.trainStep(corpusIds, 4);
    }
  }

  return { tokenizer, corpusIds, model };
}

const MODE_ITEMS: Array<{ id: ViewMode; label: string }> = [
  { id: 'overview', label: 'Overview' },
  { id: 'embeddings', label: 'Embeddings' },
  { id: 'attention', label: 'Attention' },
  { id: 'pathways', label: 'Pathways' },
  { id: 'logits', label: 'Logits' },
];

export default function App() {
  const initialPreset = CORPUS_PRESETS[0];

  const [viewMode, setViewMode] = useState<ViewMode>('overview');
  const [selectedPresetId, setSelectedPresetId] = useState<string>(initialPreset.id);
  const [corpusDraft, setCorpusDraft] = useState<string>(initialPreset.text);
  const [appliedCorpus, setAppliedCorpus] = useState<string>(initialPreset.text);

  const [dModel, setDModel] = useState<number>(32);
  const [nLayers, setNLayers] = useState<number>(2);
  const [nHeads, setNHeads] = useState<number>(4);
  const [learningRate, setLearningRate] = useState<number>(0.0045);

  const [promptText, setPromptText] = useState<string>(initialPreset.defaultPrompt);
  const [temperature, setTemperature] = useState<number>(0.85);
  const [topK, setTopK] = useState<number>(8);

  const [activeLayer, setActiveLayer] = useState<number>(0);
  const [activeHead, setActiveHead] = useState<number>(-1);
  const [selection, setSelection] = useState<SelectionTarget | null>(null);

  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);

  const [cameras, setCameras] = useState<Record<LensKind, CameraState>>(() => ({
    embeddings: createDefaultCamera('3d'),
    attention: createDefaultCamera('2d'),
    pathways: createDefaultCamera('2d'),
    logits: createDefaultCamera('2d'),
  }));

  const engineRef = useRef<EngineBundle | null>(null);
  if (!engineRef.current) {
    engineRef.current = buildEngineBundle(
      initialPreset.text,
      dModel,
      nLayers,
      nHeads,
      learningRate,
      70
    );
  }

  const [frame, setFrame] = useState<ThoughtFrame>(() =>
    engineRef.current!.model.captureThoughtFrame(
      initialPreset.defaultPrompt,
      engineRef.current!.tokenizer,
      temperature,
      topK
    )
  );
  const [lossHistory, setLossHistory] = useState<TrainStepMetrics[]>(() => [
    ...engineRef.current!.model.lossHistory,
  ]);

  const refreshThoughtFrame = useCallback(
    (promptOverride?: string, tempOverride?: number, topKOverride?: number) => {
      const bundle = engineRef.current;
      if (!bundle) return;
      const nextFrame = bundle.model.captureThoughtFrame(
        promptOverride ?? promptText,
        bundle.tokenizer,
        tempOverride ?? temperature,
        topKOverride ?? topK
      );
      setFrame(nextFrame);
      setLossHistory([...bundle.model.lossHistory]);
    },
    [promptText, temperature, topK]
  );

  // Sync learning rate directly onto active model
  const handleLearningRateChange = (lr: number) => {
    setLearningRate(lr);
    if (engineRef.current) {
      engineRef.current.model.config.learningRate = lr;
    }
  };

  // Re-evaluate forward pass when prompt, temperature, or topK changes
  useEffect(() => {
    refreshThoughtFrame(promptText, temperature, topK);
  }, [promptText, temperature, topK, refreshThoughtFrame]);

  // Rebuild engine helper
  const rebuildEngine = useCallback(
    (
      nextCorpus: string,
      nextDModel: number,
      nextNLayers: number,
      nextNHeads: number,
      nextPrompt?: string,
      warmSteps = 70
    ) => {
      setIsTraining(false);
      setIsGenerating(false);
      setSelection(null);
      const bundle = buildEngineBundle(
        nextCorpus,
        nextDModel,
        nextNLayers,
        nextNHeads,
        learningRate,
        warmSteps
      );
      engineRef.current = bundle;
      setActiveLayer((prev) => Math.min(nextNLayers - 1, prev));
      setActiveHead((prev) => (prev >= nextNHeads ? -1 : prev));
      const usePrompt = nextPrompt ?? promptText;
      setFrame(bundle.model.captureThoughtFrame(usePrompt, bundle.tokenizer, temperature, topK));
      setLossHistory([...bundle.model.lossHistory]);
    },
    [learningRate, promptText, temperature, topK]
  );

  const handleSelectPreset = (presetId: string) => {
    const preset = CORPUS_PRESETS.find((p) => p.id === presetId);
    if (!preset) return;
    setSelectedPresetId(preset.id);
    setCorpusDraft(preset.text);
    setAppliedCorpus(preset.text);
    setPromptText(preset.defaultPrompt);
    rebuildEngine(preset.text, dModel, nLayers, nHeads, preset.defaultPrompt, 70);
  };

  const handleApplyCustomCorpus = () => {
    const clean = corpusDraft.trim().length >= 8 ? corpusDraft : appliedCorpus;
    setAppliedCorpus(clean);
    const seed = clean.slice(0, Math.min(16, clean.length));
    setPromptText(seed);
    rebuildEngine(clean, dModel, nLayers, nHeads, seed, 70);
  };

  const handleArchitectureChange = (nextD: number, nextL: number, nextH: number) => {
    setDModel(nextD);
    setNLayers(nextL);
    setNHeads(nextH);
    rebuildEngine(appliedCorpus, nextD, nextL, nextH, promptText, 70);
  };

  const handleResetModel = () => {
    rebuildEngine(appliedCorpus, dModel, nLayers, nHeads, promptText, 0);
  };

  const handleStepTrain = () => {
    const bundle = engineRef.current;
    if (!bundle) return;
    bundle.model.trainStep(bundle.corpusIds, 4);
    refreshThoughtFrame();
  };

  // Continuous training loop
  useEffect(() => {
    if (!isTraining) return;
    let rafId = 0;
    let tick = 0;

    const loop = () => {
      const bundle = engineRef.current;
      if (bundle) {
        for (let i = 0; i < 3; i++) {
          bundle.model.trainStep(bundle.corpusIds, 4);
        }
        tick += 1;
        if (tick % 2 === 0) {
          refreshThoughtFrame();
        }
      }
      rafId = requestAnimationFrame(loop);
    };

    rafId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafId);
  }, [isTraining, refreshThoughtFrame]);

  // Step / stream token generation
  const stepOneToken = useCallback(() => {
    const bundle = engineRef.current;
    if (!bundle) return;
    const currentFrame = bundle.model.captureThoughtFrame(
      promptText,
      bundle.tokenizer,
      temperature,
      topK
    );
    const nextCand = bundle.model.sampleFromFrame(currentFrame, topK);
    const nextPrompt = (promptText + nextCand.token).slice(-120);
    setPromptText(nextPrompt);
  }, [promptText, temperature, topK]);

  useEffect(() => {
    if (!isGenerating) return;
    const timer = window.setInterval(() => {
      stepOneToken();
    }, 220);
    return () => window.clearInterval(timer);
  }, [isGenerating, stepOneToken]);

  const handleResetPrompt = () => {
    setIsGenerating(false);
    const preset = CORPUS_PRESETS.find((p) => p.id === selectedPresetId);
    if (preset && appliedCorpus === preset.text) {
      setPromptText(preset.defaultPrompt);
    } else {
      setPromptText(appliedCorpus.slice(0, Math.min(16, appliedCorpus.length)));
    }
  };

  const updateCamera = (lens: LensKind, next: CameraState) => {
    setCameras((prev) => ({ ...prev, [lens]: next }));
  };

  const resetCamera = (lens: LensKind) => {
    setCameras((prev) => ({
      ...prev,
      [lens]: createDefaultCamera(lens === 'embeddings' ? prev.embeddings.mode : '2d'),
    }));
  };

  const corpusDirty = corpusDraft !== appliedCorpus;
  const bundle = engineRef.current!;

  const lensDefinitions = useMemo(
    () => ({
      embeddings: {
        title: 'Embedding Space',
        subtitle: `${cameras.embeddings.mode.toUpperCase()} PCA · V=${bundle.tokenizer.vocabSize} · Cosine Web`,
      },
      attention: {
        title: 'Attention Heatmaps & Flow',
        subtitle: `Layer ${activeLayer} · ${activeHead < 0 ? 'Head Avg' : `Head ${activeHead}`} · QKᵀ/√d`,
      },
      pathways: {
        title: 'Activation Pathways',
        subtitle: `Embed → ${nLayers} Blocks → MLP Neurons`,
      },
      logits: {
        title: 'Logit & Probability Landscape',
        subtitle: `T=${temperature.toFixed(2)} · Top-${topK} · H=${frame.entropy.toFixed(2)}b`,
      },
    }),
    [
      cameras.embeddings.mode,
      bundle.tokenizer.vocabSize,
      activeLayer,
      activeHead,
      nLayers,
      temperature,
      topK,
      frame.entropy,
    ]
  );

  return (
    <div className="min-h-screen lg:h-screen bg-[#07090E] text-slate-100 flex flex-col overflow-x-hidden lg:overflow-hidden">
      {/* Top Bar Contract: 3 zones separated by gap-8 */}
      <header className="flex items-center justify-between gap-8 px-5 py-3 bg-[#0A0E17] border-b border-[#1E293B] shrink-0">
        <a
          href="#top"
          onClick={(e) => {
            e.preventDefault();
            setViewMode('overview');
          }}
          className="text-base font-semibold tracking-tight text-slate-100 whitespace-nowrap shrink-0"
        >
          Tiny Transformer Observatory
        </a>

        <nav className="flex items-center gap-6 overflow-x-auto">
          {MODE_ITEMS.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => setViewMode(item.id)}
              className={`text-xs font-medium whitespace-nowrap shrink-0 pb-0.5 border-b transition-colors ${
                viewMode === item.id
                  ? 'text-cyan-300 border-cyan-400'
                  : 'text-slate-400 border-transparent hover:text-slate-100'
              }`}
            >
              {item.label}
            </button>
          ))}
        </nav>

        <div className="flex items-center shrink-0">
          <button
            type="button"
            onClick={() => setIsTraining((v) => !v)}
            className={`flex items-center gap-1.5 px-4 py-1.5 rounded-md text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
              isTraining
                ? 'bg-amber-500/20 text-amber-200 border border-amber-500/50 hover:bg-amber-500/30'
                : 'bg-cyan-500 text-slate-950 hover:bg-cyan-400'
            }`}
          >
            {isTraining ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
            <span>{isTraining ? 'Pause' : 'Train'}</span>
          </button>
        </div>
      </header>

      {/* Main Split Console */}
      <div className="flex-1 min-h-0 p-3 flex flex-col lg:flex-row gap-3">
        <ControlDeck
          selectedPresetId={selectedPresetId}
          onSelectPreset={handleSelectPreset}
          corpusText={corpusDraft}
          onCorpusTextChange={setCorpusDraft}
          onApplyCustomCorpus={handleApplyCustomCorpus}
          corpusDirty={corpusDirty}
          vocabSize={bundle.tokenizer.vocabSize}
          corpusLength={appliedCorpus.length}
          paramCount={bundle.model.paramCount}
          isTraining={isTraining}
          onToggleTraining={() => setIsTraining((v) => !v)}
          onStepTrain={handleStepTrain}
          onResetModel={handleResetModel}
          dModel={dModel}
          nLayers={nLayers}
          nHeads={nHeads}
          learningRate={learningRate}
          onArchitectureChange={handleArchitectureChange}
          onLearningRateChange={handleLearningRateChange}
          promptText={promptText}
          onPromptChange={setPromptText}
          temperature={temperature}
          onTemperatureChange={setTemperature}
          topK={topK}
          onTopKChange={setTopK}
          onStepToken={stepOneToken}
          isGenerating={isGenerating}
          onToggleGenerate={() => setIsGenerating((v) => !v)}
          onResetPrompt={handleResetPrompt}
        />

        {/* Right Stage: Inspector + WebGL2 Viewports + Bottom Telemetry */}
        <main className="flex-1 min-w-0 min-h-0 flex flex-col gap-3">
          <SelectionInspector
            selection={selection}
            onClear={() => setSelection(null)}
            onSelect={setSelection}
            frame={frame}
            model={bundle.model}
            tokenizer={bundle.tokenizer}
            activeLayer={activeLayer}
            activeHead={activeHead}
          />

          {/* WebGL2 Multi-Lens Grid or Single Focused Mode */}
          <div className="flex-1 min-h-[420px]">
            {viewMode === 'overview' ? (
              <div className="grid grid-cols-1 md:grid-cols-2 grid-rows-2 gap-3 w-full h-full">
                {(['embeddings', 'attention', 'pathways', 'logits'] as LensKind[]).map((lens) => (
                  <div key={lens} className="min-h-[240px] lg:min-h-0 h-full">
                    <WebGLLensViewport
                      lens={lens}
                      title={lensDefinitions[lens].title}
                      subtitle={lensDefinitions[lens].subtitle}
                      frame={frame}
                      tokenizer={bundle.tokenizer}
                      camera={cameras[lens]}
                      onCameraChange={(next) => updateCamera(lens, next)}
                      onResetCamera={() => resetCamera(lens)}
                      selection={selection}
                      onSelect={setSelection}
                      activeLayer={activeLayer}
                      activeHead={activeHead}
                      topK={topK}
                      compact
                      isExpanded={false}
                      onToggleExpand={() => setViewMode(lens)}
                      onLayerChange={setActiveLayer}
                      onHeadChange={setActiveHead}
                    />
                  </div>
                ))}
              </div>
            ) : (
              <div className="w-full h-full min-h-[420px]">
                <WebGLLensViewport
                  lens={viewMode}
                  title={lensDefinitions[viewMode].title}
                  subtitle={lensDefinitions[viewMode].subtitle}
                  frame={frame}
                  tokenizer={bundle.tokenizer}
                  camera={cameras[viewMode]}
                  onCameraChange={(next) => updateCamera(viewMode, next)}
                  onResetCamera={() => resetCamera(viewMode)}
                  selection={selection}
                  onSelect={setSelection}
                  activeLayer={activeLayer}
                  activeHead={activeHead}
                  topK={topK}
                  compact={false}
                  isExpanded
                  onToggleExpand={() => setViewMode('overview')}
                  onLayerChange={setActiveLayer}
                  onHeadChange={setActiveHead}
                />
              </div>
            )}
          </div>

          <BottomTelemetryTray
            lossHistory={lossHistory}
            frame={frame}
            selection={selection}
            onSelect={setSelection}
          />
        </main>
      </div>
    </div>
  );
}
