import { ThoughtFrame } from '../ml/transformer';
import { CharTokenizer, formatTokenLabel } from '../ml/tokenizer';
import { cosineSimilarity, vectorL2Norm } from '../ml/pca';
import { CameraState, projectWorldPoint } from './camera';
import { observatoryHeatColor, signedActivationColor } from './colormaps';
import { NodeInstance, QuadInstance, SegmentInstance } from './ObservatoryGL';
import { HitTarget, OverlayLabel, SelectionTarget } from '../types/observatory';

export interface SceneBuildResult {
  quads: QuadInstance[];
  segments: SegmentInstance[];
  nodes: NodeInstance[];
  labels: OverlayLabel[];
  hits: HitTarget[];
}

export function buildEmbeddingsScene(
  frame: ThoughtFrame,
  tokenizer: CharTokenizer,
  cam: CameraState,
  width: number,
  height: number,
  selection: SelectionTarget | null,
  smoothedCoords: Float32Array,
  reducedMotion: boolean
): SceneBuildResult {
  const quads: QuadInstance[] = [];
  const segments: SegmentInstance[] = [];
  const nodes: NodeInstance[] = [];
  const labels: OverlayLabel[] = [];
  const hits: HitTarget[] = [];

  const V = tokenizer.vocabSize;

  // Coordinate reference axes
  const axisLen = 1.15;
  segments.push(
    { x1: -axisLen, y1: 0, z1: 0, x2: axisLen, y2: 0, z2: 0, thicknessPx: 1.0, r: 0.18, g: 0.25, b: 0.36, a: 0.55, pulseSpeed: 0, pulsePhase: 0 },
    { x1: 0, y1: -axisLen, z1: 0, x2: 0, y2: axisLen, z2: 0, thicknessPx: 1.0, r: 0.18, g: 0.25, b: 0.36, a: 0.55, pulseSpeed: 0, pulsePhase: 0 }
  );
  if (cam.mode === '3d') {
    segments.push({
      x1: 0, y1: 0, z1: -axisLen, x2: 0, y2: 0, z2: axisLen,
      thicknessPx: 1.0, r: 0.18, g: 0.25, b: 0.36, a: 0.45, pulseSpeed: 0, pulsePhase: 0,
    });
  }

  // Axis end labels
  const ax1 = projectWorldPoint(axisLen + 0.08, 0, 0, cam, width, height);
  const ax2 = projectWorldPoint(0, axisLen + 0.08, 0, cam, width, height);
  if (ax1.visible) labels.push({ sx: ax1.sx, sy: ax1.sy, text: 'PC1', color: '#64748b', fontSize: 10 });
  if (ax2.visible) labels.push({ sx: ax2.sx, sy: ax2.sy, text: 'PC2', color: '#64748b', fontSize: 10 });
  if (cam.mode === '3d') {
    const ax3 = projectWorldPoint(0, 0, axisLen + 0.08, cam, width, height);
    if (ax3.visible) labels.push({ sx: ax3.sx, sy: ax3.sy, text: 'PC3', color: '#64748b', fontSize: 10 });
  }

  // Reference ring in XY plane
  const ringSegs = 36;
  for (let i = 0; i < ringSegs; i++) {
    const a0 = (i / ringSegs) * Math.PI * 2;
    const a1 = ((i + 1) / ringSegs) * Math.PI * 2;
    segments.push({
      x1: Math.cos(a0) * 0.85, y1: Math.sin(a0) * 0.85, z1: 0,
      x2: Math.cos(a1) * 0.85, y2: Math.sin(a1) * 0.85, z2: 0,
      thicknessPx: 0.8, r: 0.14, g: 0.2, b: 0.3, a: 0.35, pulseSpeed: 0, pulsePhase: 0,
    });
  }

  // Determine selected tokenId
  let selectedTokenId: number | null = null;
  if (selection?.kind === 'token') selectedTokenId = selection.tokenId;
  else if (selection?.kind === 'attentionCell') selectedTokenId = frame.tokenIds[selection.qIdx] ?? null;
  else if (selection?.kind === 'pathwayNode') selectedTokenId = frame.tokenIds[selection.tokenPos] ?? null;

  const neighborMap = new Map<number, number>();
  if (selectedTokenId !== null && selectedTokenId < V) {
    const sims: Array<{ id: number; sim: number }> = [];
    for (let i = 0; i < V; i++) {
      if (i === selectedTokenId) continue;
      const sim = cosineSimilarity(
        frame.embeddingVectors, selectedTokenId * frame.dModel,
        frame.embeddingVectors, i * frame.dModel,
        frame.dModel
      );
      sims.push({ id: i, sim });
    }
    sims.sort((a, b) => b.sim - a.sim);
    for (const item of sims.slice(0, 6)) {
      neighborMap.set(item.id, item.sim);
    }
  }

  // Training trails for vocab embeddings
  const trails = frame.embeddingTrails;
  if (trails.length >= 2) {
    for (let v = 0; v < V; v++) {
      const isFocus = selectedTokenId === null || selectedTokenId === v || neighborMap.has(v);
      const trailAlpha = isFocus ? 0.26 : 0.06;
      for (let t = 0; t < trails.length - 1; t++) {
        const cA = trails[t];
        const cB = trails[t + 1];
        if (cA.length < V * 3 || cB.length < V * 3) continue;
        const zA = cam.mode === '3d' ? cA[v * 3 + 2] : 0;
        const zB = cam.mode === '3d' ? cB[v * 3 + 2] : 0;
        const fade = (t + 1) / trails.length;
        segments.push({
          x1: cA[v * 3], y1: cA[v * 3 + 1], z1: zA,
          x2: cB[v * 3], y2: cB[v * 3 + 1], z2: zB,
          thicknessPx: 1.1,
          r: 0.08, g: 0.65, b: 0.85, a: trailAlpha * fade,
          pulseSpeed: 0, pulsePhase: 0,
        });
      }
    }
  }

  // Prompt sequence trajectory segments
  const promptSet = new Set(frame.tokenIds);
  for (let i = 0; i < frame.tokenIds.length - 1; i++) {
    const idA = frame.tokenIds[i];
    const idB = frame.tokenIds[i + 1];
    if (idA === idB) continue;
    const x1 = smoothedCoords[idA * 3];
    const y1 = smoothedCoords[idA * 3 + 1];
    const z1 = cam.mode === '3d' ? smoothedCoords[idA * 3 + 2] : 0;
    const x2 = smoothedCoords[idB * 3];
    const y2 = smoothedCoords[idB * 3 + 1];
    const z2 = cam.mode === '3d' ? smoothedCoords[idB * 3 + 2] : 0;
    const dimmed = selectedTokenId !== null && selectedTokenId !== idA && selectedTokenId !== idB;
    segments.push({
      x1, y1, z1, x2, y2, z2,
      thicknessPx: dimmed ? 1.0 : 1.7,
      r: 0.03, g: 0.78, b: 0.92, a: dimmed ? 0.14 : 0.45,
      pulseSpeed: reducedMotion || dimmed ? 0 : 0.45,
      pulsePhase: i * 0.17,
    });
  }

  // Cosine-similarity neighbor filaments from selected token
  if (selectedTokenId !== null && selectedTokenId < V) {
    const sx = smoothedCoords[selectedTokenId * 3];
    const sy = smoothedCoords[selectedTokenId * 3 + 1];
    const sz = cam.mode === '3d' ? smoothedCoords[selectedTokenId * 3 + 2] : 0;
    neighborMap.forEach((sim, nbId) => {
      const nx = smoothedCoords[nbId * 3];
      const ny = smoothedCoords[nbId * 3 + 1];
      const nz = cam.mode === '3d' ? smoothedCoords[nbId * 3 + 2] : 0;
      const strength = Math.max(0.15, (sim + 1) * 0.5);
      segments.push({
        x1: sx, y1: sy, z1: sz,
        x2: nx, y2: ny, z2: nz,
        thicknessPx: 1.5 + strength * 1.8,
        r: 0.96, g: 0.62, b: 0.06, a: 0.55 + strength * 0.4,
        pulseSpeed: reducedMotion ? 0 : 0.75 * strength,
        pulsePhase: nbId * 0.13,
      });
    });
  }

  // Draw all vocabulary token nodes
  for (let v = 0; v < V; v++) {
    const x = smoothedCoords[v * 3];
    const y = smoothedCoords[v * 3 + 1];
    const z = cam.mode === '3d' ? smoothedCoords[v * 3 + 2] : 0;

    const isSelected = selectedTokenId === v;
    const isNeighbor = neighborMap.has(v);
    const inPrompt = promptSet.has(v);
    const freq = tokenizer.frequencies[v] ?? 0.01;

    let alpha = 0.88;
    if (selectedTokenId !== null && !isSelected && !isNeighbor) {
      alpha = 0.22;
    }

    let r = 0.35;
    let g = 0.65;
    let b = 0.82;
    let ring = 0.0;
    let rad = 5.0 + Math.min(6.0, Math.sqrt(freq) * 20.0);

    if (isSelected) {
      r = 0.98; g = 0.72; b = 0.12;
      ring = 1.0;
      rad += 3.5;
    } else if (isNeighbor) {
      r = 0.04; g = 0.85; b = 0.95;
      ring = 0.65;
      rad += 1.5;
    } else if (inPrompt) {
      r = 0.06; g = 0.74; b = 0.88;
      ring = 0.35;
    }

    nodes.push({ x, y, z, radiusPx: rad, r, g, b, a: alpha, ring });

    const proj = projectWorldPoint(x, y, z, cam, width, height);
    if (proj.visible) {
      const tokLabel = formatTokenLabel(tokenizer.idToToken[v] ?? '?');
      const simVal = neighborMap.get(v);
      labels.push({
        sx: proj.sx,
        sy: proj.sy - rad - 5,
        text: isNeighbor && simVal !== undefined ? `${tokLabel} (${simVal.toFixed(2)})` : tokLabel,
        color: isSelected ? '#fcd34d' : isNeighbor ? '#67e8f9' : inPrompt ? '#e2e8f0' : '#94a3b8',
        fontSize: isSelected ? 12 : 10,
        alpha,
      });

      hits.push({
        sx: proj.sx,
        sy: proj.sy,
        radiusPx: Math.max(12, rad + 6),
        target: { kind: 'token', tokenId: v },
        hoverTitle: `Token ${tokLabel}`,
        hoverDetail: `Norm ${frame.embeddingNorms[v].toFixed(2)} · Freq ${(freq * 100).toFixed(1)}%`,
      });
    }
  }

  return { quads, segments, nodes, labels, hits };
}

export function buildAttentionScene(
  frame: ThoughtFrame,
  _tokenizer: CharTokenizer,
  cam: CameraState,
  width: number,
  height: number,
  selection: SelectionTarget | null,
  activeLayer: number,
  activeHead: number, // -1 for head average
  reducedMotion: boolean
): SceneBuildResult {
  const quads: QuadInstance[] = [];
  const segments: SegmentInstance[] = [];
  const nodes: NodeInstance[] = [];
  const labels: OverlayLabel[] = [];
  const hits: HitTarget[] = [];

  const T = frame.seqLenActual;
  const layer = Math.max(0, Math.min(frame.nLayers - 1, activeLayer));
  const ly = frame.layers[layer];
  if (!ly || T === 0) return { quads, segments, nodes, labels, hits };

  // Compute active matrix A[q, k] and raw score S[q, k]
  const matrix = new Float32Array(T * T);
  const scores = new Float32Array(T * T);
  if (activeHead < 0) {
    const invH = 1 / frame.nHeads;
    for (let h = 0; h < frame.nHeads; h++) {
      const base = h * T * T;
      for (let i = 0; i < T * T; i++) {
        matrix[i] += ly.attnProbs[base + i] * invH;
        scores[i] += ly.attnScores[base + i] * invH;
      }
    }
  } else {
    const h = Math.min(frame.nHeads - 1, activeHead);
    const base = h * T * T;
    for (let i = 0; i < T * T; i++) {
      matrix[i] = ly.attnProbs[base + i];
      scores[i] = ly.attnScores[base + i];
    }
  }

  // Layout heatmap on left/center [-1.05, 0.25] x [-0.65, 0.65]
  // and curved token flow arcs on right [0.42, 1.05]
  const matSize = 1.24;
  const matCenterX = -0.34;
  const matCenterY = 0.0;
  const cellStep = matSize / T;
  const cellSize = cellStep * 0.88;

  let selQ: number | null = null;
  let selK: number | null = null;
  let selTokenId: number | null = null;
  if (selection?.kind === 'attentionCell') {
    selQ = selection.qIdx;
    selK = selection.kIdx;
  } else if (selection?.kind === 'pathwayNode') {
    selQ = selection.tokenPos;
  } else if (selection?.kind === 'token') {
    selTokenId = selection.tokenId;
    if (selection.tokenPos !== undefined) selQ = selection.tokenPos;
  }

  // Heatmap cells
  for (let q = 0; q < T; q++) {
    for (let k = 0; k < T; k++) {
      const wx = matCenterX - matSize * 0.5 + (k + 0.5) * cellStep;
      const wy = matCenterY + matSize * 0.5 - (q + 0.5) * cellStep;

      const isCausal = k <= q;
      const w = isCausal ? matrix[q * T + k] : 0;
      const [cr, cg, cb] = isCausal ? observatoryHeatColor(Math.pow(w, 0.65)) : [0.05, 0.07, 0.1];

      const isExactCell = selQ === q && selK === k;
      const isRowMatch = selQ === q || (selTokenId !== null && frame.tokenIds[q] === selTokenId);
      const isColMatch = selK === k || (selTokenId !== null && frame.tokenIds[k] === selTokenId);

      let alpha = isCausal ? 0.94 : 0.22;
      if ((selQ !== null || selTokenId !== null) && !isRowMatch && !isColMatch) {
        alpha *= 0.34;
      }

      quads.push({
        x: wx,
        y: wy,
        z: 0,
        w: cellSize,
        h: cellSize,
        r: cr,
        g: cg,
        b: cb,
        a: alpha,
        border: isExactCell ? 1.0 : isRowMatch && isCausal && w > 0.2 ? 0.35 : 0.0,
      });

      if (isCausal) {
        const p = projectWorldPoint(wx, wy, 0, cam, width, height);
        if (p.visible) {
          const hitR = Math.max(5, cellStep * Math.min(width, height) * 0.2 * cam.zoom);
          hits.push({
            sx: p.sx,
            sy: p.sy,
            radiusPx: hitR,
            halfW: hitR,
            halfH: hitR,
            target: {
              kind: 'attentionCell',
              layer,
              head: activeHead,
              qIdx: q,
              kIdx: k,
            },
            hoverTitle: `Q[${q}] '${formatTokenLabel(frame.tokens[q])}' ← K[${k}] '${formatTokenLabel(frame.tokens[k])}'`,
            hoverDetail: `Weight ${(w * 100).toFixed(1)}% · Q·K/√d ${scores[q * T + k].toFixed(2)}`,
          });
        }
      }
    }
  }

  // Row (Query) & Col (Key) labels around heatmap
  for (let i = 0; i < T; i++) {
    const tok = formatTokenLabel(frame.tokens[i]);
    const isQActive = selQ === i || (selTokenId !== null && frame.tokenIds[i] === selTokenId);
    const isKActive = selK === i || (selTokenId !== null && frame.tokenIds[i] === selTokenId);

    const rowPos = projectWorldPoint(
      matCenterX - matSize * 0.5 - cellStep * 0.75,
      matCenterY + matSize * 0.5 - (i + 0.5) * cellStep,
      0, cam, width, height
    );
    if (rowPos.visible) {
      labels.push({
        sx: rowPos.sx,
        sy: rowPos.sy,
        text: tok,
        color: isQActive ? '#67e8f9' : '#94a3b8',
        fontSize: T > 20 ? 9 : 11,
      });
    }

    const colPos = projectWorldPoint(
      matCenterX - matSize * 0.5 + (i + 0.5) * cellStep,
      matCenterY + matSize * 0.5 + cellStep * 0.7,
      0, cam, width, height
    );
    if (colPos.visible) {
      labels.push({
        sx: colPos.sx,
        sy: colPos.sy,
        text: tok,
        color: isKActive ? '#fcd34d' : '#94a3b8',
        fontSize: T > 20 ? 9 : 11,
      });
    }
  }

  // Axis headers
  const qHdr = projectWorldPoint(matCenterX - matSize * 0.5 - cellStep * 1.9, matCenterY, 0, cam, width, height);
  const kHdr = projectWorldPoint(matCenterX, matCenterY + matSize * 0.5 + cellStep * 1.65, 0, cam, width, height);
  if (qHdr.visible) labels.push({ sx: qHdr.sx, sy: qHdr.sy, text: 'QUERY', color: '#64748b', fontSize: 10 });
  if (kHdr.visible) labels.push({ sx: kHdr.sx, sy: kHdr.sy, text: 'KEY', color: '#64748b', fontSize: 10 });

  // Vertical Token Flow Arcs on the right side [0.56 .. 0.98]
  const arcXKey = 0.52;
  const arcXQuery = 0.98;
  const arcHeader = projectWorldPoint((arcXKey + arcXQuery) * 0.5, matCenterY + matSize * 0.5 + cellStep * 1.65, 0, cam, width, height);
  if (arcHeader.visible) {
    labels.push({ sx: arcHeader.sx, sy: arcHeader.sy, text: 'FLOW ARCS (K → Q)', color: '#64748b', fontSize: 10 });
  }

  for (let q = 0; q < T; q++) {
    const qy = matCenterY + matSize * 0.5 - (q + 0.5) * cellStep;
    const isFocusQ = selQ === q;
    nodes.push({
      x: arcXQuery,
      y: qy,
      z: 0,
      radiusPx: isFocusQ ? 6 : 4,
      r: isFocusQ ? 0.06 : 0.25,
      g: isFocusQ ? 0.85 : 0.65,
      b: isFocusQ ? 0.95 : 0.85,
      a: selQ !== null && !isFocusQ ? 0.35 : 0.9,
      ring: isFocusQ ? 0.8 : 0.0,
    });
    nodes.push({
      x: arcXKey,
      y: qy,
      z: 0,
      radiusPx: selK === q ? 6 : 3.5,
      r: selK === q ? 0.98 : 0.45,
      g: selK === q ? 0.7 : 0.52,
      b: selK === q ? 0.15 : 0.62,
      a: selK !== null && selK !== q ? 0.35 : 0.85,
      ring: selK === q ? 0.8 : 0.0,
    });

    const pQ = projectWorldPoint(arcXQuery + 0.07, qy, 0, cam, width, height);
    if (pQ.visible) {
      labels.push({
        sx: pQ.sx,
        sy: pQ.sy,
        text: formatTokenLabel(frame.tokens[q]),
        color: isFocusQ ? '#67e8f9' : '#cbd5e1',
        fontSize: T > 20 ? 9 : 10,
      });
    }

    for (let k = 0; k <= q; k++) {
      const w = matrix[q * T + k];
      if (w < 0.05 && !(selQ === q && selK === k)) continue;
      const ky = matCenterY + matSize * 0.5 - (k + 0.5) * cellStep;

      const isPairSelected = selQ === q && selK === k;
      const isRowSelected = selQ === q && selK === null;
      const dimmed = (selQ !== null && selQ !== q) || (selK !== null && selK !== k);

      const alpha = isPairSelected
        ? 0.98
        : isRowSelected
        ? Math.min(0.95, 0.25 + w * 0.9)
        : dimmed
        ? 0.06
        : Math.min(0.85, w * 0.95);

      const [r, g, b] = isPairSelected ? [0.98, 0.72, 0.12] : observatoryHeatColor(0.35 + w * 0.65);

      // 4-segment smooth curve bow
      const bow = 0.09 * Math.sin(((q - k) / Math.max(1, T)) * Math.PI);
      const pts: Array<[number, number]> = [];
      const steps = 4;
      for (let s = 0; s <= steps; s++) {
        const t = s / steps;
        const bx = arcXKey + (arcXQuery - arcXKey) * t;
        const by = ky + (qy - ky) * t + Math.sin(t * Math.PI) * bow;
        pts.push([bx, by]);
      }
      for (let s = 0; s < steps; s++) {
        segments.push({
          x1: pts[s][0], y1: pts[s][1], z1: 0,
          x2: pts[s + 1][0], y2: pts[s + 1][1], z2: 0,
          thicknessPx: isPairSelected ? 2.6 : 0.9 + w * 2.0,
          r, g, b, a: alpha,
          pulseSpeed: reducedMotion || dimmed ? 0 : 0.35 + w * 0.85,
          pulsePhase: (q * 0.17 + k * 0.11 + s * 0.05) % 1,
        });
      }
    }
  }

  return { quads, segments, nodes, labels, hits };
}

export function buildPathwaysScene(
  frame: ThoughtFrame,
  _tokenizer: CharTokenizer,
  cam: CameraState,
  width: number,
  height: number,
  selection: SelectionTarget | null,
  activeLayer: number,
  reducedMotion: boolean
): SceneBuildResult {
  const quads: QuadInstance[] = [];
  const segments: SegmentInstance[] = [];
  const nodes: NodeInstance[] = [];
  const labels: OverlayLabel[] = [];
  const hits: HitTarget[] = [];

  const T = frame.seqLenActual;
  const L = frame.nLayers;
  if (T === 0) return { quads, segments, nodes, labels, hits };

  // Stages: Stage 0 = Input (E+P), Stages 1..L = Block l Residual Out
  const numStages = L + 1;
  const leftX = -0.95;
  const rightX = 0.25;
  const topY = 0.62;
  const botY = -0.62;

  const stageX = (s: number) =>
    numStages === 1 ? (leftX + rightX) * 0.5 : leftX + (s / (numStages - 1)) * (rightX - leftX);
  const tokY = (t: number) =>
    T === 1 ? 0 : topY - (t / (T - 1)) * (topY - botY);

  let selStage: number | null = null;
  let selTokPos: number | null = null;
  let selNeuronLayer: number | null = null;
  let selNeuronIdx: number | null = null;

  if (selection?.kind === 'pathwayNode') {
    selStage = selection.stage;
    selTokPos = selection.tokenPos;
  } else if (selection?.kind === 'mlpNeuron') {
    selNeuronLayer = selection.layer;
    selNeuronIdx = selection.neuronIdx;
    selTokPos = selection.tokenPos;
  } else if (selection?.kind === 'attentionCell') {
    selStage = selection.layer + 1;
    selTokPos = selection.qIdx;
  } else if (selection?.kind === 'token' && selection.tokenPos !== undefined) {
    selTokPos = selection.tokenPos;
  }

  // Stage column headers
  for (let s = 0; s < numStages; s++) {
    const sx = stageX(s);
    const p = projectWorldPoint(sx, topY + 0.14, 0, cam, width, height);
    if (p.visible) {
      labels.push({
        sx: p.sx,
        sy: p.sy,
        text: s === 0 ? 'EMBED' : `BLOCK ${s - 1}`,
        color: '#94a3b8',
        fontSize: 10,
      });
    }
  }

  // Draw cross-token attention pathways & residual stream links between stages
  for (let s = 0; s < L; s++) {
    const ly = frame.layers[s];
    const xA = stageX(s);
    const xB = stageX(s + 1);

    for (let q = 0; q < T; q++) {
      const yB = tokY(q);

      // Residual skip highway (direct token t -> t)
      const isUpstreamOfSel =
        selTokPos !== null && (selStage === null || s < selStage) && q === selTokPos;
      const isDownstreamOfSel =
        selTokPos !== null && selStage !== null && s >= selStage && q === selTokPos;
      const resActive = selTokPos === null || isUpstreamOfSel || isDownstreamOfSel;

      const deltaEnergy = Math.min(
        1,
        (vectorL2Norm(ly.attnDelta, q * frame.dModel, frame.dModel) +
          vectorL2Norm(ly.mlpDelta, q * frame.dModel, frame.dModel)) *
          0.35
      );

      segments.push({
        x1: xA, y1: yB, z1: 0,
        x2: xB, y2: yB, z2: 0,
        thicknessPx: resActive ? 1.5 + deltaEnergy * 1.4 : 0.8,
        r: 0.05, g: 0.72, b: 0.88,
        a: resActive ? 0.55 + deltaEnergy * 0.35 : 0.1,
        pulseSpeed: reducedMotion || !resActive ? 0 : 0.4 + deltaEnergy * 0.4,
        pulsePhase: (s * 0.23 + q * 0.09) % 1,
      });

      // Cross-token attention pathways (mean across heads)
      for (let k = 0; k < q; k++) {
        let meanW = 0;
        for (let h = 0; h < frame.nHeads; h++) {
          meanW += ly.attnProbs[h * T * T + q * T + k];
        }
        meanW /= frame.nHeads;
        if (meanW < 0.08) continue;

        const yA = tokY(k);
        const linkMatches =
          selTokPos === null ||
          (q === selTokPos && (selStage === null || s + 1 === selStage || s < selStage)) ||
          (k === selTokPos && (selStage === null || s === selStage));

        segments.push({
          x1: xA, y1: yA, z1: 0,
          x2: xB, y2: yB, z2: 0,
          thicknessPx: linkMatches ? 0.9 + meanW * 2.4 : 0.6,
          r: 0.96, g: 0.62, b: 0.08,
          a: linkMatches ? Math.min(0.9, 0.2 + meanW * 0.95) : 0.05,
          pulseSpeed: reducedMotion || !linkMatches ? 0 : 0.35 + meanW * 0.75,
          pulsePhase: (k * 0.13 + q * 0.07) % 1,
        });
      }
    }
  }

  // Draw residual stream nodes per stage & token
  for (let s = 0; s < numStages; s++) {
    const x = stageX(s);
    const vecs = s === 0 ? frame.inputEmbeds : frame.layers[s - 1].residualOut;

    for (let t = 0; t < T; t++) {
      const y = tokY(t);
      const norm = vectorL2Norm(vecs, t * frame.dModel, frame.dModel);
      const energy = Math.min(1, norm / (Math.sqrt(frame.dModel) * 1.1));
      const [cr, cg, cb] = observatoryHeatColor(0.2 + energy * 0.75);

      const isSelected = selStage === s && selTokPos === t;
      const isTokMatch = selTokPos === null || selTokPos === t;

      nodes.push({
        x, y, z: 0,
        radiusPx: isSelected ? 8.5 : 4.2 + energy * 2.5,
        r: isSelected ? 0.98 : cr,
        g: isSelected ? 0.74 : cg,
        b: isSelected ? 0.14 : cb,
        a: isTokMatch || isSelected ? 0.95 : 0.25,
        ring: isSelected ? 1.0 : t === T - 1 ? 0.35 : 0.0,
      });

      if (s === 0) {
        const pTok = projectWorldPoint(x - 0.1, y, 0, cam, width, height);
        if (pTok.visible) {
          labels.push({
            sx: pTok.sx,
            sy: pTok.sy,
            text: formatTokenLabel(frame.tokens[t]),
            color: selTokPos === t ? '#67e8f9' : '#cbd5e1',
            fontSize: T > 20 ? 9 : 11,
          });
        }
      }

      const pNode = projectWorldPoint(x, y, 0, cam, width, height);
      if (pNode.visible) {
        hits.push({
          sx: pNode.sx,
          sy: pNode.sy,
          radiusPx: 11,
          target: { kind: 'pathwayNode', stage: s, tokenPos: t },
          hoverTitle: `${s === 0 ? 'Embed' : `Block ${s - 1}`} · Pos ${t} '${formatTokenLabel(frame.tokens[t])}'`,
          hoverDetail: `Residual ||x||₂ = ${norm.toFixed(2)}`,
        });
      }
    }
  }

  // Right-side MLP Neuron Activation Matrix for activeLayer (T rows x 24 top-variance/first neurons)
  const inspectLayer = Math.max(0, Math.min(L - 1, selNeuronLayer ?? activeLayer));
  const mlp = frame.layers[inspectLayer].mlpHidden;
  const shownNeurons = Math.min(24, frame.dMlp);
  const mlpLeft = 0.45;
  const mlpRight = 1.06;
  const mlpW = mlpRight - mlpLeft;
  const nStep = mlpW / shownNeurons;
  const rowStep = (topY - botY) / Math.max(1, T - 1);

  const mlpHdr = projectWorldPoint((mlpLeft + mlpRight) * 0.5, topY + 0.14, 0, cam, width, height);
  if (mlpHdr.visible) {
    labels.push({
      sx: mlpHdr.sx,
      sy: mlpHdr.sy,
      text: `B${inspectLayer} MLP NEURONS`,
      color: '#94a3b8',
      fontSize: 10,
    });
  }

  // Connect selected or last token's final block node to active MLP neurons
  const activeTok = selTokPos ?? T - 1;
  const srcX = stageX(Math.min(numStages - 1, inspectLayer + 1));
  const srcY = tokY(activeTok);

  for (let t = 0; t < T; t++) {
    const wy = tokY(t);
    for (let n = 0; n < shownNeurons; n++) {
      const wx = mlpLeft + (n + 0.5) * nStep;
      const act = mlp[t * frame.dMlp + n];
      const [cr, cg, cb] = signedActivationColor(Math.tanh(act * 1.4));
      const isCellSelected = selNeuronLayer === inspectLayer && selNeuronIdx === n && selTokPos === t;
      const rowFocus = selTokPos === null || selTokPos === t;

      quads.push({
        x: wx,
        y: wy,
        z: 0,
        w: nStep * 0.82,
        h: Math.min(0.065, Math.max(0.025, rowStep * 0.72)),
        r: cr,
        g: cg,
        b: cb,
        a: rowFocus ? 0.92 : 0.25,
        border: isCellSelected ? 1.0 : 0.0,
      });

      if (t === activeTok && Math.abs(act) > 0.22) {
        segments.push({
          x1: srcX, y1: srcY, z1: 0,
          x2: wx, y2: wy, z2: 0,
          thicknessPx: isCellSelected ? 2.0 : 0.85,
          r: act > 0 ? 0.04 : 0.94,
          g: act > 0 ? 0.8 : 0.26,
          b: act > 0 ? 0.92 : 0.38,
          a: isCellSelected ? 0.95 : 0.35,
          pulseSpeed: reducedMotion ? 0 : 0.5,
          pulsePhase: n * 0.07,
        });
      }

      const pCell = projectWorldPoint(wx, wy, 0, cam, width, height);
      if (pCell.visible) {
        hits.push({
          sx: pCell.sx,
          sy: pCell.sy,
          radiusPx: 7,
          halfW: 6,
          halfH: 6,
          target: { kind: 'mlpNeuron', layer: inspectLayer, neuronIdx: n, tokenPos: t },
          hoverTitle: `Block ${inspectLayer} Neuron #${n} · Pos ${t}`,
          hoverDetail: `GELU activation = ${act.toFixed(3)}`,
        });
      }
    }
  }

  return { quads, segments, nodes, labels, hits };
}

export function buildLogitsScene(
  frame: ThoughtFrame,
  _tokenizer: CharTokenizer,
  cam: CameraState,
  width: number,
  height: number,
  selection: SelectionTarget | null,
  topK: number,
  reducedMotion: boolean
): SceneBuildResult {
  const quads: QuadInstance[] = [];
  const segments: SegmentInstance[] = [];
  const nodes: NodeInstance[] = [];
  const labels: OverlayLabel[] = [];
  const hits: HitTarget[] = [];

  const candidates = frame.topCandidates;
  const shown = candidates.slice(0, Math.min(28, candidates.length));
  if (shown.length === 0) return { quads, segments, nodes, labels, hits };

  const baseY = -0.62;
  const maxBarH = 1.24;
  const spanX = 1.9;
  const barStep = spanX / shown.length;
  const barW = barStep * 0.72;

  // Baseline axis + probability guide lines
  segments.push({
    x1: -spanX * 0.52, y1: baseY, z1: 0,
    x2: spanX * 0.52, y2: baseY, z2: 0,
    thicknessPx: 1.2, r: 0.22, g: 0.3, b: 0.42, a: 0.7, pulseSpeed: 0, pulsePhase: 0,
  });

  for (const pct of [0.25, 0.5, 0.75]) {
    const gy = baseY + pct * maxBarH;
    segments.push({
      x1: -spanX * 0.52, y1: gy, z1: 0,
      x2: spanX * 0.52, y2: gy, z2: 0,
      thicknessPx: 0.7, r: 0.16, g: 0.22, b: 0.32, a: 0.4, pulseSpeed: 0, pulsePhase: 0,
    });
    const pTick = projectWorldPoint(-spanX * 0.56, gy, 0, cam, width, height);
    if (pTick.visible) {
      labels.push({
        sx: pTick.sx,
        sy: pTick.sy,
        text: `${Math.round(pct * 100)}%`,
        color: '#64748b',
        fontSize: 9,
      });
    }
  }

  // Top-K cutoff marker line
  if (topK < shown.length) {
    const cutX = -spanX * 0.5 + topK * barStep;
    segments.push({
      x1: cutX, y1: baseY - 0.06, z1: 0,
      x2: cutX, y2: baseY + maxBarH + 0.08, z2: 0,
      thicknessPx: 1.2, r: 0.96, g: 0.62, b: 0.08, a: 0.75,
      pulseSpeed: reducedMotion ? 0 : 0.3, pulsePhase: 0,
    });
    const pCut = projectWorldPoint(cutX, baseY + maxBarH + 0.14, 0, cam, width, height);
    if (pCut.visible) {
      labels.push({
        sx: pCut.sx,
        sy: pCut.sy,
        text: `TOP-K=${topK}`,
        color: '#fbbf24',
        fontSize: 10,
      });
    }
  }

  const selectedTokenId = selection?.kind === 'token' ? selection.tokenId : null;
  const maxP = Math.max(0.15, shown[0]?.prob ?? 0.5);

  for (let i = 0; i < shown.length; i++) {
    const c = shown[i];
    const x = -spanX * 0.5 + (i + 0.5) * barStep;
    const normH = Math.max(0.02, (c.prob / Math.max(0.35, maxP * 1.08)) * maxBarH);
    const cy = baseY + normH * 0.5;
    const topY = baseY + normH;

    const isSelected = selectedTokenId === c.tokenId;
    const dimmed = selectedTokenId !== null && !isSelected;

    let cr = 0.25;
    let cg = 0.35;
    let cb = 0.48;
    if (isSelected) {
      cr = 0.98; cg = 0.72; cb = 0.12;
    } else if (i === 0) {
      cr = 0.06; cg = 0.84; cb = 0.94;
    } else if (c.inTopK) {
      [cr, cg, cb] = observatoryHeatColor(0.35 + (c.prob / maxP) * 0.45);
    }

    quads.push({
      x, y: cy, z: 0,
      w: barW, h: normH,
      r: cr, g: cg, b: cb,
      a: dimmed ? 0.25 : c.inTopK ? 0.92 : 0.45,
      border: isSelected ? 1.0 : i === 0 ? 0.4 : 0.0,
    });

    nodes.push({
      x, y: topY, z: 0,
      radiusPx: isSelected ? 6.5 : c.inTopK ? 4.0 : 2.5,
      r: cr, g: cg, b: cb,
      a: dimmed ? 0.3 : 0.98,
      ring: isSelected ? 1.0 : i === 0 ? 0.5 : 0.0,
    });

    const pTop = projectWorldPoint(x, topY + 0.07, 0, cam, width, height);
    const pBot = projectWorldPoint(x, baseY - 0.07, 0, cam, width, height);
    const tokLabel = formatTokenLabel(c.token);

    if (pBot.visible) {
      labels.push({
        sx: pBot.sx,
        sy: pBot.sy,
        text: tokLabel,
        color: isSelected ? '#fcd34d' : c.inTopK ? '#e2e8f0' : '#64748b',
        fontSize: shown.length > 22 ? 9 : 11,
      });
    }
    if (pTop.visible && (i < 6 || isSelected)) {
      labels.push({
        sx: pTop.sx,
        sy: pTop.sy,
        text: `${(c.prob * 100).toFixed(1)}%`,
        color: isSelected ? '#fcd34d' : '#67e8f9',
        fontSize: 9,
      });
    }

    const pHit = projectWorldPoint(x, cy, 0, cam, width, height);
    if (pHit.visible) {
      hits.push({
        sx: pHit.sx,
        sy: pHit.sy,
        radiusPx: 14,
        halfW: Math.max(10, barStep * Math.min(width, height) * 0.22 * cam.zoom),
        halfH: Math.max(18, normH * Math.min(width, height) * 0.25 * cam.zoom),
        target: { kind: 'token', tokenId: c.tokenId },
        hoverTitle: `Candidate '${tokLabel}' (Rank #${i + 1})`,
        hoverDetail: `P = ${(c.prob * 100).toFixed(2)}% · Logit ${c.logit.toFixed(2)}`,
      });
    }
  }

  return { quads, segments, nodes, labels, hits };
}
