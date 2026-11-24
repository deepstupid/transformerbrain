import { CharTokenizer, encodeText } from './tokenizer';
import { computeStabilizedPca3D, cosineSimilarity, PcaBasis3D, vectorL2Norm } from './pca';

export interface TransformerConfig {
  vocabSize: number;
  dModel: number;
  nLayers: number;
  nHeads: number;
  seqLen: number;
  learningRate: number;
}

export interface LayerThought {
  /** Residual vector after block (seqLen x dModel) */
  residualOut: Float32Array;
  /** Attention output before residual add (seqLen x dModel) */
  attnDelta: Float32Array;
  /** MLP output before residual add (seqLen x dModel) */
  mlpDelta: Float32Array;
  /** Attention probability matrix per head: nHeads * (seqLen x seqLen) */
  attnProbs: Float32Array;
  /** Pre-softmax scaled QK^T scores per head: nHeads * (seqLen x seqLen) */
  attnScores: Float32Array;
  /** MLP hidden GELU activations (seqLen x dMlp) */
  mlpHidden: Float32Array;
}

export interface TokenCandidate {
  tokenId: number;
  token: string;
  logit: number;
  prob: number;
  inTopK: boolean;
}

export interface ThoughtFrame {
  step: number;
  tokens: string[];
  tokenIds: number[];
  dModel: number;
  dMlp: number;
  nLayers: number;
  nHeads: number;
  seqLenActual: number;
  /** Initial embedding + positional vectors (seqLenActual x dModel) */
  inputEmbeds: Float32Array;
  layers: LayerThought[];
  /** Final LayerNorm output before logits (seqLenActual x dModel) */
  finalHidden: Float32Array;
  /** Logits for next token (vocabSize) */
  logits: Float32Array;
  /** Softmax probabilities for next token (vocabSize) */
  probs: Float32Array;
  entropy: number;
  perplexity: number;
  topCandidates: TokenCandidate[];
  /** Stabilized 3D PCA coordinates for all vocab embeddings (vocabSize x 3) */
  embeddingCoords3D: Float32Array;
  /** Historical 3D PCA trails per vocab token: trailLen x vocabSize x 3 */
  embeddingTrails: Float32Array[];
  /** Vocab token embedding vectors (vocabSize x dModel) */
  embeddingVectors: Float32Array;
  /** Embedding L2 norms (vocabSize) */
  embeddingNorms: Float32Array;
  /** Prompt token trajectory in same PCA basis: (nLayers + 1) * seqLenActual * 3 */
  residualTrajectory3D: Float32Array;
}

export interface TrainStepMetrics {
  step: number;
  loss: number;
  perplexity: number;
  gradNorm: number;
}

interface ParamTensor {
  data: Float32Array;
  grad: Float32Array;
  m: Float32Array;
  v: Float32Array;
  weightDecay: boolean;
}

interface BlockParams {
  ln1Gamma: ParamTensor;
  ln1Beta: ParamTensor;
  wq: ParamTensor;
  bq: ParamTensor;
  wk: ParamTensor;
  bk: ParamTensor;
  wv: ParamTensor;
  bv: ParamTensor;
  wo: ParamTensor;
  bo: ParamTensor;
  ln2Gamma: ParamTensor;
  ln2Beta: ParamTensor;
  w1: ParamTensor;
  b1: ParamTensor;
  w2: ParamTensor;
  b2: ParamTensor;
}

interface BlockCache {
  xIn: Float32Array;
  ln1Out: Float32Array;
  ln1Rstd: Float32Array;
  ln1Xhat: Float32Array;
  q: Float32Array;
  k: Float32Array;
  v: Float32Array;
  attnProbs: Float32Array;
  attnOutPreProj: Float32Array;
  attnProj: Float32Array;
  xMid: Float32Array;
  ln2Out: Float32Array;
  ln2Rstd: Float32Array;
  ln2Xhat: Float32Array;
  mlpPreGelu: Float32Array;
  mlpPostGelu: Float32Array;
  mlpProj: Float32Array;
  xOut: Float32Array;
}

const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI);
const GELU_COEFF = 0.044715;
const LN_EPS = 1e-5;

function createParam(size: number, initScale: number, weightDecay = true, fillValue?: number): ParamTensor {
  const data = new Float32Array(size);
  if (fillValue !== undefined) {
    data.fill(fillValue);
  } else if (initScale > 0) {
    // Box-Muller deterministic-friendly normal init
    for (let i = 0; i < size; i += 2) {
      const u1 = Math.max(1e-7, Math.random());
      const u2 = Math.random();
      const mag = initScale * Math.sqrt(-2 * Math.log(u1));
      data[i] = mag * Math.cos(2 * Math.PI * u2);
      if (i + 1 < size) {
        data[i + 1] = mag * Math.sin(2 * Math.PI * u2);
      }
    }
  }
  return {
    data,
    grad: new Float32Array(size),
    m: new Float32Array(size),
    v: new Float32Array(size),
    weightDecay,
  };
}

export class TinyTransformer {
  public config: TransformerConfig;
  public dMlp: number;
  public dHead: number;
  public paramCount: number;
  public step = 0;
  public lossHistory: TrainStepMetrics[] = [];

  private wte: ParamTensor;
  private wpe: ParamTensor;
  private blocks: BlockParams[];
  private lnfGamma: ParamTensor;
  private lnfBeta: ParamTensor;
  private lmBias: ParamTensor;
  private allParams: ParamTensor[] = [];

  private pcaBasis: PcaBasis3D | null = null;
  private embeddingTrails: Float32Array[] = [];
  private cursor = 0;

  constructor(config: TransformerConfig) {
    this.config = { ...config };
    this.dMlp = config.dModel * 4;
    this.dHead = Math.floor(config.dModel / config.nHeads);

    const std = 0.08;
    const projStd = std / Math.sqrt(2 * config.nLayers);

    this.wte = createParam(config.vocabSize * config.dModel, 0.22, true);
    this.wpe = createParam(config.seqLen * config.dModel, 0.08, true);
    this.allParams.push(this.wte, this.wpe);

    this.blocks = [];
    for (let l = 0; l < config.nLayers; l++) {
      const b: BlockParams = {
        ln1Gamma: createParam(config.dModel, 0, false, 1),
        ln1Beta: createParam(config.dModel, 0, false, 0),
        wq: createParam(config.dModel * config.dModel, std, true),
        bq: createParam(config.dModel, 0, false, 0),
        wk: createParam(config.dModel * config.dModel, std, true),
        bk: createParam(config.dModel, 0, false, 0),
        wv: createParam(config.dModel * config.dModel, std, true),
        bv: createParam(config.dModel, 0, false, 0),
        wo: createParam(config.dModel * config.dModel, projStd, true),
        bo: createParam(config.dModel, 0, false, 0),
        ln2Gamma: createParam(config.dModel, 0, false, 1),
        ln2Beta: createParam(config.dModel, 0, false, 0),
        w1: createParam(config.dModel * this.dMlp, std, true),
        b1: createParam(this.dMlp, 0, false, 0),
        w2: createParam(this.dMlp * config.dModel, projStd, true),
        b2: createParam(config.dModel, 0, false, 0),
      };
      this.blocks.push(b);
      this.allParams.push(
        b.ln1Gamma,
        b.ln1Beta,
        b.wq,
        b.bq,
        b.wk,
        b.bk,
        b.wv,
        b.bv,
        b.wo,
        b.bo,
        b.ln2Gamma,
        b.ln2Beta,
        b.w1,
        b.b1,
        b.w2,
        b.b2
      );
    }

    this.lnfGamma = createParam(config.dModel, 0, false, 1);
    this.lnfBeta = createParam(config.dModel, 0, false, 0);
    this.lmBias = createParam(config.vocabSize, 0, false, 0);
    this.allParams.push(this.lnfGamma, this.lnfBeta, this.lmBias);

    this.paramCount = this.allParams.reduce((acc, p) => acc + p.data.length, 0);
    this.recordEmbeddingSnapshot();
  }

  public recordEmbeddingSnapshot(): void {
    const { basis, coords } = computeStabilizedPca3D(
      this.wte.data,
      this.config.vocabSize,
      this.config.dModel,
      this.pcaBasis
    );
    this.pcaBasis = basis;
    this.embeddingTrails.push(coords);
    if (this.embeddingTrails.length > 24) {
      this.embeddingTrails.shift();
    }
  }

  public trainStep(corpusIds: number[], batchSize = 4): TrainStepMetrics {
    const { seqLen, dModel, nLayers, nHeads, vocabSize } = this.config;
    if (corpusIds.length < 4) {
      return { step: this.step, loss: 0, perplexity: 1, gradNorm: 0 };
    }

    const effT = Math.min(seqLen, corpusIds.length - 1);
    const B = batchSize;
    const N = B * effT;

    const inputs = new Int32Array(N);
    const targets = new Int32Array(N);

    const maxStart = Math.max(1, corpusIds.length - effT - 1);
    for (let b = 0; b < B; b++) {
      // Strided + jittered sampling covers the corpus smoothly
      const start = (this.cursor + b * Math.max(1, Math.floor(maxStart / B)) + Math.floor(Math.random() * 5)) % maxStart;
      for (let t = 0; t < effT; t++) {
        inputs[b * effT + t] = corpusIds[start + t];
        targets[b * effT + t] = corpusIds[start + t + 1];
      }
    }
    this.cursor = (this.cursor + 7) % maxStart;

    // Zero grads
    for (const p of this.allParams) {
      p.grad.fill(0);
    }

    // Forward pass
    const x0 = new Float32Array(N * dModel);
    for (let b = 0; b < B; b++) {
      for (let t = 0; t < effT; t++) {
        const idx = b * effT + t;
        const tokId = inputs[idx];
        const xOff = idx * dModel;
        const wteOff = tokId * dModel;
        const wpeOff = t * dModel;
        for (let d = 0; d < dModel; d++) {
          x0[xOff + d] = this.wte.data[wteOff + d] + this.wpe.data[wpeOff + d];
        }
      }
    }

    const caches: BlockCache[] = [];
    let currentX = x0;

    for (let l = 0; l < nLayers; l++) {
      const blk = this.blocks[l];
      const ln1 = layerNormForward(currentX, blk.ln1Gamma.data, blk.ln1Beta.data, N, dModel);

      const q = linearForward(ln1.out, blk.wq.data, blk.bq.data, N, dModel, dModel);
      const k = linearForward(ln1.out, blk.wk.data, blk.bk.data, N, dModel, dModel);
      const v = linearForward(ln1.out, blk.wv.data, blk.bv.data, N, dModel, dModel);

      const attnProbs = new Float32Array(B * nHeads * effT * effT);
      const attnOutPreProj = new Float32Array(N * dModel);
      const scale = 1 / Math.sqrt(this.dHead);

      for (let b = 0; b < B; b++) {
        for (let h = 0; h < nHeads; h++) {
          const hOff = h * this.dHead;
          const pBase = (b * nHeads + h) * effT * effT;
          for (let i = 0; i < effT; i++) {
            const qRow = (b * effT + i) * dModel + hOff;
            let maxScore = -Infinity;
            // Scores for j <= i
            for (let j = 0; j <= i; j++) {
              const kRow = (b * effT + j) * dModel + hOff;
              let dot = 0;
              for (let c = 0; c < this.dHead; c++) {
                dot += q[qRow + c] * k[kRow + c];
              }
              const s = dot * scale;
              attnProbs[pBase + i * effT + j] = s;
              if (s > maxScore) maxScore = s;
            }
            let sumExp = 0;
            for (let j = 0; j <= i; j++) {
              const e = Math.exp(attnProbs[pBase + i * effT + j] - maxScore);
              attnProbs[pBase + i * effT + j] = e;
              sumExp += e;
            }
            const invSum = 1 / Math.max(1e-9, sumExp);
            for (let j = 0; j <= i; j++) {
              attnProbs[pBase + i * effT + j] *= invSum;
            }
            // Weighted sum of V
            const outRow = (b * effT + i) * dModel + hOff;
            for (let j = 0; j <= i; j++) {
              const p = attnProbs[pBase + i * effT + j];
              const vRow = (b * effT + j) * dModel + hOff;
              for (let c = 0; c < this.dHead; c++) {
                attnOutPreProj[outRow + c] += p * v[vRow + c];
              }
            }
          }
        }
      }

      const attnProj = linearForward(attnOutPreProj, blk.wo.data, blk.bo.data, N, dModel, dModel);
      const xMid = new Float32Array(N * dModel);
      for (let i = 0; i < xMid.length; i++) {
        xMid[i] = currentX[i] + attnProj[i];
      }

      const ln2 = layerNormForward(xMid, blk.ln2Gamma.data, blk.ln2Beta.data, N, dModel);
      const mlpPreGelu = linearForward(ln2.out, blk.w1.data, blk.b1.data, N, dModel, this.dMlp);
      const mlpPostGelu = new Float32Array(mlpPreGelu.length);
      for (let i = 0; i < mlpPreGelu.length; i++) {
        const u = mlpPreGelu[i];
        const inner = SQRT_2_OVER_PI * (u + GELU_COEFF * u * u * u);
        mlpPostGelu[i] = 0.5 * u * (1 + Math.tanh(inner));
      }
      const mlpProj = linearForward(mlpPostGelu, blk.w2.data, blk.b2.data, N, this.dMlp, dModel);
      const xOut = new Float32Array(N * dModel);
      for (let i = 0; i < xOut.length; i++) {
        xOut[i] = xMid[i] + mlpProj[i];
      }

      caches.push({
        xIn: currentX,
        ln1Out: ln1.out,
        ln1Rstd: ln1.rstd,
        ln1Xhat: ln1.xhat,
        q,
        k,
        v,
        attnProbs,
        attnOutPreProj,
        attnProj,
        xMid,
        ln2Out: ln2.out,
        ln2Rstd: ln2.rstd,
        ln2Xhat: ln2.xhat,
        mlpPreGelu,
        mlpPostGelu,
        mlpProj,
        xOut,
      });

      currentX = xOut;
    }

    // Final LayerNorm + Weight-tied LM head
    const lnf = layerNormForward(currentX, this.lnfGamma.data, this.lnfBeta.data, N, dModel);
    const logits = new Float32Array(N * vocabSize);
    let totalLoss = 0;
    const dLogits = new Float32Array(N * vocabSize);
    const invN = 1 / N;

    for (let n = 0; n < N; n++) {
      const hOff = n * dModel;
      const lOff = n * vocabSize;
      let maxL = -Infinity;
      for (let vIdx = 0; vIdx < vocabSize; vIdx++) {
        const wOff = vIdx * dModel;
        let dot = this.lmBias.data[vIdx];
        for (let d = 0; d < dModel; d++) {
          dot += lnf.out[hOff + d] * this.wte.data[wOff + d];
        }
        logits[lOff + vIdx] = dot;
        if (dot > maxL) maxL = dot;
      }

      let sumExp = 0;
      for (let vIdx = 0; vIdx < vocabSize; vIdx++) {
        const e = Math.exp(logits[lOff + vIdx] - maxL);
        dLogits[lOff + vIdx] = e;
        sumExp += e;
      }

      const invSum = 1 / Math.max(1e-9, sumExp);
      const tgt = targets[n];
      const tgtProb = Math.max(1e-9, dLogits[lOff + tgt] * invSum);
      totalLoss -= Math.log(tgtProb);

      for (let vIdx = 0; vIdx < vocabSize; vIdx++) {
        const prob = dLogits[lOff + vIdx] * invSum;
        dLogits[lOff + vIdx] = (prob - (vIdx === tgt ? 1 : 0)) * invN;
      }
    }

    const meanLoss = totalLoss * invN;

    // Backward LM head (tied weights with WTE)
    const dLnfOut = new Float32Array(N * dModel);
    for (let n = 0; n < N; n++) {
      const hOff = n * dModel;
      const lOff = n * vocabSize;
      for (let vIdx = 0; vIdx < vocabSize; vIdx++) {
        const dl = dLogits[lOff + vIdx];
        this.lmBias.grad[vIdx] += dl;
        const wOff = vIdx * dModel;
        for (let d = 0; d < dModel; d++) {
          dLnfOut[hOff + d] += dl * this.wte.data[wOff + d];
          this.wte.grad[wOff + d] += dl * lnf.out[hOff + d];
        }
      }
    }

    let dX = layerNormBackward(
      dLnfOut,
      lnf.xhat,
      lnf.rstd,
      this.lnfGamma.data,
      this.lnfGamma.grad,
      this.lnfBeta.grad,
      N,
      dModel
    );

    // Backward through blocks
    for (let l = nLayers - 1; l >= 0; l--) {
      const blk = this.blocks[l];
      const c = caches[l];

      // Residual split at xOut = xMid + mlpProj
      const dMlpProj = dX;
      const dMlpPostGelu = linearBackward(
        dMlpProj,
        c.mlpPostGelu,
        blk.w2.data,
        blk.w2.grad,
        blk.b2.grad,
        N,
        this.dMlp,
        dModel
      );

      const dMlpPreGelu = new Float32Array(dMlpPostGelu.length);
      for (let i = 0; i < dMlpPostGelu.length; i++) {
        const u = c.mlpPreGelu[i];
        const inner = SQRT_2_OVER_PI * (u + GELU_COEFF * u * u * u);
        const t = Math.tanh(inner);
        const dgelu = 0.5 * (1 + t) + 0.5 * u * (1 - t * t) * SQRT_2_OVER_PI * (1 + 3 * GELU_COEFF * u * u);
        dMlpPreGelu[i] = dMlpPostGelu[i] * dgelu;
      }

      const dLn2Out = linearBackward(
        dMlpPreGelu,
        c.ln2Out,
        blk.w1.data,
        blk.w1.grad,
        blk.b1.grad,
        N,
        dModel,
        this.dMlp
      );

      const dXMidFromMlp = layerNormBackward(
        dLn2Out,
        c.ln2Xhat,
        c.ln2Rstd,
        blk.ln2Gamma.data,
        blk.ln2Gamma.grad,
        blk.ln2Beta.grad,
        N,
        dModel
      );

      const dXMid = new Float32Array(N * dModel);
      for (let i = 0; i < dXMid.length; i++) {
        dXMid[i] = dX[i] + dXMidFromMlp[i];
      }

      // Residual split at xMid = xIn + attnProj
      const dAttnOutPreProj = linearBackward(
        dXMid,
        c.attnOutPreProj,
        blk.wo.data,
        blk.wo.grad,
        blk.bo.grad,
        N,
        dModel,
        dModel
      );

      const dQ = new Float32Array(N * dModel);
      const dK = new Float32Array(N * dModel);
      const dV = new Float32Array(N * dModel);
      const scale = 1 / Math.sqrt(this.dHead);

      for (let b = 0; b < B; b++) {
        for (let h = 0; h < nHeads; h++) {
          const hOff = h * this.dHead;
          const pBase = (b * nHeads + h) * effT * effT;
          for (let i = 0; i < effT; i++) {
            const outRow = (b * effT + i) * dModel + hOff;
            const qRow = outRow;
            let rowDot = 0;
            const dProbsRow = new Float32Array(i + 1);

            for (let j = 0; j <= i; j++) {
              const p = c.attnProbs[pBase + i * effT + j];
              const vRow = (b * effT + j) * dModel + hOff;
              let dp = 0;
              for (let ch = 0; ch < this.dHead; ch++) {
                const dout = dAttnOutPreProj[outRow + ch];
                dV[vRow + ch] += p * dout;
                dp += dout * c.v[vRow + ch];
              }
              dProbsRow[j] = dp;
              rowDot += p * dp;
            }

            for (let j = 0; j <= i; j++) {
              const p = c.attnProbs[pBase + i * effT + j];
              const ds = p * (dProbsRow[j] - rowDot) * scale;
              const kRow = (b * effT + j) * dModel + hOff;
              for (let ch = 0; ch < this.dHead; ch++) {
                dQ[qRow + ch] += ds * c.k[kRow + ch];
                dK[kRow + ch] += ds * c.q[qRow + ch];
              }
            }
          }
        }
      }

      const dLn1FromQ = linearBackward(dQ, c.ln1Out, blk.wq.data, blk.wq.grad, blk.bq.grad, N, dModel, dModel);
      const dLn1FromK = linearBackward(dK, c.ln1Out, blk.wk.data, blk.wk.grad, blk.bk.grad, N, dModel, dModel);
      const dLn1FromV = linearBackward(dV, c.ln1Out, blk.wv.data, blk.wv.grad, blk.bv.grad, N, dModel, dModel);

      const dLn1Out = new Float32Array(N * dModel);
      for (let i = 0; i < dLn1Out.length; i++) {
        dLn1Out[i] = dLn1FromQ[i] + dLn1FromK[i] + dLn1FromV[i];
      }

      const dXInFromAttn = layerNormBackward(
        dLn1Out,
        c.ln1Xhat,
        c.ln1Rstd,
        blk.ln1Gamma.data,
        blk.ln1Gamma.grad,
        blk.ln1Beta.grad,
        N,
        dModel
      );

      const nextDX = new Float32Array(N * dModel);
      for (let i = 0; i < nextDX.length; i++) {
        nextDX[i] = dXMid[i] + dXInFromAttn[i];
      }
      dX = nextDX;
    }

    // Backward embeddings
    for (let b = 0; b < B; b++) {
      for (let t = 0; t < effT; t++) {
        const idx = b * effT + t;
        const tokId = inputs[idx];
        const xOff = idx * dModel;
        const wteOff = tokId * dModel;
        const wpeOff = t * dModel;
        for (let d = 0; d < dModel; d++) {
          const g = dX[xOff + d];
          this.wte.grad[wteOff + d] += g;
          this.wpe.grad[wpeOff + d] += g;
        }
      }
    }

    // Global gradient norm + clipping
    let gradSq = 0;
    for (const p of this.allParams) {
      for (let i = 0; i < p.grad.length; i++) {
        gradSq += p.grad[i] * p.grad[i];
      }
    }
    const gradNorm = Math.sqrt(gradSq);
    const maxNorm = 1.0;
    const clipScale = gradNorm > maxNorm ? maxNorm / gradNorm : 1.0;

    // AdamW step
    this.step += 1;
    const lr = this.config.learningRate;
    const beta1 = 0.9;
    const beta2 = 0.999;
    const eps = 1e-8;
    const wd = 0.01;
    const bc1 = 1 - Math.pow(beta1, this.step);
    const bc2 = 1 - Math.pow(beta2, this.step);

    for (const p of this.allParams) {
      const data = p.data;
      const grad = p.grad;
      const m = p.m;
      const v = p.v;
      const applyWd = p.weightDecay;
      for (let i = 0; i < data.length; i++) {
        const g = grad[i] * clipScale;
        const mNew = beta1 * m[i] + (1 - beta1) * g;
        const vNew = beta2 * v[i] + (1 - beta2) * g * g;
        m[i] = mNew;
        v[i] = vNew;
        const mHat = mNew / bc1;
        const vHat = vNew / bc2;
        if (applyWd) {
          data[i] -= lr * wd * data[i];
        }
        data[i] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
      }
    }

    if (this.step % 5 === 0) {
      this.recordEmbeddingSnapshot();
    }

    const metric: TrainStepMetrics = {
      step: this.step,
      loss: meanLoss,
      perplexity: Math.exp(Math.min(12, meanLoss)),
      gradNorm,
    };
    this.lossHistory.push(metric);
    if (this.lossHistory.length > 240) {
      this.lossHistory.shift();
    }
    return metric;
  }

  public captureThoughtFrame(
    promptText: string,
    tokenizer: CharTokenizer,
    temperature = 0.85,
    topK = 8
  ): ThoughtFrame {
    const { seqLen, dModel, nLayers, nHeads, vocabSize } = this.config;
    let tokenIds = encodeText(tokenizer, promptText);
    if (tokenIds.length === 0) {
      tokenIds = [0];
    }
    if (tokenIds.length > seqLen) {
      tokenIds = tokenIds.slice(tokenIds.length - seqLen);
    }
    const T = tokenIds.length;
    const tokens = tokenIds.map((id) => tokenizer.idToToken[id] ?? '?');

    const inputEmbeds = new Float32Array(T * dModel);
    for (let t = 0; t < T; t++) {
      const id = tokenIds[t];
      const xOff = t * dModel;
      const wteOff = id * dModel;
      const wpeOff = t * dModel;
      for (let d = 0; d < dModel; d++) {
        inputEmbeds[xOff + d] = this.wte.data[wteOff + d] + this.wpe.data[wpeOff + d];
      }
    }

    let currentX = inputEmbeds;
    const layers: LayerThought[] = [];

    for (let l = 0; l < nLayers; l++) {
      const blk = this.blocks[l];
      const ln1 = layerNormForward(currentX, blk.ln1Gamma.data, blk.ln1Beta.data, T, dModel);
      const q = linearForward(ln1.out, blk.wq.data, blk.bq.data, T, dModel, dModel);
      const k = linearForward(ln1.out, blk.wk.data, blk.bk.data, T, dModel, dModel);
      const v = linearForward(ln1.out, blk.wv.data, blk.bv.data, T, dModel, dModel);

      const attnProbs = new Float32Array(nHeads * T * T);
      const attnScores = new Float32Array(nHeads * T * T);
      const attnOutPreProj = new Float32Array(T * dModel);
      const scale = 1 / Math.sqrt(this.dHead);

      for (let h = 0; h < nHeads; h++) {
        const hOff = h * this.dHead;
        const pBase = h * T * T;
        for (let i = 0; i < T; i++) {
          const qRow = i * dModel + hOff;
          let maxScore = -Infinity;
          for (let j = 0; j <= i; j++) {
            const kRow = j * dModel + hOff;
            let dot = 0;
            for (let c = 0; c < this.dHead; c++) {
              dot += q[qRow + c] * k[kRow + c];
            }
            const s = dot * scale;
            attnScores[pBase + i * T + j] = s;
            attnProbs[pBase + i * T + j] = s;
            if (s > maxScore) maxScore = s;
          }
          let sumExp = 0;
          for (let j = 0; j <= i; j++) {
            const e = Math.exp(attnProbs[pBase + i * T + j] - maxScore);
            attnProbs[pBase + i * T + j] = e;
            sumExp += e;
          }
          const invSum = 1 / Math.max(1e-9, sumExp);
          for (let j = 0; j <= i; j++) {
            attnProbs[pBase + i * T + j] *= invSum;
          }
          const outRow = i * dModel + hOff;
          for (let j = 0; j <= i; j++) {
            const p = attnProbs[pBase + i * T + j];
            const vRow = j * dModel + hOff;
            for (let c = 0; c < this.dHead; c++) {
              attnOutPreProj[outRow + c] += p * v[vRow + c];
            }
          }
        }
      }

      const attnDelta = linearForward(attnOutPreProj, blk.wo.data, blk.bo.data, T, dModel, dModel);
      const xMid = new Float32Array(T * dModel);
      for (let i = 0; i < xMid.length; i++) {
        xMid[i] = currentX[i] + attnDelta[i];
      }

      const ln2 = layerNormForward(xMid, blk.ln2Gamma.data, blk.ln2Beta.data, T, dModel);
      const mlpPre = linearForward(ln2.out, blk.w1.data, blk.b1.data, T, dModel, this.dMlp);
      const mlpHidden = new Float32Array(mlpPre.length);
      for (let i = 0; i < mlpPre.length; i++) {
        const u = mlpPre[i];
        const inner = SQRT_2_OVER_PI * (u + GELU_COEFF * u * u * u);
        mlpHidden[i] = 0.5 * u * (1 + Math.tanh(inner));
      }
      const mlpDelta = linearForward(mlpHidden, blk.w2.data, blk.b2.data, T, this.dMlp, dModel);
      const residualOut = new Float32Array(T * dModel);
      for (let i = 0; i < residualOut.length; i++) {
        residualOut[i] = xMid[i] + mlpDelta[i];
      }

      layers.push({
        residualOut,
        attnDelta,
        mlpDelta,
        attnProbs,
        attnScores,
        mlpHidden,
      });
      currentX = residualOut;
    }

    const lnf = layerNormForward(currentX, this.lnfGamma.data, this.lnfBeta.data, T, dModel);
    const lastHiddenOff = (T - 1) * dModel;
    const logits = new Float32Array(vocabSize);
    const invTemp = 1 / Math.max(0.05, temperature);

    let maxLogit = -Infinity;
    for (let vIdx = 0; vIdx < vocabSize; vIdx++) {
      const wOff = vIdx * dModel;
      let dot = this.lmBias.data[vIdx];
      for (let d = 0; d < dModel; d++) {
        dot += lnf.out[lastHiddenOff + d] * this.wte.data[wOff + d];
      }
      logits[vIdx] = dot;
      if (dot * invTemp > maxLogit) maxLogit = dot * invTemp;
    }

    const probs = new Float32Array(vocabSize);
    let sumExp = 0;
    for (let vIdx = 0; vIdx < vocabSize; vIdx++) {
      const e = Math.exp(logits[vIdx] * invTemp - maxLogit);
      probs[vIdx] = e;
      sumExp += e;
    }
    const invSum = 1 / Math.max(1e-9, sumExp);
    let entropy = 0;
    for (let vIdx = 0; vIdx < vocabSize; vIdx++) {
      probs[vIdx] *= invSum;
      const p = probs[vIdx];
      if (p > 1e-8) {
        entropy -= p * Math.log2(p);
      }
    }

    const sortedIds = Array.from({ length: vocabSize }, (_, idx) => idx).sort(
      (a, b) => probs[b] - probs[a]
    );
    const clampedK = Math.max(1, Math.min(vocabSize, topK));
    const topCandidates: TokenCandidate[] = sortedIds.map((id, rank) => ({
      tokenId: id,
      token: tokenizer.idToToken[id] ?? '?',
      logit: logits[id],
      prob: probs[id],
      inTopK: rank < clampedK,
    }));

    // Update stabilized PCA projection
    const { basis, coords: embeddingCoords3D } = computeStabilizedPca3D(
      this.wte.data,
      vocabSize,
      dModel,
      this.pcaBasis
    );
    this.pcaBasis = basis;

    const embeddingNorms = new Float32Array(vocabSize);
    for (let vIdx = 0; vIdx < vocabSize; vIdx++) {
      embeddingNorms[vIdx] = vectorL2Norm(this.wte.data, vIdx * dModel, dModel);
    }

    // Project prompt token residual trajectories onto same PCA basis
    const residualTrajectory3D = new Float32Array((nLayers + 1) * T * 3);
    const stageVectors: Float32Array[] = [inputEmbeds, ...layers.map((ly) => ly.residualOut)];
    let maxTrajR = 1e-6;
    for (let s = 0; s <= nLayers; s++) {
      const vecs = stageVectors[s];
      for (let t = 0; t < T; t++) {
        const vOff = t * dModel;
        let x = 0;
        let y = 0;
        let z = 0;
        for (let d = 0; d < dModel; d++) {
          const val = vecs[vOff + d] - basis.mean[d];
          x += val * basis.axes[0][d];
          y += val * basis.axes[1][d];
          z += val * basis.axes[2][d];
        }
        const outOff = (s * T + t) * 3;
        residualTrajectory3D[outOff] = x;
        residualTrajectory3D[outOff + 1] = y;
        residualTrajectory3D[outOff + 2] = z;
        const r = Math.hypot(x, y, z);
        if (r > maxTrajR) maxTrajR = r;
      }
    }
    const trajScale = 1 / maxTrajR;
    for (let i = 0; i < residualTrajectory3D.length; i++) {
      residualTrajectory3D[i] *= trajScale;
    }

    return {
      step: this.step,
      tokens,
      tokenIds,
      dModel,
      dMlp: this.dMlp,
      nLayers,
      nHeads,
      seqLenActual: T,
      inputEmbeds,
      layers,
      finalHidden: lnf.out,
      logits,
      probs,
      entropy,
      perplexity: Math.pow(2, entropy),
      topCandidates,
      embeddingCoords3D,
      embeddingTrails: this.embeddingTrails,
      embeddingVectors: new Float32Array(this.wte.data),
      embeddingNorms,
      residualTrajectory3D,
    };
  }

  public sampleFromFrame(frame: ThoughtFrame, topK: number): TokenCandidate {
    const pool = frame.topCandidates.slice(0, Math.max(1, Math.min(frame.topCandidates.length, topK)));
    let total = 0;
    for (const c of pool) total += c.prob;
    let r = Math.random() * Math.max(1e-9, total);
    for (const c of pool) {
      r -= c.prob;
      if (r <= 0) return c;
    }
    return pool[0];
  }

  public getTopCosineNeighbors(tokenId: number, k = 6): Array<{ tokenId: number; similarity: number }> {
    const { vocabSize, dModel } = this.config;
    const sims: Array<{ tokenId: number; similarity: number }> = [];
    for (let i = 0; i < vocabSize; i++) {
      if (i === tokenId) continue;
      sims.push({
        tokenId: i,
        similarity: cosineSimilarity(this.wte.data, tokenId * dModel, this.wte.data, i * dModel, dModel),
      });
    }
    sims.sort((a, b) => b.similarity - a.similarity);
    return sims.slice(0, k);
  }
}

function layerNormForward(
  x: Float32Array,
  gamma: Float32Array,
  beta: Float32Array,
  N: number,
  D: number
): { out: Float32Array; rstd: Float32Array; xhat: Float32Array } {
  const out = new Float32Array(N * D);
  const rstd = new Float32Array(N);
  const xhat = new Float32Array(N * D);
  const invD = 1 / D;

  for (let n = 0; n < N; n++) {
    const off = n * D;
    let mean = 0;
    for (let d = 0; d < D; d++) mean += x[off + d];
    mean *= invD;

    let varSum = 0;
    for (let d = 0; d < D; d++) {
      const diff = x[off + d] - mean;
      varSum += diff * diff;
    }
    const r = 1 / Math.sqrt(varSum * invD + LN_EPS);
    rstd[n] = r;

    for (let d = 0; d < D; d++) {
      const xh = (x[off + d] - mean) * r;
      xhat[off + d] = xh;
      out[off + d] = gamma[d] * xh + beta[d];
    }
  }
  return { out, rstd, xhat };
}

function layerNormBackward(
  dOut: Float32Array,
  xhat: Float32Array,
  rstd: Float32Array,
  gamma: Float32Array,
  dGamma: Float32Array,
  dBeta: Float32Array,
  N: number,
  D: number
): Float32Array {
  const dX = new Float32Array(N * D);
  const invD = 1 / D;

  for (let n = 0; n < N; n++) {
    const off = n * D;
    const r = rstd[n];
    let sumG = 0;
    let sumGXhat = 0;

    for (let d = 0; d < D; d++) {
      const dy = dOut[off + d];
      const xh = xhat[off + d];
      dGamma[d] += dy * xh;
      dBeta[d] += dy;
      const g = dy * gamma[d];
      sumG += g;
      sumGXhat += g * xh;
    }

    for (let d = 0; d < D; d++) {
      const g = dOut[off + d] * gamma[d];
      const xh = xhat[off + d];
      dX[off + d] = r * (g - invD * (sumG + xh * sumGXhat));
    }
  }
  return dX;
}

function linearForward(
  x: Float32Array,
  w: Float32Array,
  b: Float32Array,
  N: number,
  I: number,
  O: number
): Float32Array {
  const out = new Float32Array(N * O);
  for (let n = 0; n < N; n++) {
    const xOff = n * I;
    const yOff = n * O;
    for (let o = 0; o < O; o++) {
      let sum = b[o];
      for (let i = 0; i < I; i++) {
        sum += x[xOff + i] * w[i * O + o];
      }
      out[yOff + o] = sum;
    }
  }
  return out;
}

function linearBackward(
  dOut: Float32Array,
  x: Float32Array,
  w: Float32Array,
  dW: Float32Array,
  dB: Float32Array,
  N: number,
  I: number,
  O: number
): Float32Array {
  const dX = new Float32Array(N * I);
  for (let n = 0; n < N; n++) {
    const xOff = n * I;
    const yOff = n * O;
    for (let o = 0; o < O; o++) {
      const dy = dOut[yOff + o];
      dB[o] += dy;
      for (let i = 0; i < I; i++) {
        dX[xOff + i] += dy * w[i * O + o];
        dW[i * O + o] += x[xOff + i] * dy;
      }
    }
  }
  return dX;
}
