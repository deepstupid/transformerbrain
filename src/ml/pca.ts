export interface PcaBasis3D {
  mean: Float32Array;
  axes: [Float32Array, Float32Array, Float32Array];
}

/**
 * Computes a stabilized 3-component PCA basis via power iteration with
 * sign alignment against the previous basis so 2D/3D projections move smoothly
 * across training steps without sudden axis reflections.
 */
export function computeStabilizedPca3D(
  vectors: Float32Array,
  count: number,
  dim: number,
  prevBasis?: PcaBasis3D | null
): { basis: PcaBasis3D; coords: Float32Array } {
  const mean = new Float32Array(dim);
  if (count === 0 || dim === 0) {
    return {
      basis: {
        mean,
        axes: [new Float32Array(dim), new Float32Array(dim), new Float32Array(dim)],
      },
      coords: new Float32Array(count * 3),
    };
  }

  for (let i = 0; i < count; i++) {
    const off = i * dim;
    for (let d = 0; d < dim; d++) {
      mean[d] += vectors[off + d];
    }
  }
  const invCount = 1 / count;
  for (let d = 0; d < dim; d++) {
    mean[d] *= invCount;
  }

  // Centered matrix X (count x dim)
  const centered = new Float32Array(count * dim);
  for (let i = 0; i < count; i++) {
    const off = i * dim;
    for (let d = 0; d < dim; d++) {
      centered[off + d] = vectors[off + d] - mean[d];
    }
  }

  const axes: [Float32Array, Float32Array, Float32Array] = [
    new Float32Array(dim),
    new Float32Array(dim),
    new Float32Array(dim),
  ];

  // Work on deflated residual matrix for each component
  const work = new Float32Array(centered);
  const tmp = new Float32Array(count);

  for (let comp = 0; comp < 3; comp++) {
    const v = axes[comp];
    if (prevBasis && prevBasis.axes[comp].length === dim) {
      v.set(prevBasis.axes[comp]);
    } else {
      for (let d = 0; d < dim; d++) {
        v[d] = Math.sin((comp + 1) * 1.7 + d * 0.63) + (d === comp ? 1.0 : 0.0);
      }
    }

    // Normalize initial vector
    normalizeInPlace(v);

    // Power iterations: v <- X^T (X v)
    for (let iter = 0; iter < 18; iter++) {
      // tmp = work * v
      for (let i = 0; i < count; i++) {
        const off = i * dim;
        let dot = 0;
        for (let d = 0; d < dim; d++) {
          dot += work[off + d] * v[d];
        }
        tmp[i] = dot;
      }

      // nextV = work^T * tmp
      v.fill(0);
      for (let i = 0; i < count; i++) {
        const off = i * dim;
        const t = tmp[i];
        for (let d = 0; d < dim; d++) {
          v[d] += work[off + d] * t;
        }
      }

      // Gram-Schmidt against previous axes
      for (let prev = 0; prev < comp; prev++) {
        const pv = axes[prev];
        let proj = 0;
        for (let d = 0; d < dim; d++) proj += v[d] * pv[d];
        for (let d = 0; d < dim; d++) v[d] -= proj * pv[d];
      }

      const norm = normalizeInPlace(v);
      if (norm < 1e-8) {
        // Degenerate axis fallback orthogonal to prior axes
        v.fill(0);
        v[comp % dim] = 1;
        for (let prev = 0; prev < comp; prev++) {
          const pv = axes[prev];
          let proj = 0;
          for (let d = 0; d < dim; d++) proj += v[d] * pv[d];
          for (let d = 0; d < dim; d++) v[d] -= proj * pv[d];
        }
        normalizeInPlace(v);
        break;
      }
    }

    // Stabilize sign against previous basis
    if (prevBasis && prevBasis.axes[comp].length === dim) {
      let align = 0;
      for (let d = 0; d < dim; d++) align += v[d] * prevBasis.axes[comp][d];
      if (align < 0) {
        for (let d = 0; d < dim; d++) v[d] = -v[d];
      }
    }

    // Deflate work matrix
    for (let i = 0; i < count; i++) {
      const off = i * dim;
      let proj = 0;
      for (let d = 0; d < dim; d++) proj += work[off + d] * v[d];
      for (let d = 0; d < dim; d++) work[off + d] -= proj * v[d];
    }
  }

  // Project original centered vectors onto the 3 axes and normalize scale to [-1, 1] range
  const coords = new Float32Array(count * 3);
  let maxRadius = 1e-6;

  for (let i = 0; i < count; i++) {
    const off = i * dim;
    let x = 0;
    let y = 0;
    let z = 0;
    for (let d = 0; d < dim; d++) {
      const val = centered[off + d];
      x += val * axes[0][d];
      y += val * axes[1][d];
      z += val * axes[2][d];
    }
    coords[i * 3] = x;
    coords[i * 3 + 1] = y;
    coords[i * 3 + 2] = z;
    const r = Math.hypot(x, y, z);
    if (r > maxRadius) maxRadius = r;
  }

  const scale = 1 / maxRadius;
  for (let i = 0; i < coords.length; i++) {
    coords[i] *= scale;
  }

  return { basis: { mean, axes }, coords };
}

function normalizeInPlace(v: Float32Array): number {
  let sq = 0;
  for (let i = 0; i < v.length; i++) sq += v[i] * v[i];
  const n = Math.sqrt(sq);
  if (n > 1e-10) {
    const inv = 1 / n;
    for (let i = 0; i < v.length; i++) v[i] *= inv;
  }
  return n;
}

export function cosineSimilarity(
  a: Float32Array,
  aOffset: number,
  b: Float32Array,
  bOffset: number,
  dim: number
): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let d = 0; d < dim; d++) {
    const va = a[aOffset + d];
    const vb = b[bOffset + d];
    dot += va * vb;
    na += va * va;
    nb += vb * vb;
  }
  const denom = Math.sqrt(na * nb);
  return denom > 1e-8 ? dot / denom : 0;
}

export function vectorL2Norm(arr: Float32Array, offset: number, dim: number): number {
  let sq = 0;
  for (let d = 0; d < dim; d++) {
    const v = arr[offset + d];
    sq += v * v;
  }
  return Math.sqrt(sq);
}
