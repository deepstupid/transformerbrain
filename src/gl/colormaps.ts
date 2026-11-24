export type RGB = [number, number, number];

/**
 * Calibrated observatory heatmap ramp:
 * deep obsidian -> slate blue -> teal -> laser cyan -> telemetry amber -> incandescent cream
 */
export function observatoryHeatColor(t: number): RGB {
  const v = Math.max(0, Math.min(1, t));
  if (v < 0.25) {
    const u = v / 0.25;
    return [lerp(0.05, 0.06, u), lerp(0.08, 0.22, u), lerp(0.14, 0.36, u)];
  }
  if (v < 0.55) {
    const u = (v - 0.25) / 0.3;
    return [lerp(0.06, 0.02, u), lerp(0.22, 0.71, u), lerp(0.36, 0.83, u)];
  }
  if (v < 0.82) {
    const u = (v - 0.55) / 0.27;
    return [lerp(0.02, 0.96, u), lerp(0.71, 0.62, u), lerp(0.83, 0.04, u)];
  }
  const u = (v - 0.82) / 0.18;
  return [lerp(0.96, 1.0, u), lerp(0.62, 0.93, u), lerp(0.04, 0.72, u)];
}

/**
 * Signed activation ramp: rose (-1) -> slate (0) -> cyan (+1)
 */
export function signedActivationColor(val: number): RGB {
  const v = Math.max(-1, Math.min(1, val));
  if (v >= 0) {
    return [lerp(0.12, 0.02, v), lerp(0.16, 0.72, v), lerp(0.24, 0.84, v)];
  }
  const u = -v;
  return [lerp(0.12, 0.95, u), lerp(0.16, 0.25, u), lerp(0.24, 0.38, u)];
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}
