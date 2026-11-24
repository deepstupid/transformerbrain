export type ProjectionMode = '2d' | '3d';
export type PointerTool = 'pan' | 'rotate';

export interface CameraState {
  panX: number;
  panY: number;
  zoom: number;
  rot2D: number;
  yaw: number;
  pitch: number;
  mode: ProjectionMode;
}

export function createDefaultCamera(mode: ProjectionMode = '2d'): CameraState {
  return {
    panX: 0,
    panY: 0,
    zoom: 1,
    rot2D: 0,
    yaw: 0.52,
    pitch: 0.34,
    mode,
  };
}

export interface ProjectedPoint {
  sx: number;
  sy: number;
  depth: number;
  perspective: number;
  visible: boolean;
}

const CAM_DIST = 3.4;

export function projectWorldPoint(
  x: number,
  y: number,
  z: number,
  cam: CameraState,
  width: number,
  height: number
): ProjectedPoint {
  let wx: number;
  let wy: number;
  let depth = 0;
  let persp = 1;

  if (cam.mode === '3d') {
    const cy = Math.cos(cam.yaw);
    const sy = Math.sin(cam.yaw);
    const cp = Math.cos(cam.pitch);
    const sp = Math.sin(cam.pitch);

    const x1 = x * cy - z * sy;
    const z1 = x * sy + z * cy;
    const y2 = y * cp - z1 * sp;
    const z2 = y * sp + z1 * cp;

    depth = z2;
    persp = CAM_DIST / Math.max(0.45, CAM_DIST - z2);
    wx = x1 * persp * cam.zoom + cam.panX;
    wy = y2 * persp * cam.zoom + cam.panY;
  } else {
    const c = Math.cos(cam.rot2D);
    const s = Math.sin(cam.rot2D);
    const xr = x * c - y * s;
    const yr = x * s + y * c;
    wx = xr * cam.zoom + cam.panX;
    wy = yr * cam.zoom + cam.panY;
  }

  const baseScale = Math.min(width, height) * 0.42;
  const sx = width * 0.5 + wx * baseScale;
  const sy = height * 0.5 - wy * baseScale;

  return {
    sx,
    sy,
    depth,
    perspective: persp,
    visible: cam.mode === '2d' || CAM_DIST - depth > 0.35,
  };
}

export function screenToWorld2D(
  sx: number,
  sy: number,
  cam: CameraState,
  width: number,
  height: number
): { x: number; y: number } {
  const baseScale = Math.max(1, Math.min(width, height) * 0.42);
  const wx = (sx - width * 0.5) / baseScale;
  const wy = (height * 0.5 - sy) / baseScale;

  const ux = (wx - cam.panX) / cam.zoom;
  const uy = (wy - cam.panY) / cam.zoom;

  const c = Math.cos(-cam.rot2D);
  const s = Math.sin(-cam.rot2D);
  return {
    x: ux * c - uy * s,
    y: ux * s + uy * c,
  };
}

export function applyZoomAtPoint(
  cam: CameraState,
  factor: number,
  cursorX: number,
  cursorY: number,
  width: number,
  height: number
): CameraState {
  const nextZoom = Math.max(0.35, Math.min(8.0, cam.zoom * factor));
  if (Math.abs(nextZoom - cam.zoom) < 1e-6) return cam;

  const baseScale = Math.max(1, Math.min(width, height) * 0.42);
  const wx = (cursorX - width * 0.5) / baseScale;
  const wy = (height * 0.5 - cursorY) / baseScale;

  const ratio = nextZoom / cam.zoom;
  return {
    ...cam,
    zoom: nextZoom,
    panX: wx - (wx - cam.panX) * ratio,
    panY: wy - (wy - cam.panY) * ratio,
  };
}
