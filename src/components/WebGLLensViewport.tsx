import React, { useEffect, useRef, useState } from 'react';
import { Expand, Hand, Minimize2, RotateCw, RotateCcw, ZoomIn, ZoomOut } from 'lucide-react';
import { ObservatoryGL } from '../gl/ObservatoryGL';
import { applyZoomAtPoint, CameraState, PointerTool } from '../gl/camera';
import {
  buildAttentionScene,
  buildEmbeddingsScene,
  buildLogitsScene,
  buildPathwaysScene,
} from '../gl/scenes';
import { CharTokenizer } from '../ml/tokenizer';
import { ThoughtFrame } from '../ml/transformer';
import { HitTarget, SelectionTarget, ViewMode } from '../types/observatory';

export type LensKind = Exclude<ViewMode, 'overview'>;

interface WebGLLensViewportProps {
  lens: LensKind;
  title: string;
  subtitle: string;
  frame: ThoughtFrame;
  tokenizer: CharTokenizer;
  camera: CameraState;
  onCameraChange: (next: CameraState) => void;
  onResetCamera: () => void;
  selection: SelectionTarget | null;
  onSelect: (target: SelectionTarget | null) => void;
  activeLayer: number;
  activeHead: number;
  topK: number;
  compact?: boolean;
  isExpanded?: boolean;
  onToggleExpand?: () => void;
  onLayerChange?: (layer: number) => void;
  onHeadChange?: (head: number) => void;
}

export const WebGLLensViewport: React.FC<WebGLLensViewportProps> = ({
  lens,
  title,
  subtitle,
  frame,
  tokenizer,
  camera,
  onCameraChange,
  onResetCamera,
  selection,
  onSelect,
  activeLayer,
  activeHead,
  topK,
  compact = false,
  isExpanded = false,
  onToggleExpand,
  onLayerChange,
  onHeadChange,
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const glCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const labelCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const rendererRef = useRef<ObservatoryGL | null>(null);

  const [tool, setTool] = useState<PointerTool>(lens === 'embeddings' && camera.mode === '3d' ? 'rotate' : 'pan');
  const [hovered, setHovered] = useState<{ hit: HitTarget; x: number; y: number } | null>(null);
  const [glError, setGlError] = useState<string | null>(null);

  // Refs for RAF loop without recreating WebGL context
  const stateRef = useRef({
    frame,
    tokenizer,
    camera,
    selection,
    activeLayer,
    activeHead,
    topK,
    lens,
  });
  stateRef.current = {
    frame,
    tokenizer,
    camera,
    selection,
    activeLayer,
    activeHead,
    topK,
    lens,
  };

  const hitsRef = useRef<HitTarget[]>([]);
  const smoothedCoordsRef = useRef<Float32Array>(new Float32Array(frame.embeddingCoords3D));
  const pointerStateRef = useRef<{
    dragging: boolean;
    startX: number;
    startY: number;
    lastX: number;
    lastY: number;
    movedPx: number;
    mode: PointerTool;
    camSnapshot: CameraState;
  } | null>(null);

  // Sync default tool when switching 2D/3D on embeddings
  useEffect(() => {
    if (lens === 'embeddings' && camera.mode === '3d') {
      setTool('rotate');
    }
  }, [lens, camera.mode]);

  // Initialize WebGL2 context once per mount
  useEffect(() => {
    const glCanvas = glCanvasRef.current;
    if (!glCanvas) return;

    try {
      const renderer = new ObservatoryGL(glCanvas);
      rendererRef.current = renderer;
      setGlError(null);
    } catch (err) {
      setGlError(err instanceof Error ? err.message : 'WebGL2 initialization failed');
      return;
    }

    let rafId = 0;
    const startTime = performance.now();
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    const renderLoop = (now: number) => {
      const container = containerRef.current;
      const canvas = glCanvasRef.current;
      const overlay = labelCanvasRef.current;
      const renderer = rendererRef.current;
      if (!container || !canvas || !overlay || !renderer) {
        rafId = requestAnimationFrame(renderLoop);
        return;
      }

      const rect = container.getBoundingClientRect();
      const cssW = Math.max(100, Math.floor(rect.width));
      const cssH = Math.max(100, Math.floor(rect.height));
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const pxW = Math.floor(cssW * dpr);
      const pxH = Math.floor(cssH * dpr);

      if (canvas.width !== pxW || canvas.height !== pxH) {
        canvas.width = pxW;
        canvas.height = pxH;
      }
      if (overlay.width !== pxW || overlay.height !== pxH) {
        overlay.width = pxW;
        overlay.height = pxH;
      }

      const st = stateRef.current;
      const targetCoords = st.frame.embeddingCoords3D;
      if (smoothedCoordsRef.current.length !== targetCoords.length) {
        smoothedCoordsRef.current = new Float32Array(targetCoords);
      } else {
        const lerpRate = reducedMotion ? 1.0 : 0.22;
        const sm = smoothedCoordsRef.current;
        for (let i = 0; i < targetCoords.length; i++) {
          sm[i] += (targetCoords[i] - sm[i]) * lerpRate;
        }
      }

      let scene;
      if (st.lens === 'embeddings') {
        scene = buildEmbeddingsScene(
          st.frame,
          st.tokenizer,
          st.camera,
          cssW,
          cssH,
          st.selection,
          smoothedCoordsRef.current,
          reducedMotion
        );
      } else if (st.lens === 'attention') {
        scene = buildAttentionScene(
          st.frame,
          st.tokenizer,
          st.camera,
          cssW,
          cssH,
          st.selection,
          st.activeLayer,
          st.activeHead,
          reducedMotion
        );
      } else if (st.lens === 'pathways') {
        scene = buildPathwaysScene(
          st.frame,
          st.tokenizer,
          st.camera,
          cssW,
          cssH,
          st.selection,
          st.activeLayer,
          reducedMotion
        );
      } else {
        scene = buildLogitsScene(
          st.frame,
          st.tokenizer,
          st.camera,
          cssW,
          cssH,
          st.selection,
          st.topK,
          reducedMotion
        );
      }

      hitsRef.current = scene.hits;
      const timeSec = (now - startTime) * 0.001;

      renderer.beginFrame(pxW, pxH);
      renderer.drawQuads(scene.quads, st.camera, cssW, cssH);
      renderer.drawSegments(scene.segments, st.camera, cssW, cssH, timeSec);
      renderer.drawNodes(scene.nodes, st.camera, cssW, cssH);

      // Draw crisp Canvas2D labels
      const ctx = overlay.getContext('2d');
      if (ctx) {
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, cssW, cssH);
        ctx.textBaseline = 'middle';
        for (const lbl of scene.labels) {
          if (lbl.sx < -40 || lbl.sx > cssW + 40 || lbl.sy < -20 || lbl.sy > cssH + 20) continue;
          ctx.globalAlpha = lbl.alpha ?? 1;
          ctx.fillStyle = lbl.color;
          ctx.font = `500 ${lbl.fontSize ?? 10}px "IBM Plex Mono", monospace`;
          ctx.textAlign = lbl.align ?? 'center';
          ctx.fillText(lbl.text, lbl.sx, lbl.sy);
        }
        ctx.globalAlpha = 1;
      }

      rafId = requestAnimationFrame(renderLoop);
    };

    rafId = requestAnimationFrame(renderLoop);
    return () => {
      cancelAnimationFrame(rafId);
      rendererRef.current?.dispose();
      rendererRef.current = null;
    };
  }, []);

  const findHitAt = (x: number, y: number): HitTarget | null => {
    let best: HitTarget | null = null;
    let bestDist = Infinity;
    for (const h of hitsRef.current) {
      if (h.halfW !== undefined && h.halfH !== undefined) {
        if (Math.abs(x - h.sx) <= h.halfW && Math.abs(y - h.sy) <= h.halfH) {
          const d = Math.hypot(x - h.sx, y - h.sy);
          if (d < bestDist) {
            bestDist = d;
            best = h;
          }
        }
      } else {
        const d = Math.hypot(x - h.sx, y - h.sy);
        if (d <= h.radiusPx && d < bestDist) {
          bestDist = d;
          best = h;
        }
      }
    }
    return best;
  };

  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0 && e.button !== 2) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    const activeTool: PointerTool = e.shiftKey || e.button === 2 ? 'rotate' : tool;
    pointerStateRef.current = {
      dragging: true,
      startX: e.clientX,
      startY: e.clientY,
      lastX: e.clientX,
      lastY: e.clientY,
      movedPx: 0,
      mode: activeTool,
      camSnapshot: camera,
    };
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const container = containerRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    const lx = e.clientX - rect.left;
    const ly = e.clientY - rect.top;

    const drag = pointerStateRef.current;
    if (drag && drag.dragging) {
      const dx = e.clientX - drag.lastX;
      const dy = e.clientY - drag.lastY;
      drag.movedPx += Math.hypot(dx, dy);
      drag.lastX = e.clientX;
      drag.lastY = e.clientY;

      const baseScale = Math.max(1, Math.min(rect.width, rect.height) * 0.42);
      if (drag.mode === 'pan') {
        onCameraChange({
          ...camera,
          panX: camera.panX + dx / baseScale,
          panY: camera.panY - dy / baseScale,
        });
      } else if (camera.mode === '3d') {
        onCameraChange({
          ...camera,
          yaw: camera.yaw + dx * 0.0085,
          pitch: Math.max(-1.45, Math.min(1.45, camera.pitch + dy * 0.0085)),
        });
      } else {
        // 2D rotation around viewport center
        const cx = rect.width * 0.5;
        const cy = rect.height * 0.5;
        const prevAngle = Math.atan2(ly - dy - cy, lx - dx - cx);
        const currAngle = Math.atan2(ly - cy, lx - cx);
        let dAngle = currAngle - prevAngle;
        if (dAngle > Math.PI) dAngle -= Math.PI * 2;
        if (dAngle < -Math.PI) dAngle += Math.PI * 2;
        onCameraChange({
          ...camera,
          rot2D: camera.rot2D - dAngle,
        });
      }
      setHovered(null);
      return;
    }

    const hit = findHitAt(lx, ly);
    if (hit) {
      setHovered({ hit, x: lx, y: ly });
    } else if (hovered) {
      setHovered(null);
    }
  };

  const handlePointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    const drag = pointerStateRef.current;
    pointerStateRef.current = null;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
    if (!drag || drag.movedPx >= 5) return;

    const container = containerRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    const lx = e.clientX - rect.left;
    const ly = e.clientY - rect.top;
    const hit = findHitAt(lx, ly);
    if (hit) {
      onSelect(hit.target);
    } else {
      onSelect(null);
    }
  };

  // Non-passive wheel listener for smooth cursor-anchored zoom
  const cameraRef = useRef(camera);
  cameraRef.current = camera;
  const onCameraChangeRef = useRef(onCameraChange);
  onCameraChangeRef.current = onCameraChange;

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const onWheelNative = (e: WheelEvent) => {
      e.preventDefault();
      const rect = container.getBoundingClientRect();
      const factor = e.deltaY < 0 ? 1.12 : 0.89;
      onCameraChangeRef.current(
        applyZoomAtPoint(
          cameraRef.current,
          factor,
          e.clientX - rect.left,
          e.clientY - rect.top,
          rect.width,
          rect.height
        )
      );
    };
    container.addEventListener('wheel', onWheelNative, { passive: false });
    return () => container.removeEventListener('wheel', onWheelNative);
  }, []);

  const zoomCenter = (factor: number) => {
    const container = containerRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    onCameraChange(
      applyZoomAtPoint(camera, factor, rect.width * 0.5, rect.height * 0.5, rect.width, rect.height)
    );
  };

  const angleDeg =
    camera.mode === '3d'
      ? Math.round(((camera.yaw * 180) / Math.PI) % 360)
      : Math.round(((camera.rot2D * 180) / Math.PI) % 360);

  return (
    <div
      className="relative flex flex-col w-full h-full bg-[#07090E] border border-[#1E293B] rounded-lg overflow-hidden select-none"
      onContextMenu={(e) => e.preventDefault()}
    >
      {/* Top Lens Header Bar */}
      <div className="flex items-center justify-between gap-2 px-3 py-2 bg-[#0C111B]/95 border-b border-[#1E293B] shrink-0">
        <div className="flex items-baseline gap-2 min-w-0">
          <h3 className="text-xs font-semibold text-slate-100 whitespace-nowrap">{title}</h3>
          <span className="text-[11px] font-mono text-slate-400 truncate hidden sm:inline">
            {subtitle}
          </span>
        </div>

        <div className="flex items-center gap-1.5 shrink-0">
          {/* Layer / Head quick selectors when relevant */}
          {(lens === 'attention' || lens === 'pathways') && onLayerChange && (
            <div className="flex items-center bg-[#131A26] border border-[#1E293B] rounded px-1 py-0.5 gap-0.5">
              {Array.from({ length: frame.nLayers }, (_, l) => (
                <button
                  key={l}
                  type="button"
                  onClick={() => onLayerChange(l)}
                  className={`px-1.5 py-0.5 text-[10px] font-mono rounded transition-colors whitespace-nowrap shrink-0 ${
                    activeLayer === l
                      ? 'bg-cyan-500/20 text-cyan-300'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  L{l}
                </button>
              ))}
            </div>
          )}

          {lens === 'attention' && onHeadChange && (
            <div className="flex items-center bg-[#131A26] border border-[#1E293B] rounded px-1 py-0.5 gap-0.5">
              <button
                type="button"
                onClick={() => onHeadChange(-1)}
                className={`px-1.5 py-0.5 text-[10px] font-mono rounded transition-colors whitespace-nowrap shrink-0 ${
                  activeHead === -1
                    ? 'bg-amber-500/20 text-amber-300'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Avg
              </button>
              {Array.from({ length: frame.nHeads }, (_, h) => (
                <button
                  key={h}
                  type="button"
                  onClick={() => onHeadChange(h)}
                  className={`px-1.5 py-0.5 text-[10px] font-mono rounded transition-colors whitespace-nowrap shrink-0 ${
                    activeHead === h
                      ? 'bg-cyan-500/20 text-cyan-300'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  H{h}
                </button>
              ))}
            </div>
          )}

          {/* 2D / 3D Projection Switch for Embeddings */}
          {lens === 'embeddings' && (
            <div className="flex items-center bg-[#131A26] border border-[#1E293B] rounded px-1 py-0.5 gap-0.5">
              <button
                type="button"
                onClick={() => onCameraChange({ ...camera, mode: '2d' })}
                className={`px-1.5 py-0.5 text-[10px] font-mono rounded transition-colors whitespace-nowrap shrink-0 ${
                  camera.mode === '2d'
                    ? 'bg-cyan-500/20 text-cyan-300'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                2D
              </button>
              <button
                type="button"
                onClick={() => onCameraChange({ ...camera, mode: '3d' })}
                className={`px-1.5 py-0.5 text-[10px] font-mono rounded transition-colors whitespace-nowrap shrink-0 ${
                  camera.mode === '3d'
                    ? 'bg-cyan-500/20 text-cyan-300'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                3D
              </button>
            </div>
          )}

          {onToggleExpand && (
            <button
              type="button"
              onClick={onToggleExpand}
              title={isExpanded ? 'Return to 2×2 Grid' : 'Focus Full Lens'}
              className="p-1 text-slate-400 hover:text-cyan-300 hover:bg-[#131A26] rounded transition-colors shrink-0"
            >
              {isExpanded ? <Minimize2 className="w-3.5 h-3.5" /> : <Expand className="w-3.5 h-3.5" />}
            </button>
          )}
        </div>
      </div>

      {/* Interactive WebGL2 Stage */}
      <div
        ref={containerRef}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerLeave={() => setHovered(null)}
        className={`relative flex-1 min-h-0 touch-none ${
          hovered ? 'cursor-pointer' : tool === 'rotate' ? 'cursor-grab' : 'cursor-crosshair'
        }`}
      >
        <canvas ref={glCanvasRef} className="absolute inset-0 w-full h-full block" />
        <canvas ref={labelCanvasRef} className="absolute inset-0 w-full h-full block pointer-events-none" />

        {glError && (
          <div className="absolute inset-0 flex items-center justify-center bg-[#07090E]/90 p-4 text-center">
            <p className="text-xs font-mono text-rose-400">{glError}</p>
          </div>
        )}

        {/* Camera Controls Floating HUD */}
        <div
          onPointerDown={(e) => e.stopPropagation()}
          className="absolute bottom-2.5 right-2.5 flex items-center gap-1 bg-[#0C111B]/90 border border-[#1E293B] rounded-md p-1 backdrop-blur-sm"
        >
          <button
            type="button"
            onClick={() => setTool('pan')}
            title="Pan view (Drag)"
            className={`flex items-center gap-1 px-1.5 py-1 text-[10px] font-mono rounded transition-colors whitespace-nowrap shrink-0 ${
              tool === 'pan'
                ? 'bg-cyan-500/20 text-cyan-300'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Hand className="w-3 h-3" />
            {!compact && <span>Pan</span>}
          </button>
          <button
            type="button"
            onClick={() => setTool('rotate')}
            title="Rotate view (Drag or Shift+Drag)"
            className={`flex items-center gap-1 px-1.5 py-1 text-[10px] font-mono rounded transition-colors whitespace-nowrap shrink-0 ${
              tool === 'rotate'
                ? 'bg-cyan-500/20 text-cyan-300'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <RotateCw className="w-3 h-3" />
            {!compact && <span>Rotate</span>}
          </button>
          <span className="w-px h-3.5 bg-[#1E293B] mx-0.5" aria-hidden="true" />
          <button
            type="button"
            onClick={() => zoomCenter(1.22)}
            title="Zoom in"
            className="p-1 text-slate-400 hover:text-slate-100 rounded transition-colors"
          >
            <ZoomIn className="w-3.5 h-3.5" />
          </button>
          <button
            type="button"
            onClick={() => zoomCenter(0.82)}
            title="Zoom out"
            className="p-1 text-slate-400 hover:text-slate-100 rounded transition-colors"
          >
            <ZoomOut className="w-3.5 h-3.5" />
          </button>
          <button
            type="button"
            onClick={onResetCamera}
            title="Reset camera"
            className="p-1 text-slate-400 hover:text-amber-300 rounded transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
          {!compact && (
            <span className="px-1.5 text-[10px] font-mono tabular-nums text-slate-400 border-l border-[#1E293B]">
              {camera.zoom.toFixed(2)}× · {angleDeg}°
            </span>
          )}
        </div>

        {/* Hover Crosshair Readout */}
        {hovered && (
          <div
            className="pointer-events-none absolute z-20 px-2.5 py-1.5 bg-[#0C111B]/95 border border-cyan-500/40 rounded shadow-lg max-w-[240px]"
            style={{
              left: Math.min(
                (containerRef.current?.clientWidth ?? 400) - 210,
                Math.max(8, hovered.x + 12)
              ),
              top: Math.max(8, hovered.y - 44),
            }}
          >
            <div className="text-[11px] font-mono font-medium text-cyan-200 truncate">
              {hovered.hit.hoverTitle}
            </div>
            <div className="text-[10px] font-mono tabular-nums text-slate-300 truncate">
              {hovered.hit.hoverDetail}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
