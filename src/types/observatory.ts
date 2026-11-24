import { CameraState } from '../gl/camera';

export type ViewMode = 'overview' | 'embeddings' | 'attention' | 'pathways' | 'logits';

export type SelectionTarget =
  | { kind: 'token'; tokenId: number; tokenPos?: number }
  | { kind: 'attentionCell'; layer: number; head: number; qIdx: number; kIdx: number }
  | { kind: 'pathwayNode'; stage: number; tokenPos: number }
  | { kind: 'mlpNeuron'; layer: number; neuronIdx: number; tokenPos: number };

export interface OverlayLabel {
  sx: number;
  sy: number;
  text: string;
  color: string;
  align?: CanvasTextAlign;
  fontSize?: number;
  alpha?: number;
}

export interface HitTarget {
  sx: number;
  sy: number;
  radiusPx: number;
  halfW?: number;
  halfH?: number;
  target: SelectionTarget;
  hoverTitle: string;
  hoverDetail: string;
}

export interface LensViewportProps {
  mode: Exclude<ViewMode, 'overview'>;
  camera: CameraState;
  onCameraChange: (next: CameraState) => void;
  selection: SelectionTarget | null;
  onSelect: (target: SelectionTarget | null) => void;
}
