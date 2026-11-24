import { CameraState } from './camera';

export interface QuadInstance {
  x: number;
  y: number;
  z: number;
  w: number;
  h: number;
  r: number;
  g: number;
  b: number;
  a: number;
  border: number; // 0..1 border highlight
}

export interface NodeInstance {
  x: number;
  y: number;
  z: number;
  radiusPx: number;
  r: number;
  g: number;
  b: number;
  a: number;
  ring: number; // 0..1 ring intensity
}

export interface SegmentInstance {
  x1: number;
  y1: number;
  z1: number;
  x2: number;
  y2: number;
  z2: number;
  thicknessPx: number;
  r: number;
  g: number;
  b: number;
  a: number;
  pulseSpeed: number;
  pulsePhase: number;
}

const COMMON_GLSL_CAMERA = /* glsl */ `
uniform vec2 uResolution;
uniform float uBaseScale;
uniform vec2 uPan;
uniform float uZoom;
uniform float uRot2D;
uniform float uYaw;
uniform float uPitch;
uniform int uMode3D;

vec4 projectWorld(vec3 p, out float persp) {
  vec2 w;
  float depthNorm = 0.0;
  if (uMode3D == 1) {
    float cy = cos(uYaw);
    float sy = sin(uYaw);
    float cp = cos(uPitch);
    float sp = sin(uPitch);

    float x1 = p.x * cy - p.z * sy;
    float z1 = p.x * sy + p.z * cy;
    float y2 = p.y * cp - z1 * sp;
    float z2 = p.y * sp + z1 * cp;

    float camDist = 3.4;
    persp = camDist / max(0.45, camDist - z2);
    w = vec2(x1 * persp * uZoom + uPan.x, y2 * persp * uZoom + uPan.y);
    depthNorm = clamp(-z2 * 0.25, -0.95, 0.95);
  } else {
    float c = cos(uRot2D);
    float s = sin(uRot2D);
    vec2 r = vec2(p.x * c - p.y * s, p.x * s + p.y * c);
    persp = 1.0;
    w = r * uZoom + uPan;
  }
  vec2 ndc = vec2(
    (w.x * uBaseScale) / (uResolution.x * 0.5),
    (w.y * uBaseScale) / (uResolution.y * 0.5)
  );
  return vec4(ndc, depthNorm, 1.0);
}
`;

const QUAD_VERT = /* glsl */ `#version 300 es
precision highp float;
layout(location = 0) in vec2 aCorner;
layout(location = 1) in vec3 aCenter;
layout(location = 2) in vec2 aSize;
layout(location = 3) in vec4 aColor;
layout(location = 4) in float aBorder;

${COMMON_GLSL_CAMERA}

out vec2 vUv;
out vec4 vColor;
out float vBorder;

void main() {
  vec3 worldPos = aCenter + vec3(aCorner * aSize, 0.0);
  float persp;
  gl_Position = projectWorld(worldPos, persp);
  vUv = aCorner + 0.5;
  vColor = aColor;
  vBorder = aBorder;
}
`;

const QUAD_FRAG = /* glsl */ `#version 300 es
precision highp float;
in vec2 vUv;
in vec4 vColor;
in float vBorder;
out vec4 outColor;

void main() {
  float edgeX = min(vUv.x, 1.0 - vUv.x);
  float edgeY = min(vUv.y, 1.0 - vUv.y);
  float edge = min(edgeX, edgeY);

  vec3 col = vColor.rgb;
  float alpha = vColor.a;

  if (vBorder > 0.01) {
    float borderMask = 1.0 - smoothstep(0.06, 0.14, edge);
    col = mix(col, vec3(0.92, 0.98, 1.0), borderMask * vBorder * 0.9);
    alpha = max(alpha, borderMask * vBorder * 0.92);
  }

  outColor = vec4(col, alpha);
}
`;

const NODE_VERT = /* glsl */ `#version 300 es
precision highp float;
layout(location = 0) in vec2 aCorner;
layout(location = 1) in vec3 aPos;
layout(location = 2) in float aRadiusPx;
layout(location = 3) in vec4 aColor;
layout(location = 4) in float aRing;

${COMMON_GLSL_CAMERA}

out vec2 vLocal;
out float vRadiusPx;
out vec4 vColor;
out float vRing;

void main() {
  float persp;
  vec4 centerClip = projectWorld(aPos, persp);
  float rPx = aRadiusPx * (uMode3D == 1 ? clamp(persp, 0.6, 1.8) : 1.0);
  float padPx = rPx + 6.0;
  vec2 offsetNdc = (aCorner * 2.0 * padPx) / uResolution;
  gl_Position = vec4(centerClip.xy + offsetNdc, centerClip.z, 1.0);
  vLocal = aCorner * 2.0 * padPx;
  vRadiusPx = rPx;
  vColor = aColor;
  vRing = aRing;
}
`;

const NODE_FRAG = /* glsl */ `#version 300 es
precision highp float;
in vec2 vLocal;
in float vRadiusPx;
in vec4 vColor;
in float vRing;
out vec4 outColor;

void main() {
  float dist = length(vLocal);
  float core = 1.0 - smoothstep(vRadiusPx - 1.2, vRadiusPx + 0.8, dist);
  float halo = exp(-max(0.0, dist - vRadiusPx * 0.4) * 0.28) * (0.28 + vRing * 0.45);
  float ring = (1.0 - smoothstep(1.0, 2.2, abs(dist - (vRadiusPx + 2.5)))) * vRing;

  vec3 col = vColor.rgb + vec3(0.25, 0.35, 0.4) * ring;
  float alpha = max(core * vColor.a, max(halo * vColor.a, ring * 0.95));
  if (alpha < 0.005) discard;
  outColor = vec4(col, clamp(alpha, 0.0, 1.0));
}
`;

const SEG_VERT = /* glsl */ `#version 300 es
precision highp float;
layout(location = 0) in vec2 aQuad; // x in [0,1] along segment, y in [-0.5, 0.5] across thickness
layout(location = 1) in vec3 aStart;
layout(location = 2) in vec3 aEnd;
layout(location = 3) in float aThickness;
layout(location = 4) in vec4 aColor;
layout(location = 5) in vec2 aPulse; // x: speed, y: phase

${COMMON_GLSL_CAMERA}

out float vAlong;
out float vAcross;
out vec4 vColor;
out vec2 vPulse;

void main() {
  float p1;
  float p2;
  vec4 c1 = projectWorld(aStart, p1);
  vec4 c2 = projectWorld(aEnd, p2);

  vec2 s1 = c1.xy * uResolution * 0.5;
  vec2 s2 = c2.xy * uResolution * 0.5;
  vec2 dir = s2 - s1;
  float len = max(0.001, length(dir));
  vec2 tangent = dir / len;
  vec2 normal = vec2(-tangent.y, tangent.x);

  float halfW = aThickness + 1.5;
  vec2 screenPos = mix(s1, s2, aQuad.x) + normal * (aQuad.y * 2.0 * halfW);
  vec2 ndc = screenPos / (uResolution * 0.5);
  float z = mix(c1.z, c2.z, aQuad.x);

  gl_Position = vec4(ndc, z, 1.0);
  vAlong = aQuad.x;
  vAcross = aQuad.y * 2.0; // -1..1
  vColor = aColor;
  vPulse = aPulse;
}
`;

const SEG_FRAG = /* glsl */ `#version 300 es
precision highp float;
uniform float uTime;
in float vAlong;
in float vAcross;
in vec4 vColor;
in vec2 vPulse;
out vec4 outColor;

void main() {
  float radial = 1.0 - smoothstep(0.25, 1.0, abs(vAcross));
  float alpha = vColor.a * radial;
  vec3 col = vColor.rgb;

  if (vPulse.x > 0.001) {
    float head = fract(uTime * vPulse.x + vPulse.y);
    float d = abs(vAlong - head);
    d = min(d, 1.0 - d);
    float spark = exp(-d * 18.0);
    col = mix(col, vec3(0.95, 0.99, 1.0), spark * 0.85);
    alpha = clamp(alpha + spark * 0.55 * radial, 0.0, 1.0);
  }

  if (alpha < 0.004) discard;
  outColor = vec4(col, alpha);
}
`;

function compileShader(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const sh = gl.createShader(type);
  if (!sh) throw new Error('Unable to create WebGL2 shader');
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const msg = gl.getShaderInfoLog(sh) || 'Shader compilation error';
    gl.deleteShader(sh);
    throw new Error(msg);
  }
  return sh;
}

function createProgram(gl: WebGL2RenderingContext, vsSrc: string, fsSrc: string): WebGLProgram {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vsSrc);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSrc);
  const prog = gl.createProgram();
  if (!prog) throw new Error('Unable to create WebGL2 program');
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const msg = gl.getProgramInfoLog(prog) || 'Program link error';
    gl.deleteProgram(prog);
    throw new Error(msg);
  }
  return prog;
}

export class ObservatoryGL {
  private gl: WebGL2RenderingContext;
  private quadProg: WebGLProgram;
  private nodeProg: WebGLProgram;
  private segProg: WebGLProgram;

  private quadVao: WebGLVertexArrayObject;
  private quadInstBuf: WebGLBuffer;
  private nodeVao: WebGLVertexArrayObject;
  private nodeInstBuf: WebGLBuffer;
  private segVao: WebGLVertexArrayObject;
  private segInstBuf: WebGLBuffer;
  private cornerBuf: WebGLBuffer;
  private segCornerBuf: WebGLBuffer;

  private quadData = new Float32Array(1024 * 10);
  private nodeData = new Float32Array(512 * 9);
  private segData = new Float32Array(2048 * 13);

  constructor(canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2', {
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance',
    });
    if (!gl) {
      throw new Error('WebGL2 is unavailable in this browser context');
    }
    this.gl = gl;

    this.quadProg = createProgram(gl, QUAD_VERT, QUAD_FRAG);
    this.nodeProg = createProgram(gl, NODE_VERT, NODE_FRAG);
    this.segProg = createProgram(gl, SEG_VERT, SEG_FRAG);

    // Base quad [-0.5, 0.5]^2
    this.cornerBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.cornerBuf);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5]),
      gl.STATIC_DRAW
    );

    // Segment quad [0,1] x [-0.5, 0.5]
    this.segCornerBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.segCornerBuf);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([0, -0.5, 1, -0.5, 0, 0.5, 1, -0.5, 1, 0.5, 0, 0.5]),
      gl.STATIC_DRAW
    );

    // Quad VAO (stride 10 floats = 40 bytes)
    this.quadVao = gl.createVertexArray()!;
    this.quadInstBuf = gl.createBuffer()!;
    gl.bindVertexArray(this.quadVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.cornerBuf);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadInstBuf);
    const qStride = 10 * 4;
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, qStride, 0);
    gl.vertexAttribDivisor(1, 1);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 2, gl.FLOAT, false, qStride, 3 * 4);
    gl.vertexAttribDivisor(2, 1);
    gl.enableVertexAttribArray(3);
    gl.vertexAttribPointer(3, 4, gl.FLOAT, false, qStride, 5 * 4);
    gl.vertexAttribDivisor(3, 1);
    gl.enableVertexAttribArray(4);
    gl.vertexAttribPointer(4, 1, gl.FLOAT, false, qStride, 9 * 4);
    gl.vertexAttribDivisor(4, 1);

    // Node VAO (stride 9 floats = 36 bytes)
    this.nodeVao = gl.createVertexArray()!;
    this.nodeInstBuf = gl.createBuffer()!;
    gl.bindVertexArray(this.nodeVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.cornerBuf);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.nodeInstBuf);
    const nStride = 9 * 4;
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, nStride, 0);
    gl.vertexAttribDivisor(1, 1);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 1, gl.FLOAT, false, nStride, 3 * 4);
    gl.vertexAttribDivisor(2, 1);
    gl.enableVertexAttribArray(3);
    gl.vertexAttribPointer(3, 4, gl.FLOAT, false, nStride, 4 * 4);
    gl.vertexAttribDivisor(3, 1);
    gl.enableVertexAttribArray(4);
    gl.vertexAttribPointer(4, 1, gl.FLOAT, false, nStride, 8 * 4);
    gl.vertexAttribDivisor(4, 1);

    // Segment VAO (stride 13 floats = 52 bytes)
    this.segVao = gl.createVertexArray()!;
    this.segInstBuf = gl.createBuffer()!;
    gl.bindVertexArray(this.segVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.segCornerBuf);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.segInstBuf);
    const sStride = 13 * 4;
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, sStride, 0);
    gl.vertexAttribDivisor(1, 1);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 3, gl.FLOAT, false, sStride, 3 * 4);
    gl.vertexAttribDivisor(2, 1);
    gl.enableVertexAttribArray(3);
    gl.vertexAttribPointer(3, 1, gl.FLOAT, false, sStride, 6 * 4);
    gl.vertexAttribDivisor(3, 1);
    gl.enableVertexAttribArray(4);
    gl.vertexAttribPointer(4, 4, gl.FLOAT, false, sStride, 7 * 4);
    gl.vertexAttribDivisor(4, 1);
    gl.enableVertexAttribArray(5);
    gl.vertexAttribPointer(5, 2, gl.FLOAT, false, sStride, 11 * 4);
    gl.vertexAttribDivisor(5, 1);

    gl.bindVertexArray(null);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  }

  public beginFrame(pixelWidth: number, pixelHeight: number): void {
    const gl = this.gl;
    gl.viewport(0, 0, pixelWidth, pixelHeight);
    gl.clearColor(0.027, 0.035, 0.055, 1.0); // #07090E
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  }

  private setCameraUniforms(
    prog: WebGLProgram,
    cam: CameraState,
    cssWidth: number,
    cssHeight: number
  ): void {
    const gl = this.gl;
    gl.uniform2f(gl.getUniformLocation(prog, 'uResolution'), cssWidth, cssHeight);
    gl.uniform1f(gl.getUniformLocation(prog, 'uBaseScale'), Math.min(cssWidth, cssHeight) * 0.42);
    gl.uniform2f(gl.getUniformLocation(prog, 'uPan'), cam.panX, cam.panY);
    gl.uniform1f(gl.getUniformLocation(prog, 'uZoom'), cam.zoom);
    gl.uniform1f(gl.getUniformLocation(prog, 'uRot2D'), cam.rot2D);
    gl.uniform1f(gl.getUniformLocation(prog, 'uYaw'), cam.yaw);
    gl.uniform1f(gl.getUniformLocation(prog, 'uPitch'), cam.pitch);
    gl.uniform1i(gl.getUniformLocation(prog, 'uMode3D'), cam.mode === '3d' ? 1 : 0);
  }

  public drawQuads(
    quads: QuadInstance[],
    cam: CameraState,
    cssWidth: number,
    cssHeight: number
  ): void {
    if (quads.length === 0) return;
    const gl = this.gl;
    const needed = quads.length * 10;
    if (this.quadData.length < needed) {
      this.quadData = new Float32Array(needed * 2);
    }
    for (let i = 0; i < quads.length; i++) {
      const q = quads[i];
      const off = i * 10;
      this.quadData[off] = q.x;
      this.quadData[off + 1] = q.y;
      this.quadData[off + 2] = q.z;
      this.quadData[off + 3] = q.w;
      this.quadData[off + 4] = q.h;
      this.quadData[off + 5] = q.r;
      this.quadData[off + 6] = q.g;
      this.quadData[off + 7] = q.b;
      this.quadData[off + 8] = q.a;
      this.quadData[off + 9] = q.border;
    }

    gl.useProgram(this.quadProg);
    this.setCameraUniforms(this.quadProg, cam, cssWidth, cssHeight);
    gl.bindVertexArray(this.quadVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadInstBuf);
    gl.bufferData(gl.ARRAY_BUFFER, this.quadData.subarray(0, needed), gl.DYNAMIC_DRAW);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, quads.length);
  }

  public drawSegments(
    segments: SegmentInstance[],
    cam: CameraState,
    cssWidth: number,
    cssHeight: number,
    timeSec: number
  ): void {
    if (segments.length === 0) return;
    const gl = this.gl;
    const needed = segments.length * 13;
    if (this.segData.length < needed) {
      this.segData = new Float32Array(needed * 2);
    }
    for (let i = 0; i < segments.length; i++) {
      const s = segments[i];
      const off = i * 13;
      this.segData[off] = s.x1;
      this.segData[off + 1] = s.y1;
      this.segData[off + 2] = s.z1;
      this.segData[off + 3] = s.x2;
      this.segData[off + 4] = s.y2;
      this.segData[off + 5] = s.z2;
      this.segData[off + 6] = s.thicknessPx;
      this.segData[off + 7] = s.r;
      this.segData[off + 8] = s.g;
      this.segData[off + 9] = s.b;
      this.segData[off + 10] = s.a;
      this.segData[off + 11] = s.pulseSpeed;
      this.segData[off + 12] = s.pulsePhase;
    }

    gl.useProgram(this.segProg);
    this.setCameraUniforms(this.segProg, cam, cssWidth, cssHeight);
    gl.uniform1f(gl.getUniformLocation(this.segProg, 'uTime'), timeSec);
    gl.bindVertexArray(this.segVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.segInstBuf);
    gl.bufferData(gl.ARRAY_BUFFER, this.segData.subarray(0, needed), gl.DYNAMIC_DRAW);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, segments.length);
  }

  public drawNodes(
    nodes: NodeInstance[],
    cam: CameraState,
    cssWidth: number,
    cssHeight: number
  ): void {
    if (nodes.length === 0) return;
    const gl = this.gl;
    const needed = nodes.length * 9;
    if (this.nodeData.length < needed) {
      this.nodeData = new Float32Array(needed * 2);
    }
    for (let i = 0; i < nodes.length; i++) {
      const n = nodes[i];
      const off = i * 9;
      this.nodeData[off] = n.x;
      this.nodeData[off + 1] = n.y;
      this.nodeData[off + 2] = n.z;
      this.nodeData[off + 3] = n.radiusPx;
      this.nodeData[off + 4] = n.r;
      this.nodeData[off + 5] = n.g;
      this.nodeData[off + 6] = n.b;
      this.nodeData[off + 7] = n.a;
      this.nodeData[off + 8] = n.ring;
    }

    gl.useProgram(this.nodeProg);
    this.setCameraUniforms(this.nodeProg, cam, cssWidth, cssHeight);
    gl.bindVertexArray(this.nodeVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.nodeInstBuf);
    gl.bufferData(gl.ARRAY_BUFFER, this.nodeData.subarray(0, needed), gl.DYNAMIC_DRAW);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, nodes.length);
  }

  public dispose(): void {
    const gl = this.gl;
    gl.deleteProgram(this.quadProg);
    gl.deleteProgram(this.nodeProg);
    gl.deleteProgram(this.segProg);
    gl.deleteBuffer(this.quadInstBuf);
    gl.deleteBuffer(this.nodeInstBuf);
    gl.deleteBuffer(this.segInstBuf);
    gl.deleteBuffer(this.cornerBuf);
    gl.deleteBuffer(this.segCornerBuf);
    gl.deleteVertexArray(this.quadVao);
    gl.deleteVertexArray(this.nodeVao);
    gl.deleteVertexArray(this.segVao);
  }
}
