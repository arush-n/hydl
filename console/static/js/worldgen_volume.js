/* Stable WebGL2 renderer for the WorldGen semantic voxel volume.
 *
 * The server sends a real occupied 3D field as vertical material runs. This
 * module decodes that field, extracts only exposed faces (including faces
 * revealed by a cut plane), and lets the depth buffer resolve visibility.
 * Camera movement therefore never changes painter ordering or cube topology.
 */

const VOLUME_SCHEMA = "hytalerl_worldgen_v2_semantic_voxel_volume_v1";
const STRIDE = 10;

const VERTEX_SHADER = `#version 300 es
precision highp float;
layout(location=0) in vec3 aPosition;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec4 aColor;

uniform vec3 uCenter;
uniform float uYaw;
uniform float uPitch;
uniform float uExaggeration;
uniform vec2 uClipScale;
uniform float uDepthScale;

out vec3 vNormal;
out vec4 vColor;
out vec3 vPosition;

void main() {
  vec3 d = aPosition - uCenter;
  d.y *= uExaggeration;
  float cy = cos(uYaw), sy = sin(uYaw);
  float x = d.x * cy - d.z * sy;
  float zr = d.x * sy + d.z * cy;
  float cp = cos(uPitch), sp = sin(uPitch);
  // Positive pitch places the camera above the world: higher cells become
  // nearer while distant ground rises toward the horizon. The former Canvas
  // projection used the opposite depth sign, which a painter could mask but
  // a real depth buffer correctly exposed as an underside view.
  float y = d.y * cp + zr * sp;
  float depth = -d.y * sp + zr * cp;
  gl_Position = vec4(x * uClipScale.x, y * uClipScale.y,
                     depth * uDepthScale, 1.0);
  vNormal = aNormal;
  vColor = aColor;
  vPosition = aPosition;
}`;

const FRAGMENT_SHADER = `#version 300 es
precision highp float;
in vec3 vNormal;
in vec4 vColor;
in vec3 vPosition;
uniform float uGrid;
uniform float uVoxelSize;
uniform vec3 uOrigin;
out vec4 outColor;

void main() {
  vec3 lightDirection = normalize(vec3(-0.45, 0.82, -0.35));
  float lambert = 0.38 + 0.62 * max(0.0, dot(normalize(vNormal), lightDirection));
  vec3 color = vColor.rgb * lambert;
  if (uGrid > 0.5) {
    vec3 cell = fract((vPosition - uOrigin) / uVoxelSize + vec3(0.0001));
    vec3 edge = min(cell, vec3(1.0) - cell);
    float distanceToEdge;
    if (abs(vNormal.x) > 0.5) distanceToEdge = min(edge.y, edge.z);
    else if (abs(vNormal.y) > 0.5) distanceToEdge = min(edge.x, edge.z);
    else distanceToEdge = min(edge.x, edge.y);
    float lineWidth = max(fwidth(distanceToEdge) * 1.2, 0.018);
    float line = 1.0 - smoothstep(0.0, lineWidth, distanceToEdge);
    color *= mix(1.0, 0.52, line);
  }
  outColor = vec4(color, vColor.a);
}`;

const FACES = [
  { delta: [-1, 0, 0], normal: [-1, 0, 0], corners: [
    [0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0],
  ] },
  { delta: [1, 0, 0], normal: [1, 0, 0], corners: [
    [1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1],
  ] },
  { delta: [0, -1, 0], normal: [0, -1, 0], corners: [
    [0, 0, 1], [0, 0, 0], [1, 0, 0], [1, 0, 1],
  ] },
  { delta: [0, 1, 0], normal: [0, 1, 0], corners: [
    [0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0],
  ] },
  { delta: [0, 0, -1], normal: [0, 0, -1], corners: [
    [1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1, 0],
  ] },
  { delta: [0, 0, 1], normal: [0, 0, 1], corners: [
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
  ] },
];
const TRIANGLES = [0, 1, 2, 0, 2, 3];

function compile(gl, kind, source) {
  const shader = gl.createShader(kind);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const message = gl.getShaderInfoLog(shader) || "unknown shader failure";
    gl.deleteShader(shader);
    throw new Error(`WorldGen volume shader failed: ${message}`);
  }
  return shader;
}

function program(gl) {
  const value = gl.createProgram();
  const vertex = compile(gl, gl.VERTEX_SHADER, VERTEX_SHADER);
  const fragment = compile(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER);
  gl.attachShader(value, vertex);
  gl.attachShader(value, fragment);
  gl.linkProgram(value);
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);
  if (!gl.getProgramParameter(value, gl.LINK_STATUS)) {
    const message = gl.getProgramInfoLog(value) || "unknown link failure";
    gl.deleteProgram(value);
    throw new Error(`WorldGen volume program failed: ${message}`);
  }
  return value;
}

function channels(hex) {
  const value = String(hex).replace("#", "");
  if (!/^[0-9a-f]{6}$/i.test(value)) throw new Error(`invalid voxel color ${hex}`);
  return [0, 2, 4].map((offset) => parseInt(value.slice(offset, offset + 2), 16) / 255);
}

function decode(volume) {
  if (!volume || volume.schema !== VOLUME_SCHEMA) {
    throw new Error("WorldGen preview has no supported occupied voxel volume");
  }
  const [nx, ny, nz] = volume.dimensions;
  if (![nx, ny, nz].every((value) => Number.isInteger(value) && value > 0)) {
    throw new Error("voxel volume dimensions are invalid");
  }
  if (!Array.isArray(volume.columns) || volume.columns.length !== nx * nz) {
    throw new Error("voxel volume column count differs from its dimensions");
  }
  const palette = new Map();
  for (const entry of volume.palette) {
    if (!Number.isInteger(entry.id) || entry.id < 0 || entry.id > 65535 || palette.has(entry.id)) {
      throw new Error("voxel volume palette IDs must be unique uint16 values");
    }
    palette.set(entry.id, {
      ...entry,
      rgba: [...channels(entry.color), Number(entry.opacity)],
    });
  }
  // Captured composites retain stable block identities across Region-local
  // palettes. A byte is enough for the synthetic authoring preview but not a
  // safe project-wide identity ceiling, so the shared renderer uses uint16.
  const cells = new Uint16Array(nx * ny * nz);
  for (let column = 0; column < volume.columns.length; column++) {
    const runs = volume.columns[column];
    if (!Array.isArray(runs) || runs.length % 3) {
      throw new Error("voxel volume column run encoding is invalid");
    }
    let previousEnd = 0;
    for (let index = 0; index < runs.length; index += 3) {
      const start = runs[index], length = runs[index + 1], material = runs[index + 2];
      if (!Number.isInteger(start) || !Number.isInteger(length) || length <= 0 ||
          start < previousEnd || start + length > ny || !palette.has(material) || material === 0) {
        throw new Error("voxel volume contains an invalid or overlapping material run");
      }
      cells.fill(material, column * ny + start, column * ny + start + length);
      previousEnd = start + length;
    }
  }
  return {
    volume,
    nx,
    ny,
    nz,
    cells,
    palette,
    unit: Number(volume.voxel_size),
    origin: volume.origin.map(Number),
  };
}

function layerEnabled(material, options) {
  if (!material || material.layer === "air") return false;
  if (material.layer === "water") return options.water;
  if (material.layer === "trees") return options.trees;
  if (material.layer === "structures") return options.structures;
  return true;
}

function clipIncludes(x, y, z, decoded, options) {
  const view = decoded.volume.recommended_view || {};
  const floorY = options.cutaway
    ? Number(view.cutaway_floor_y || 0)
    : Number(view.surface_floor_y || 0);
  if (decoded.origin[1] + (y + 1) * decoded.unit <= floorY) return false;
  if (!options.cutaway || options.cutPercent <= 0) return true;
  const amount = Math.max(0, Math.min(0.94, options.cutPercent / 100));
  if (options.cutAxis === "x") return x >= Math.floor(decoded.nx * amount);
  if (options.cutAxis === "y") return y < Math.max(1, Math.ceil(decoded.ny * (1 - amount)));
  return z >= Math.floor(decoded.nz * amount);
}

function displayFloorCell(decoded, options) {
  const view = decoded.volume.recommended_view || {};
  const floorY = options.cutaway
    ? Number(view.cutaway_floor_y || 0)
    : Number(view.surface_floor_y || 0);
  return Math.max(0, Math.floor((floorY - decoded.origin[1]) / decoded.unit));
}

function meshKey(decoded, options) {
  return [
    decoded.volume.semantic_sha256,
    Number(options.water), Number(options.trees), Number(options.structures),
    Number(options.cutaway), options.cutAxis, Math.round(options.cutPercent),
  ].join(":");
}

function emitFace(target, x, y, z, face, decoded, rgba, scale = 1) {
  const { unit, origin } = decoded;
  const base = [
    origin[0] + x * unit,
    origin[1] + y * unit,
    origin[2] + z * unit,
  ];
  for (const cornerIndex of TRIANGLES) {
    const corner = face.corners[cornerIndex];
    target.push(
      base[0] + corner[0] * unit * scale,
      base[1] + corner[1] * unit * scale,
      base[2] + corner[2] * unit * scale,
      ...face.normal,
      ...rgba,
    );
  }
}

function emitMarker(target, point, unit, rgba) {
  const size = unit * 0.42;
  const decoded = { unit: size, origin: [
    point[0] - size * 0.5,
    point[1],
    point[2] - size * 0.5,
  ] };
  for (let level = 0; level < 3; level++) {
    decoded.origin[1] = point[1] + level * size * 1.18;
    for (const face of FACES) emitFace(target, 0, 0, 0, face, decoded, rgba);
  }
}

function buildMesh(decoded, payload, options) {
  const opaque = [];
  const transparent = [];
  const { nx, ny, nz, cells, palette } = decoded;
  const cellAt = (x, y, z) => {
    if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz) return 0;
    return cells[((z * nx + x) * ny) + y];
  };
  const included = (x, y, z, materialId) => (
    materialId !== 0 && clipIncludes(x, y, z, decoded, options) &&
    layerEnabled(palette.get(materialId), options)
  );
  let visibleVoxels = 0;
  let opaqueFaces = 0;
  let transparentFaces = 0;
  const floorCell = displayFloorCell(decoded, options);

  for (let z = 0; z < nz; z++) {
    for (let x = 0; x < nx; x++) {
      for (let y = 0; y < ny; y++) {
        const materialId = cellAt(x, y, z);
        if (!included(x, y, z, materialId)) continue;
        const material = palette.get(materialId);
        const isTransparent = material.opacity < 1;
        let voxelVisible = false;
        for (const face of FACES) {
          // The lower display floor clips an otherwise uninteresting deep
          // foundation. It is not a material boundary, so do not turn it into
          // a giant bottom plate that visually swallows the terrain.
          if (face.delta[1] === -1 && y === floorCell) continue;
          const bx = x + face.delta[0], by = y + face.delta[1], bz = z + face.delta[2];
          const neighborId = cellAt(bx, by, bz);
          const neighbor = palette.get(neighborId);
          const neighborIncluded = included(bx, by, bz, neighborId);
          let exposed = !neighborIncluded;
          if (!isTransparent && neighborIncluded && neighbor?.opacity < 1) exposed = true;
          if (isTransparent && neighborIncluded && neighborId !== materialId && neighbor?.opacity < 1) {
            exposed = true;
          }
          if (!exposed) continue;
          emitFace(isTransparent ? transparent : opaque, x, y, z, face, decoded, material.rgba);
          if (isTransparent) transparentFaces += 1;
          else opaqueFaces += 1;
          voxelVisible = true;
        }
        if (voxelVisible) visibleVoxels += 1;
      }
    }
  }

  if (payload.spawn) {
    emitMarker(
      opaque,
      [payload.spawn.x, payload.spawn.y, payload.spawn.z],
      decoded.unit,
      [0.21, 0.94, 0.78, 1],
    );
    opaqueFaces += FACES.length * 3;
  }
  return {
    opaque: new Float32Array(opaque),
    transparent: new Float32Array(transparent),
    stats: {
      visibleVoxels,
      exposedFaces: opaqueFaces + transparentFaces,
      triangles: (opaqueFaces + transparentFaces) * 2,
    },
  };
}

function bindLayout(gl, buffer) {
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  const bytes = STRIDE * Float32Array.BYTES_PER_ELEMENT;
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, bytes, 0);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 3, gl.FLOAT, false, bytes, 3 * 4);
  gl.enableVertexAttribArray(2);
  gl.vertexAttribPointer(2, 4, gl.FLOAT, false, bytes, 6 * 4);
}

function fit(canvas, gl) {
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(canvas.clientWidth * dpr));
  const height = Math.max(1, Math.round(canvas.clientHeight * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  gl.viewport(0, 0, width, height);
  return [width, height];
}

export function createVolumeRenderer(canvas) {
  const gl = canvas.getContext("webgl2", {
    alpha: false,
    antialias: true,
    depth: true,
    preserveDrawingBuffer: true,
  });
  if (!gl) return null;
  const shader = program(gl);
  const opaqueBuffer = gl.createBuffer();
  const transparentBuffer = gl.createBuffer();
  const opaqueVao = gl.createVertexArray();
  const transparentVao = gl.createVertexArray();
  gl.bindVertexArray(opaqueVao);
  bindLayout(gl, opaqueBuffer);
  gl.bindVertexArray(transparentVao);
  bindLayout(gl, transparentBuffer);
  gl.bindVertexArray(null);
  const uniforms = Object.fromEntries([
    "uCenter", "uYaw", "uPitch", "uExaggeration", "uClipScale",
    "uDepthScale", "uGrid", "uVoxelSize", "uOrigin",
  ].map((name) => [name, gl.getUniformLocation(shader, name)]));
  let decoded = null;
  let decodedDigest = null;
  let mesh = null;
  let currentMeshKey = null;
  let opaqueVertexCount = 0;
  let transparentVertexCount = 0;
  let meshBuilds = 0;

  function upload(target, vertices) {
    gl.bindBuffer(gl.ARRAY_BUFFER, target);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
    return vertices.length / STRIDE;
  }

  function drawVertices(targetVao, count) {
    if (!count) return;
    gl.bindVertexArray(targetVao);
    gl.drawArrays(gl.TRIANGLES, 0, count);
  }

  return {
    render(payload, camera, options) {
      const volume = payload.volume;
      if (!decoded || decodedDigest !== volume.semantic_sha256) {
        decoded = decode(volume);
        decodedDigest = volume.semantic_sha256;
        mesh = null;
        currentMeshKey = null;
      }
      const key = meshKey(decoded, options);
      if (!mesh || currentMeshKey !== key) {
        mesh = buildMesh(decoded, payload, options);
        currentMeshKey = key;
        meshBuilds += 1;
        opaqueVertexCount = upload(opaqueBuffer, mesh.opaque);
        transparentVertexCount = upload(transparentBuffer, mesh.transparent);
      }
      const [width, height] = fit(canvas, gl);
      const worldWidth = Number(payload.config.world_width);
      const worldDepth = Number(payload.config.world_depth);
      const exaggeration = Number(options.exaggeration);
      const diagonal = Math.hypot(worldWidth, worldDepth);
      const vertical = Number(volume.bounds.maximum[1]) * exaggeration;
      const frameSpan = Math.max(24, diagonal, payload.metrics.relief * exaggeration * 1.65);
      const defaultDistance = Math.max(worldWidth, worldDepth) * 1.48;
      const pixels = Math.min(width, height) * 0.78 / frameSpan *
        (defaultDistance / camera.dist);
      const depthSpan = Math.max(1, diagonal + vertical + decoded.unit * 12);

      gl.clearColor(0.045, 0.065, 0.071, 1);
      gl.clearDepth(1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);
      gl.depthFunc(gl.LEQUAL);
      gl.disable(gl.CULL_FACE);
      gl.useProgram(shader);
      const minimum = volume.bounds.minimum.map(Number);
      const maximum = volume.bounds.maximum.map(Number);
      gl.uniform3f(
        uniforms.uCenter,
        (minimum[0] + maximum[0]) * 0.5 + camera.panX,
        Number(payload.metrics.mean_height),
        (minimum[2] + maximum[2]) * 0.5 + camera.panZ,
      );
      gl.uniform1f(uniforms.uYaw, camera.yaw);
      gl.uniform1f(uniforms.uPitch, camera.pitch);
      gl.uniform1f(uniforms.uExaggeration, exaggeration);
      gl.uniform2f(uniforms.uClipScale, pixels * 2 / width, pixels * 2 / height);
      gl.uniform1f(uniforms.uDepthScale, 0.92 / depthSpan);
      gl.uniform1f(uniforms.uGrid, options.grid ? 1 : 0);
      gl.uniform1f(uniforms.uVoxelSize, decoded.unit);
      gl.uniform3f(uniforms.uOrigin, ...decoded.origin);

      gl.disable(gl.BLEND);
      gl.depthMask(true);
      drawVertices(opaqueVao, opaqueVertexCount);
      if (transparentVertexCount) {
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.depthMask(false);
        drawVertices(transparentVao, transparentVertexCount);
        gl.depthMask(true);
        gl.disable(gl.BLEND);
      }
      gl.bindVertexArray(null);
      return {
        ...mesh.stats,
        occupiedVoxels: volume.metrics.occupied_voxels,
        carvedCaveVoxels: volume.metrics.carved_cave_voxels,
        voxelSize: volume.voxel_size,
        dimensions: volume.dimensions,
        meshBuilds,
      };
    },
    dispose() {
      gl.deleteBuffer(opaqueBuffer);
      gl.deleteBuffer(transparentBuffer);
      gl.deleteVertexArray(opaqueVao);
      gl.deleteVertexArray(transparentVao);
      gl.deleteProgram(shader);
    },
  };
}

export { VOLUME_SCHEMA };
