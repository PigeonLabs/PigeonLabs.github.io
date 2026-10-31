// Realtime pipeline off the main thread: JPEG decode -> PIL-compatible resize -> DINOv2 (ONNX
// Runtime Web, WebGPU) -> causal tracker + A + C (engine.js). One frame per request, in order.
import { OnlineDetector, loadSceneModel, preprocess, IMAGE_SIZE } from "./engine.js";

const ORT_VERSION = "1.29.0";
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const CACHE_NAME = "cyclevad-onnx-v1";
// Same DINOv2 weights (stored fp16). "fp16" computes in half precision like the paper's CUDA
// autocast run and is ~1.6x faster on WebGPU; "fp32" is used when the GPU lacks shader-f16.
const MODEL_VARIANTS = {
  fp16: { graph: "../onnx/features16.onnx", shards: ["features16.0.bin", "features16.1.bin"] },
  fp32: { graph: "../onnx/features.onnx", shards: ["features.0.bin", "features.1.bin"] },
};

let ort = null, session = null, backend = null, loading = null;
const models = new Map();
let detector = null;
const canvas = new OffscreenCanvas(256, 256);
const ctx = canvas.getContext("2d", { willReadFrequently: true });
const status = (text) => postMessage({ type: "status", text });

async function cachedBytes(url, onBytes) {
  const absolute = new URL(url, self.location.href).href;
  let cache = null;
  try { cache = await caches.open(CACHE_NAME); } catch (_) { /* Cache API unavailable */ }
  let response = cache && (await cache.match(absolute).catch(() => null));
  const fromCache = !!response;
  if (!response) response = await fetch(absolute);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  const total = Number(response.headers.get("content-length")) || 0;
  const reader = response.clone().body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onBytes(received, total);
  }
  if (cache && !fromCache) cache.put(absolute, response).catch(() => {});
  const bytes = new Uint8Array(received);
  let at = 0;
  for (const c of chunks) { bytes.set(c, at); at += c.length; }
  return bytes;
}

async function pickVariant() {
  try {
    const adapter = self.navigator.gpu && (await self.navigator.gpu.requestAdapter());
    if (adapter && adapter.features.has("shader-f16")) return "fp16";
  } catch (_) { /* no WebGPU adapter */ }
  return "fp32";
}

async function loadRuntime() {
  status("ONNX Runtime 불러오는 중");
  ort = await import(`${ORT_BASE}ort.webgpu.min.mjs`);
  ort.env.wasm.wasmPaths = ORT_BASE;
  ort.env.wasm.numThreads = 1; // GitHub Pages cannot send cross-origin isolation headers
  const variant = await pickVariant();
  const files = MODEL_VARIANTS[variant];
  const urls = [files.graph, ...files.shards.map((s) => `../onnx/${s}`)];
  const progress = new Map();
  const report = () => {
    let got = 0, all = 0;
    for (const [g, t] of progress.values()) { got += g; all += t || g; }
    const mb = (b) => (b / 1048576).toFixed(0);
    status(`DINOv2 내려받는 중 ${all ? Math.floor(got / all * 100) : 0}% (${mb(got)}/${mb(all)} MB)`);
  };
  const [graph, ...shards] = await Promise.all(urls.map((url) =>
    cachedBytes(url, (g, t) => { progress.set(url, [g, t]); report(); })));
  const externalData = files.shards.map((path, i) => ({ path, data: shards[i] }));
  const gpu = !!self.navigator.gpu;
  status(gpu ? "WebGPU 초기화 중" : "WebGPU가 없어 CPU(WASM)로 초기화 중");
  let provider = gpu ? "webgpu" : "wasm";
  try {
    session = await ort.InferenceSession.create(graph, { executionProviders: [provider], externalData });
  } catch (error) {
    if (provider !== "webgpu") throw error;
    status("WebGPU를 쓸 수 없어 CPU(WASM)로 초기화 중");
    provider = "wasm";
    session = await ort.InferenceSession.create(graph, { executionProviders: [provider], externalData });
  }
  backend = `${provider === "webgpu" ? "WebGPU" : "WASM"} ${variant.toUpperCase()}`;
  status("셰이더 준비 중");
  const zeros = new ort.Tensor("float32", new Float32Array(3 * IMAGE_SIZE * IMAGE_SIZE), [1, 3, IMAGE_SIZE, IMAGE_SIZE]);
  await session.run({ pixel_values: zeros });
}

async function sceneModel(id) {
  if (!models.has(id)) {
    models.set(id, Promise.all([
      fetch(`../data/models/${id}.json`).then((r) => r.json()),
      fetch(`../data/models/${id}.bin`).then((r) => r.arrayBuffer()),
    ]).then(([meta, bin]) => loadSceneModel(meta, bin)));
  }
  return models.get(id);
}

async function features(bytes) {
  const bitmap = await createImageBitmap(new Blob([bytes], { type: "image/jpeg" }),
                                         { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  ctx.drawImage(bitmap, 0, 0, 256, 256);
  bitmap.close();
  const px = preprocess(ctx.getImageData(0, 0, 256, 256).data, 256, 256);
  const out = await session.run({ pixel_values: new ort.Tensor("float32", px, [1, 3, IMAGE_SIZE, IMAGE_SIZE]) });
  return { global: out.global.data, patches: out.patches.data };
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "start") {
      if (!loading) loading = loadRuntime().catch((e) => { loading = null; throw e; });
      const [model] = await Promise.all([sceneModel(data.scene), loading]);
      detector = new OnlineDetector(model);
      postMessage({ type: "ready", id: data.id, backend, stride: detector.stride });
    } else if (data.type === "step") {
      const began = performance.now();
      const fresh = data.index % detector.stride === 0;
      const feats = fresh ? await features(data.bytes) : null;
      const r = detector.step(feats, data.threshold);
      postMessage({ type: "result", id: data.id, result: r, latency: fresh ? performance.now() - began : null });
    }
  } catch (error) {
    postMessage({ type: "error", id: data.id, message: String(error && error.message ? error.message : error) });
  }
};
