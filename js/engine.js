// CycleVAD A+C online detector, ported from cyclevad_live (Python) for in-browser use.
// No DOM access here, so the same module runs in Node for parity tests.
//   preprocess   : PIL ImageOps.pad(BICUBIC) to 336x336 + DINOv2 rescale/normalize
//   OnlineDetector.step : StreamingPhysicalTracker + pooled/local appearance (A) +
//                         Fourier mean + shared residual subspace (C), score = max(A, C)

export const IMAGE_SIZE = 336;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

/* ---------------- PIL-compatible bicubic resampling (Pillow Resample.c, 8 bpc) ---------------- */

const PRECISION_BITS = 32 - 8 - 2;
const ONE = 2 ** PRECISION_BITS;

function bicubic(x) {
  const a = -0.5;
  x = Math.abs(x);
  if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
  if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
  return 0;
}

function coefficients(inSize, outSize) {
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1);
  const support = 2 * filterscale;
  const bounds = [];
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const ss = 1 / filterscale;
    const xmin = Math.max(Math.trunc(center - support + 0.5), 0);
    const xmax = Math.min(Math.trunc(center + support + 0.5), inSize) - xmin;
    const k = new Float64Array(xmax);
    let ww = 0;
    for (let x = 0; x < xmax; x++) { k[x] = bicubic((x + xmin - center + 0.5) * ss); ww += k[x]; }
    const ik = new Float64Array(xmax);
    for (let x = 0; x < xmax; x++) {
      const v = ww !== 0 ? k[x] / ww : k[x];
      ik[x] = v < 0 ? Math.trunc(-0.5 + v * ONE) : Math.trunc(0.5 + v * ONE);
    }
    bounds.push({ xmin, k: ik });
  }
  return bounds;
}

const clip8 = (ss) => (ss <= 0 ? 0 : ss >= ONE * 256 ? 255 : Math.floor(ss / ONE));

// rgba: Uint8ClampedArray (w*h*4). Returns Uint8Array RGB (out*out*3), exactly like PIL.
export function resizeBicubic(rgba, width, height, out = IMAGE_SIZE) {
  const hc = coefficients(width, out), vc = coefficients(height, out);
  const tmp = new Uint8Array(out * height * 3);
  for (let y = 0; y < height; y++) {
    for (let xx = 0; xx < out; xx++) {
      const { xmin, k } = hc[xx];
      for (let c = 0; c < 3; c++) {
        let ss = ONE / 2;
        for (let x = 0; x < k.length; x++) ss += rgba[((y * width) + x + xmin) * 4 + c] * k[x];
        tmp[(y * out + xx) * 3 + c] = clip8(ss);
      }
    }
  }
  const rgb = new Uint8Array(out * out * 3);
  for (let yy = 0; yy < out; yy++) {
    const { xmin, k } = vc[yy];
    for (let xx = 0; xx < out; xx++) {
      for (let c = 0; c < 3; c++) {
        let ss = ONE / 2;
        for (let y = 0; y < k.length; y++) ss += tmp[((y + xmin) * out + xx) * 3 + c] * k[y];
        rgb[(yy * out + xx) * 3 + c] = clip8(ss);
      }
    }
  }
  return rgb;
}

// Square frames only (IPAD frames are 256x256), so ImageOps.pad is a plain resize.
export function preprocess(rgba, width, height) {
  if (width !== height) throw new Error("Square frames expected");
  const rgb = resizeBicubic(rgba, width, height);
  const n = IMAGE_SIZE * IMAGE_SIZE;
  const pixels = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) pixels[c * n + i] = (rgb[i * 3 + c] / 255 - MEAN[c]) / STD[c];
  }
  return pixels;
}

/* ---------------- Model loading ---------------- */

export function loadSceneModel(meta, buffer) {
  const arrays = {};
  for (const [name, a] of Object.entries(meta.arrays)) {
    const count = a.shape.reduce((p, q) => p * q, 1);
    const Type = a.dtype === "float32" ? Float32Array : Float64Array;
    arrays[name] = { data: new Type(buffer, a.offset, count), shape: a.shape };
  }
  return { ...meta, a: arrays };
}

/* ---------------- Numerics ---------------- */

// numpy float32 -> float16 -> float32 rounding (round half to even)
export function f16round(v) {
  if (typeof Math.f16round === "function") return Math.f16round(v);
  if (!Number.isFinite(v) || v === 0) return v;
  const sign = v < 0 ? -1 : 1;
  const a = Math.abs(v);
  if (a >= 65520) return sign * Infinity;
  let e = Math.floor(Math.log2(a));
  if (2 ** e > a) e -= 1;
  if (2 ** (e + 1) <= a) e += 1;
  const ulp = 2 ** (Math.max(e, -14) - 10);
  const q = a / ulp, t = Math.floor(q), frac = q - t;
  const r = frac > 0.5 || (frac === 0.5 && t % 2 === 1) ? t + 1 : t;
  return sign * r * ulp;
}

function softmaxNeg(values, temperature) {
  let max = -Infinity;
  const out = new Float64Array(values.length);
  for (let i = 0; i < values.length; i++) { out[i] = -values[i] / temperature; if (out[i] > max) max = out[i]; }
  let sum = 0;
  for (let i = 0; i < out.length; i++) { out[i] = Math.exp(out[i] - max); sum += out[i]; }
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}

// sklearn-style subspace scores of x (length D): mean squared out-of-subspace residual and
// variance-normalized in-subspace coordinates. Returns the residual too (for the overlay).
function subspaceScore(x, mean, basis, eigen) {
  const D = mean.length, k = eigen.length;
  const centered = new Float64Array(D);
  for (let d = 0; d < D; d++) centered[d] = x[d] - mean[d];
  const coords = new Float64Array(k);
  for (let j = 0; j < k; j++) {
    let s = 0;
    const row = j * D;
    for (let d = 0; d < D; d++) s += centered[d] * basis[row + d];
    coords[j] = s;
  }
  const residual = centered;
  for (let j = 0; j < k; j++) {
    const c = coords[j], row = j * D;
    for (let d = 0; d < D; d++) residual[d] -= c * basis[row + d];
  }
  let outside = 0, inside = 0;
  for (let d = 0; d < D; d++) outside += residual[d] * residual[d];
  for (let j = 0; j < k; j++) inside += coords[j] * coords[j] / eigen[j];
  return { outside: outside / D, inside: inside / k, residual };
}

function searchsorted(sorted, v, right) {
  let lo = 0, hi = sorted.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (right ? sorted[mid] <= v : sorted[mid] < v) lo = mid + 1; else hi = mid;
  }
  return lo;
}

// TailCalibrator.score: -log(empirical survival) plus linear excess above the reference max.
function calibrate(reference, scale, v) {
  const n = reference.length;
  const lo = searchsorted(reference, v, false), hi = searchsorted(reference, v, true);
  const survival = (n - (lo + hi) / 2 + 1) / (n + 1);
  return -Math.log(survival) + Math.max(v - reference[n - 1], 0) / scale;
}

/* ---------------- Causal cycle tracker (StreamingPhysicalTracker) ---------------- */

class Tracker {
  constructor(model) {
    const g = model.grid, stride = model.stride, period = model.period;
    this.m = model;
    this.g = g;
    this.stride = stride;
    const half = Math.floor(g / 4);
    this.offsets = [];
    for (let o = -half; o <= half; o++) this.offsets.push(o);
    const sigma = Math.sqrt(stride / 2) * Math.max(1, Math.abs(g * 2 / period) * 0.5);
    const drift = g * stride / period;
    const raw = this.offsets.map((o) => -0.5 * ((o - drift) / sigma) ** 2);
    const mx = Math.max(...raw);
    let s = 0;
    this.weights = raw.map((r) => { const e = Math.exp(r - mx); s += e; return e; }).map((e) => e / s);
    this.restart = 1 - (1 - 0.001) ** (stride / 2);
    this.cos = new Float64Array(g);
    this.sin = new Float64Array(g);
    for (let i = 0; i < g; i++) { this.cos[i] = Math.cos(2 * Math.PI * i / g); this.sin[i] = Math.sin(2 * Math.PI * i / g); }
    this.reset();
  }

  reset() {
    this.posterior = new Float64Array(this.g).fill(1 / this.g);
    this.previous = new Float64Array(this.g).fill(1 / this.g);
    this.history = [];
    this.count = 0;
  }

  // scipy.ndimage.convolve1d(mode='wrap') with an odd-length kernel
  convolve(input) {
    const g = this.g, w = this.weights, L = w.length, origin = Math.floor(L / 2);
    const out = new Float64Array(g);
    for (let i = 0; i < g; i++) {
      let s = 0;
      for (let j = 0; j < L; j++) s += w[j] * input[((i + origin - j) % g + g) % g];
      out[i] = s;
    }
    return out;
  }

  step(x, index) {
    if (index !== this.count * this.stride) throw new Error("Out-of-order source index");
    const g = this.g, template = this.m.a.template.data, dim = x.length;
    this.history.push(x);
    if (this.history.length > Math.floor(4 / this.stride) + 1) this.history.shift();
    const old = this.history[0];
    const encoded = new Float64Array(2 * dim);
    for (let d = 0; d < dim; d++) { encoded[d] = x[d]; encoded[dim + d] = 0.5 * (x[d] - old[d]); }
    const distance = new Float64Array(g);
    for (let t = 0; t < g; t++) {
      let s = 0;
      for (let d = 0; d < 2 * dim; d++) { const diff = encoded[d] - template[t * 2 * dim + d]; s += diff * diff; }
      distance[t] = s / (2 * dim);
    }
    const like = softmaxNeg(distance, this.m.temperature);
    let prior = this.posterior;
    if (this.count) {
      prior = this.convolve(this.posterior);
      for (let i = 0; i < g; i++) prior[i] = (1 - this.restart) * prior[i] + this.restart / g;
    }
    const post = new Float64Array(g);
    let total = 0;
    for (let i = 0; i < g; i++) { post[i] = prior[i] * like[i]; total += post[i]; }
    total = Math.max(total, 1e-300);
    let re = 0, im = 0;
    for (let i = 0; i < g; i++) { post[i] /= total; re += post[i] * this.cos[i]; im += post[i] * this.sin[i]; }
    this.posterior = post;
    this.previous = like;
    this.count += 1;
    const angle = Math.atan2(im, re) / (2 * Math.PI);
    return ((angle % 1) + 1) % 1;
  }
}

/* ---------------- Detector ---------------- */

export class OnlineDetector {
  constructor(model) {
    this.m = model;
    this.stride = model.stride;
    this.tracker = new Tracker(model);
    this.reset();
  }

  reset() {
    this.tracker.reset();
    this.count = 0;
    this.last = null;
  }

  // features: { global: Float32Array(768), patches: Float32Array(36*64) } on observation frames
  step(features, threshold) {
    const index = this.count++;
    if (index % this.stride) {
      if (!this.last) throw new Error("No held decision");
      return { ...this.last, index, fresh: false, alarm: this.last.score > threshold };
    }
    const m = this.m, a = m.a;
    const G = features.global.length, P = features.patches.length, cells = 36, pd = P / cells;
    const D = G + P;
    // descriptor(): global / sqrt(2), patches / sqrt(2 * cells), after fp16 rounding (cache format)
    const z = new Float64Array(D);
    const gs = Math.fround(Math.SQRT2), ps = Math.fround(Math.sqrt(2 * cells));
    for (let i = 0; i < G; i++) z[i] = Math.fround(f16round(features.global[i]) / gs);
    const patches = new Float64Array(P);
    for (let i = 0; i < P; i++) { patches[i] = f16round(features.patches[i]); z[G + i] = Math.fround(patches[i] / ps); }

    // tracker input: PCA projection divided by per-axis scale
    const comps = a.reducer_components.data, mean = a.reducer_mean.data, scale = a.tracker_scale.data;
    const k = scale.length, x = new Float64Array(k);
    for (let j = 0; j < k; j++) {
      let s = 0;
      for (let d = 0; d < D; d++) s += (z[d] - mean[d]) * comps[j * D + d];
      x[j] = s / scale[j];
    }
    const theta = this.tracker.step(x, index);

    // A: pooled subspace (outside / inside) + local memory (top-10% region distance)
    const pooled = subspaceScore(z, a.pooled_mean.data, a.pooled_basis.data, a.pooled_eigen.data);
    const bank = a.local_bank.data, nb = a.local_bank.shape[0];
    const dist = new Float64Array(cells);
    for (let c = 0; c < cells; c++) {
      let xx = 0;
      for (let d = 0; d < pd; d++) xx += patches[c * pd + d] ** 2;
      let best = Infinity;
      for (let b = 0; b < nb; b++) {
        let bb = 0, xb = 0;
        const off = (b * cells + c) * pd;
        for (let d = 0; d < pd; d++) { bb += bank[off + d] ** 2; xb += patches[c * pd + d] * bank[off + d]; }
        best = Math.min(best, Math.max(xx + bb - 2 * xb, 0));
      }
      dist[c] = best;
    }
    const top = Math.max(1, Math.ceil(cells * 0.1));
    const local = Array.from(dist).sort((p, q) => p - q).slice(-top).reduce((s, v) => s + v, 0) / top;
    const sc = m.calibrator_scales;
    const A = Math.max(calibrate(a.cal_app_pooled.data, sc.app_pooled, pooled.outside),
                       calibrate(a.cal_app_inside.data, sc.app_inside, pooled.inside),
                       calibrate(a.cal_app_local.data, sc.app_local, local));

    // C: residual from the Fourier mean mu(theta), scored in the shared residual subspace
    const h = m.harmonics, coef = a.coef.data, basis = [1];
    for (let j = 1; j <= h; j++) basis.push(Math.cos(2 * Math.PI * theta * j));
    for (let j = 1; j <= h; j++) basis.push(Math.sin(2 * Math.PI * theta * j));
    const r = new Float64Array(D);
    for (let d = 0; d < D; d++) {
      let mu = 0;
      for (let j = 0; j < basis.length; j++) mu += basis[j] * coef[j * D + d];
      r[d] = z[d] - mu;
    }
    const cond = subspaceScore(r, a.space_mean.data, a.space_basis.data, a.space_eigen.data);
    const C = Math.max(calibrate(a.cal_c_outside.data, sc.c_outside, cond.outside),
                       calibrate(a.cal_c_inside.data, sc.c_inside, cond.inside));

    // overlay: per-region energy of C's residual vs. its normal statistics
    const med = a.overlay_median.data, spread = a.overlay_spread.data;
    const overlay = new Array(cells);
    for (let c = 0; c < cells; c++) {
      let s = 0;
      for (let d = 0; d < pd; d++) s += cond.residual[G + c * pd + d] ** 2;
      overlay[c] = (Math.sqrt(s) - med[c]) / spread[c];
    }
    const score = Math.max(A, C);
    this.last = { index, fresh: true, theta, A, C, score, alarm: score > threshold, cells: overlay };
    return this.last;
  }
}
