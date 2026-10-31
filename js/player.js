// Plays one demo video in either mode and reports frames to the UI in source order.
//   "preprocessed": replays outputs recorded by the Python online detector (frame by frame).
//   "realtime":     a Web Worker decodes each JPEG and runs DINOv2 (ONNX Runtime Web, WebGPU) +
//                   engine.js; this thread only paces playback and draws.
// Frame files are downloaded a few frames ahead only to hide network latency; the detector is
// stepped strictly in order and never receives a frame before the current one.
const PREFETCH = 6;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const frameUrl = (key, i) => `data/frames/${key}/${String(i).padStart(3, "0")}.jpg`;

class RealtimeWorker {
  constructor(onStatus) {
    this.worker = new Worker(new URL("./realtime-worker.js?v=4", import.meta.url), { type: "module" });
    this.pending = new Map();
    this.next = 0;
    this.worker.onmessage = ({ data }) => {
      if (data.type === "status") return onStatus(data.text);
      const waiter = this.pending.get(data.id);
      if (!waiter) return;
      this.pending.delete(data.id);
      if (data.type === "error") waiter.reject(new Error(data.message)); else waiter.resolve(data);
    };
  }

  request(message, transfer = []) {
    const id = ++this.next;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ ...message, id }, transfer);
    });
  }
}

export class Player {
  constructor(handlers) {
    this.h = handlers;
    this.realtime = null;
    this.token = 0;
    this.fps = 30;
    this.paused = false;
    this.resumeWaiters = [];
  }

  stop() {
    this.token += 1;
    this.resume();
  }

  pause() { this.paused = true; }

  resume() {
    this.paused = false;
    this.resumeWaiters.splice(0).forEach((resolve) => resolve());
  }

  setFps(fps) { this.fps = fps; this.reanchor = true; }

  play({ video, scene, mode, fps, threshold }) {
    this.stop();
    this.fps = fps;
    const token = this.token;
    this.#run(token, video, scene, mode, threshold).catch((error) => {
      if (token === this.token) this.h.onError(error);
    });
  }

  async #run(token, video, scene, mode, threshold) {
    const live = () => token === this.token;
    const recorded = await fetch(`data/outputs/${video.key}.json`).then((r) => r.json());
    let worker = null, backend = null;
    if (mode === "realtime") {
      if (!this.realtime) this.realtime = new RealtimeWorker((text) => this.h.onStatus(text));
      worker = this.realtime;
      ({ backend } = await worker.request({ type: "start", scene: scene.id }));
    }
    if (!live()) return;
    const n = video.frames;
    this.h.onMeta({ video: video.id, frames: n, thresholds: scene.thresholds, stride: scene.stride,
                    period: scene.period, labeled: video.labeled, mode, backend });

    const pending = new Map();
    const fetchFrame = (i) => {
      if (i < n && !pending.has(i)) pending.set(i, fetch(frameUrl(video.key, i)).then((r) => r.arrayBuffer()));
    };
    let clock = performance.now(), played = 0;
    const recent = [];
    for (let i = 0; i < n; i++) {
      if (!live()) return;
      if (this.paused) {
        await new Promise((resolve) => this.resumeWaiters.push(resolve));
        if (!live()) return;
        clock = performance.now(); played = 0;
      }
      for (let k = i; k <= i + PREFETCH; k++) fetchFrame(k);
      const bytes = await pending.get(i);
      pending.delete(i);
      const truth = recorded.frames[i].gt;
      let header;
      if (worker) {
        const fresh = i % scene.stride === 0;
        const reply = await worker.request({ type: "step", index: i, threshold, bytes: fresh ? bytes.slice(0) : null });
        const r = reply.result;
        header = { i, fresh: r.fresh, theta: r.theta, A: r.A, C: r.C, score: r.score, gt: truth };
        if (r.fresh) { header.cells = r.cells; header.latency = reply.latency; }
      } else {
        header = { ...recorded.frames[i], i };
      }
      if (!live()) return;
      if (this.fps > 0) {
        if (this.reanchor) { clock = performance.now(); played = 0; this.reanchor = false; }
        played += 1;
        const delay = clock + (played / this.fps) * 1000 - performance.now();
        if (delay > 0) await sleep(delay);
        else if (delay < -500) { clock = performance.now(); played = 0; } // fell behind: re-anchor
      }
      const now = performance.now();
      recent.push(now);
      if (recent.length > 31) recent.shift();
      header.fps = recent.length > 1 ? (recent.length - 1) / ((recent.at(-1) - recent[0]) / 1000) : 0;
      if (!live()) return;
      this.h.onFrame(header, new Uint8Array(bytes));
    }
    if (live()) this.h.onDone(video.id);
  }
}
