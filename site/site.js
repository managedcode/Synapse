// Synapse landing: page interactions and the 3D "fly brain" model graph.
// The page reads fine without this file; the scene loads three.js on demand.
const THREE_URL = "https://cdn.jsdelivr.net/npm/three@0.186.1/build/three.module.min.js";
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

// ---------- Page interactions ----------
const nav = document.querySelector(".nav");
const sheet = document.querySelector(".sheet");
const syncNav = () => { nav.dataset.theme = sheet.getBoundingClientRect().top <= 68 ? "light" : "dark"; };
addEventListener("scroll", syncNav, { passive: true });
syncNav();

const revealer = new IntersectionObserver((entries) => {
  for (const e of entries) if (e.isIntersecting) { e.target.classList.add("in"); revealer.unobserve(e.target); }
}, { rootMargin: "0px 0px -12% 0px" });
document.querySelectorAll(".reveal").forEach((el) => revealer.observe(el));

const links = [...document.querySelectorAll(".nav-links a")];
const spy = new IntersectionObserver((entries) => {
  for (const e of entries) if (e.isIntersecting) links.forEach((a) => a.classList.toggle("active", a.hash === `#${e.target.id}`));
}, { rootMargin: "-45% 0px -50% 0px" });
links.forEach((a) => { const s = document.querySelector(a.hash); if (s) spy.observe(s); });

for (const button of document.querySelectorAll(".copy")) {
  button.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(button.closest(".term").querySelector("code").innerText);
      button.textContent = "Copied"; button.classList.add("done");
    } catch { button.textContent = "Select to copy"; }
    setTimeout(() => { button.textContent = "Copy"; button.classList.remove("done"); }, 1600);
  });
}

const speedLlama = { 2: 133.7, 8: 167.2 };
for (const button of document.querySelectorAll("[data-threads]")) {
  button.addEventListener("click", () => {
    const t = button.dataset.threads;
    document.querySelectorAll("[data-threads]").forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
    const cells = [...document.querySelectorAll('[data-chart="speed"] em')];
    const values = cells.map((c) => Number(c.dataset[`v${t}`]));
    const max = Math.max(...values);
    cells.forEach((c, i) => { c.textContent = values[i].toFixed(1); c.parentElement.querySelector("b").style.setProperty("--w", `${(values[i] / max) * 100}%`); });
    document.querySelector("[data-ratio]").textContent = `${Math.round((values[0] / speedLlama[t]) * 100)}%`;
  });
}

// ---------- 3D scene ----------
const LAYERS = 12;                         // transformer blocks, plus one output head
const STATES = {
  hero: { wall: 0, cold: 0, split: 0, yaw: -0.5, zoom: 1 },
  wall: { wall: 1, cold: 0, split: 0, yaw: -0.3, zoom: 1.04 },
  wave: { wall: 0, cold: 0, split: 0, yaw: -0.62, zoom: 1.04 },
  cold: { wall: 0, cold: 1, split: 0, yaw: -0.9, zoom: 1.02 },
  split: { wall: 0, cold: 0.45, split: 1, yaw: -0.18, zoom: 0.86 },
};
const OFFSETS = [[-1.25, 0.45, 0], [0, -0.45, 0], [1.25, 0.45, 0]];

function random(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Regions: per block one attention cluster and four experts, then the output head.
function buildRegions() {
  const regions = [];
  for (let l = 0; l <= LAYERS; l++) {
    const t = l / LAYERS;
    const spine = [(t - 0.5) * 8.6, Math.sin(t * Math.PI) * 0.9 - 0.35, Math.cos(t * Math.PI * 1.4) * 0.6];
    const group = l < 4 ? 0 : l < 8 ? 1 : 2;
    const count = l === LAYERS ? 1 : 5;
    for (let r = 0; r < count; r++) {
      const a = (r - 1) * (Math.PI / 2) + Math.PI / 4;
      const off = r === 0 ? [0, 0, 0] : [0, Math.sin(a) * 1.15, Math.cos(a) * 1.15];
      regions.push({ layer: l, region: r, group, center: spine.map((v, i) => v + off[i]) });
    }
  }
  return regions;
}

const COMMON = /* glsl */`
  uniform float uTime, uFront, uWall, uCold, uSplit, uReduce;
  uniform vec3 uOff0, uOff1, uOff2;
  attribute float aLayer, aRegion, aGroup, aSeed, aSel;
  varying vec3 vColor; varying float vAlpha;
  vec3 groupOffset() { return (aGroup < 0.5 ? uOff0 : aGroup < 1.5 ? uOff1 : uOff2) * uSplit; }
  float activation() {
    float d = aLayer / 12.0 - uFront;
    float wave = max(exp(-d * d * 120.0), d < 0.0 ? 0.7 * exp(d * 2.2) : 0.0);
    float a = mix(aSel * wave, aSel * 0.75, uReduce);
    float heat = 0.55 + 0.3 * sin(uTime * 2.6 + aSeed * 40.0);
    return max(a, uWall * max(heat, wave));
  }
  vec3 activeColor() {
    if (aLayer > 11.5) return vec3(1.0, 0.74, 0.38);
    return aRegion < 0.5 ? vec3(1.0, 0.54, 0.3) : vec3(0.93, 0.25, 0.5);
  }
  vec3 place(float cold) {
    vec3 p = position + groupOffset();
    p.y -= cold * 0.5;
    return p + 0.02 * vec3(sin(uTime * 0.7 + aSeed * 12.0), cos(uTime * 0.6 + aSeed * 9.0), 0.0);
  }`;

const POINT_VERTEX = `${COMMON}
  uniform float uPixel;
  void main() {
    float a = activation();
    float cold = uCold * (1.0 - aSel) * (1.0 - uWall);
    vec4 mv = modelViewMatrix * vec4(place(cold), 1.0);
    gl_Position = projectionMatrix * mv;
    vColor = mix(mix(vec3(0.66, 0.61, 0.56), vec3(0.33, 0.39, 0.52), cold), activeColor(), clamp(a, 0.0, 1.0));
    vAlpha = mix(0.34, 0.06, cold) + a * 0.9;
    gl_PointSize = (1.0 + a * 1.7) * (1.0 - 0.45 * cold) * (0.7 + aSeed * 0.6) * uPixel * 105.0 / -mv.z;
  }`;
const POINT_FRAGMENT = `
  varying vec3 vColor; varying float vAlpha;
  void main() {
    float d = length(gl_PointCoord - 0.5) * 2.0;
    if (d > 1.0) discard;
    gl_FragColor = vec4(vColor, vAlpha * (pow(1.0 - d, 2.0) * 0.8 + smoothstep(0.35, 0.0, d) * 0.6));
  }`;
const LINE_VERTEX = `${COMMON}
  void main() {
    float a = activation();
    float cold = uCold * (1.0 - aSel) * (1.0 - uWall);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(place(cold), 1.0);
    vColor = mix(vec3(0.7, 0.66, 0.6), activeColor(), clamp(a, 0.0, 1.0));
    vAlpha = mix(0.09, 0.015, cold) + a * 0.65;
  }`;
const LINE_FRAGMENT = `varying vec3 vColor; varying float vAlpha; void main() { gl_FragColor = vec4(vColor, vAlpha); }`;

function attributes(THREE, geometry, rows) {
  const names = ["aLayer", "aRegion", "aGroup", "aSeed", "aSel"];
  names.forEach((name, k) => geometry.setAttribute(name, new THREE.Float32BufferAttribute(rows.map((r) => r[k]), 1)));
}

function buildGraph(THREE, rand, uniforms) {
  const gauss = () => Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());
  const regions = buildRegions();
  const pos = [], rows = [];
  for (const g of regions) {
    const head = g.layer === LAYERS;
    const n = head ? 70 : g.region === 0 ? 44 : 26;
    const s = head ? 0.3 : g.region === 0 ? 0.26 : 0.2;
    for (let i = 0; i < n; i++) {
      pos.push(g.center[0] + gauss() * s * 0.55, g.center[1] + gauss() * s, g.center[2] + gauss() * s);
      rows.push([g.layer, g.region, g.group, rand(), 1]);
    }
  }
  const pointGeometry = new THREE.BufferGeometry();
  pointGeometry.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  attributes(THREE, pointGeometry, rows);

  const lpos = [], lrows = [];
  const link = (a, b) => { for (const g of [a, b]) { lpos.push(...g.center); lrows.push([g.layer, g.region, g.group, 0.5, 1]); } };
  for (const a of regions) {
    for (const b of regions) {
      if (b.layer === a.layer + 1) link(a, b);
      if (b.layer === a.layer && a.region === 0 && b.region > 0) link(a, b);
    }
  }
  const lineGeometry = new THREE.BufferGeometry();
  lineGeometry.setAttribute("position", new THREE.Float32BufferAttribute(lpos, 3));
  attributes(THREE, lineGeometry, lrows);

  const common = { uniforms, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending };
  const points = new THREE.Points(pointGeometry, new THREE.ShaderMaterial({ ...common, vertexShader: POINT_VERTEX, fragmentShader: POINT_FRAGMENT }));
  const lines = new THREE.LineSegments(lineGeometry, new THREE.ShaderMaterial({ ...common, vertexShader: LINE_VERTEX, fragmentShader: LINE_FRAGMENT }));
  return { regions, points, lines, rows, lrows };
}

function buildMachines(THREE, regions) {
  return [0, 1, 2].map((group) => {
    const box = new THREE.Box3();
    regions.filter((r) => r.group === group).forEach((r) => box.expandByPoint(new THREE.Vector3(...r.center)));
    box.expandByScalar(0.55);
    const size = box.getSize(new THREE.Vector3());
    const mesh = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(size.x, size.y, size.z)),
      new THREE.LineBasicMaterial({ color: 0xf3f1ee, transparent: true, opacity: 0, depthWrite: false }));
    mesh.userData.home = box.getCenter(new THREE.Vector3());
    return mesh;
  });
}

async function startScene() {
  const canvas = document.getElementById("brain");
  const stage = document.querySelector(".stage");
  if (!canvas) return;
  const THREE = await import(THREE_URL);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "high-performance" });
  const pixel = Math.min(devicePixelRatio || 1, 2);
  renderer.setPixelRatio(pixel);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
  camera.position.set(0, 0.5, 13);
  const rig = new THREE.Group();
  const model = new THREE.Group();
  rig.add(model);
  scene.add(rig);

  const rand = random(7);
  const uniforms = {
    uTime: { value: 0 }, uFront: { value: -0.3 }, uWall: { value: 0 }, uCold: { value: 0 }, uSplit: { value: 0 },
    uReduce: { value: reduceMotion ? 1 : 0 }, uPixel: { value: pixel },
    uOff0: { value: new THREE.Vector3(...OFFSETS[0]) }, uOff1: { value: new THREE.Vector3(...OFFSETS[1]) }, uOff2: { value: new THREE.Vector3(...OFFSETS[2]) },
  };
  const graph = buildGraph(THREE, rand, uniforms);
  const machines = buildMachines(THREE, graph.regions);
  model.add(graph.lines, graph.points, ...machines);

  const dust = new THREE.BufferGeometry();
  dust.setAttribute("position", new THREE.Float32BufferAttribute(Array.from({ length: 1500 }, (_, i) => (rand() - 0.5) * [22, 12, 12][i % 3]), 3));
  const dustPoints = new THREE.Points(dust, new THREE.PointsMaterial({ color: 0xffd9c7, size: 0.025, transparent: true, opacity: 0.35, depthWrite: false }));
  scene.add(dustPoints);

  // Each wave pass the router picks one expert per block; attention and the head always run.
  const route = new Array(LAYERS).fill(1);
  const pickRoute = () => {
    for (let l = 0; l < LAYERS; l++) route[l] = 1 + Math.floor(rand() * 4);
    const sel = (r) => (r[1] === 0 || r[1] === route[r[0]] ? 1 : 0);
    for (const [geometry, rows] of [[graph.points.geometry, graph.rows], [graph.lines.geometry, graph.lrows]]) {
      const attr = geometry.getAttribute("aSel");
      rows.forEach((r, i) => { attr.array[i] = sel(r); });
      attr.needsUpdate = true;
    }
  };
  pickRoute();

  const cur = { ...STATES.hero };
  let target = STATES.hero;
  const dots = [...document.querySelectorAll(".stage-dots li")];
  const setState = (name) => {
    target = STATES[name] ?? STATES.hero;
    stage.dataset.state = name;
    dots.forEach((d) => d.classList.toggle("on", d.dataset.for === name));
  };
  setState("hero");
  const stepWatch = new IntersectionObserver((entries) => {
    for (const e of entries) if (e.isIntersecting) setState(e.target.dataset.state);
  }, { rootMargin: "-45% 0px -45% 0px" });
  document.querySelectorAll(".step[data-state]").forEach((s) => stepWatch.observe(s));

  const layout = () => {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    const wide = w > 760;
    rig.position.set(wide ? 1.35 : 0, wide ? 0.1 : 2.5, wide ? 0 : -2);
    rig.userData.scale = wide ? Math.min(0.84, w / 1600 + 0.05) : Math.min(0.9, (w / h) * 0.88);
  };
  addEventListener("resize", layout);
  layout();

  const mouse = { x: 0, y: 0, sx: 0, sy: 0 };
  addEventListener("pointermove", (e) => { mouse.x = (e.clientX / innerWidth) * 2 - 1; mouse.y = (e.clientY / innerHeight) * 2 - 1; }, { passive: true });

  const scrolly = document.querySelector(".scrolly");

  let last = performance.now(), elapsed = 0;
  const frame = (now) => {
    requestAnimationFrame(frame);
    const dt = Math.min((now - last) / 1000, 0.05);
    last = now;
    if (scrolly.getBoundingClientRect().bottom < 0) return;
    elapsed += dt;
    const t = reduceMotion ? 0 : elapsed;
    const k = reduceMotion ? 1 : 1 - Math.exp(-dt * 2.4);
    for (const key of Object.keys(cur)) cur[key] += (target[key] - cur[key]) * k;
    if (!reduceMotion) {
      uniforms.uFront.value += dt / 3.2;
      if (uniforms.uFront.value > 1.5) { uniforms.uFront.value = -0.3; pickRoute(); }
    }
    uniforms.uTime.value = t;
    uniforms.uWall.value = cur.wall;
    uniforms.uCold.value = cur.cold;
    uniforms.uSplit.value = cur.split;
    machines.forEach((m, i) => {
      m.position.copy(m.userData.home).addScaledVector(uniforms[`uOff${i}`].value, cur.split);
      m.material.opacity = cur.split * 0.16;
    });
    mouse.sx += (mouse.x - mouse.sx) * 0.05;
    mouse.sy += (mouse.y - mouse.sy) * 0.05;
    const drift = reduceMotion ? 0 : 1;
    model.rotation.set(0.12 + mouse.sy * 0.08 * drift, cur.yaw + (mouse.sx * 0.2 + Math.sin(t * 0.15) * 0.08) * drift, 0);
    rig.scale.setScalar(cur.zoom * rig.userData.scale);
    dustPoints.rotation.y = t * 0.01;
    renderer.render(scene, camera);
  };
  requestAnimationFrame(frame);
}

startScene().catch(() => document.documentElement.classList.add("no-webgl"));
