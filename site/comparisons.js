// REQ-WEB-003: measured series, explicit scopes and an always-visible Synapse.
const byId = id => document.getElementById(id);
const format = value => value === null ? "—" : new Intl.NumberFormat("en", { maximumFractionDigits: 2 }).format(value);
const statusLabel = value => value.replaceAll("_", " ");
function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
const engineNames = { synapse: "Synapse", llamacpp: "llama.cpp", llamasharp: "LLamaSharp", dotllm: "dotLLM" };
const modelNames = { "qwen2.5-0.5b": "Qwen2.5 0.5B", "qwen2.5-7b": "Qwen2.5 7B",
  "qwen3-0.6b": "Qwen3 0.6B", "phi-3.5-mini": "Phi-3.5 mini", "phi-4-mini": "Phi-4 mini",
  "mistral-7b-v0.2": "Mistral 7B v0.2", "deepseek-r1-7b": "DeepSeek R1 7B" };
const family = row => row.model?.family ?? "Model identity not reported";
const runnerClass = row => row.runner.replace(/ \(.+\)$/, "");
const scenarioKey = row => `${row.scenario} · turn ${row.turn}`;
const isFoundry = row => row.cohort.startsWith("Foundry Local");
const isCpu = row => row.cohort.startsWith("GGUF CPU");
const isSynapse = row => row.subject === "synapse";
const engine = row => isFoundry(row) ? "Foundry Local" : engineNames[row.subject] ?? row.subject;
const scopeName = row => isFoundry(row) ? "ONNX · resident CPU" : isCpu(row) ? "GGUF · CPU" : "MLX · Metal GPU";
const byEngine = (a, b) => Number(isSynapse(b)) - Number(isSynapse(a)) || engine(a).localeCompare(engine(b));
const outputLabel = row => row.output_state.includes("reasoning only") ? "reasoning only; no final answer"
  : row.output_state === "matched" ? "exact output matched" : row.output_state.includes("mismatch") ? "output mismatch" : "quality unreviewed";

function options(id, values, label, preferred) {
  const select = byId(id);
  const previous = select.value;
  const unique = [...new Set(values)].sort();
  select.replaceChildren(...unique.map(value => {
    const option = element("option", label(value));
    option.value = value;
    return option;
  }));
  select.value = unique.includes(previous) ? previous : unique.includes(preferred) ? preferred : unique[0] ?? "";
}

function bar(row, field, max, unit) {
  const item = element("li", undefined, `comparison-series${isSynapse(row) ? " synapse-series" : ""}`);
  const name = element("span", engine(row), "series-name");
  const track = element("span", undefined, "series-track");
  const fill = element("span", undefined, "series-fill");
  const value = row[field];
  fill.style.width = `${value === null ? 0 : max === 0 ? 0 : (value / max) * 100}%`;
  track.append(fill);
  const number = element("strong", value === null ? "Not reported" : `${format(value)} ${unit}`, "series-value");
  item.append(name, number, track, element("small", `${scopeName(row)} · ${format(row.samples)} samples · ${format(row.output_tokens)} output tokens · ${outputLabel(row)}`, "series-context"));
  return item;
}

function chart(title, field, unit, direction, rows, note, expectedSynapse) {
  const card = element("article", undefined, "comparison-card");
  card.append(element("h4", title), element("p", `${unit} · ${direction} is better`, "chart-unit"));
  const max = Math.max(0, ...rows.map(row => row[field] ?? 0));
  const list = element("ul", undefined, "comparison-bars");
  rows.sort(byEngine).forEach(row => list.append(bar(row, field, max, unit)));
  if (expectedSynapse && !rows.some(isSynapse)) {
    const missing = element("li", undefined, "comparison-series synapse-series series-missing");
    missing.append(element("span", "Synapse", "series-name"), element("strong", "Not measured", "series-value"));
    list.prepend(missing);
  }
  if (!rows.length) card.append(element("p", "No source reported this metric.", "metric-note"));
  card.append(list, element("p", note, "chart-note"));
  return card;
}

function renderCharts(prefix, rows) {
  const target = byId(`${prefix}-charts`);
  target.replaceChildren();
  const expected = prefix !== "metric";
  if (!rows.length) { target.append(element("p", "No measurements for this selection.")); return; }
  target.append(chart("Time to first token", "ttft_milliseconds", "ms", "lower", [...rows],
    "Wait for the first generated token. Cache and model-load policies are listed under measurement details.", expected));
  const decode = rows.filter(row => row.decode_metric_scope === "reported_decode");
  target.append(chart("Writing speed", "decode_tokens_per_second", "tokens/s", "higher", decode,
    "Reported generation phase after the first token. A token limit or fast answer does not prove answer quality.", expected));
  const native = rows.filter(row => row.decode_metric_scope === "native_eval");
  if (native.length) target.append(chart("Native evaluation speed", "decode_tokens_per_second", "tokens/s", "higher", native,
    "llama.cpp reports an internal evaluation phase. Its rate has a different definition from the writing-speed chart.", false));
  for (const scope of ["fresh_process", "request"]) {
    const scoped = rows.filter(row => row.wall_metric_scope === scope);
    if (!scoped.length) continue;
    target.append(chart(scope === "request" ? "Resident model: full answer" : "Fresh process: full answer", "wall_milliseconds", "ms", "lower", [...scoped],
      scope === "request" ? "Already-loaded model; request wall time. Compare only within this chart." : "Includes starting a process and loading its model; compare only within this chart.", false));
    const foundryMemory = scoped.some(isFoundry);
    target.append(chart(scope === "request" && foundryMemory ? "Resident process memory" : "Peak process memory", "peak_rss_mib", "MiB", "lower", [...scoped],
      scope === "request" && foundryMemory ? "Foundry reports resident bytes sampled during requests. This differs from Synapse peak RSS; compare only within this chart." : "Peak resident memory of the complete process, including mapped weights; not just KV memory.", false));
  }
}

function renderEvidence(prefix, rows, run) {
  const target = byId(`${prefix}-evidence`);
  target.replaceChildren();
  const details = element("details", undefined, "measurement-details");
  details.append(element("summary", "Measurement details · model, quality and source"));
  const list = element("ul");
  for (const row of [...rows].sort(byEngine)) {
    const item = element("li");
    item.append(element("strong", engine(row)), element("p", row.model?.label ?? "Model identity unavailable"));
    item.append(element("p", row.model?.execution?.replaceAll("_", " ") ?? row.cohort));
    item.append(element("p", `Runner: ${row.model?.hardware ?? row.runner}`));
    item.append(element("p", `Output: ${format(row.output_tokens)} tokens · ${row.output_state === "matched" ? "exact smoke output matched" : statusLabel(row.output_state)} · ${format(row.samples)} measured samples`));
    const source = element("a", `Raw artifact: ${row.source_artifact}`);
    source.href = run.html_url;
    item.append(source);
    if (row.model) item.append(element("p", `Weights SHA-256: ${row.model.weights_sha256}`, "metric-fingerprint"),
      element("p", `Scenario SHA-256: ${row.model.scenario_sha256}`, "metric-fingerprint"));
    list.append(item);
  }
  details.append(list);
  target.append(details);
}

function selectionMessage(rows, prefix) {
  const synapse = rows.find(isSynapse);
  const allMatched = rows.length && rows.every(row => row.output_state === "matched");
  const sameOutput = new Set(rows.map(row => row.output_tokens)).size === 1;
  const parts = [allMatched ? "Exact short-output parity checked." : "Answer quality is unreviewed in this scenario."];
  const reasoning = rows.filter(row => row.output_state.includes("reasoning only")).map(engine);
  if (reasoning.length) parts.push(`${reasoning.join(", ")}: reasoning only; no final answer was recorded.`);
  if (!sameOutput) parts.push("Output lengths differ; timings cover different amounts of work.");
  if (!synapse) parts.push("Synapse was not measured for this model and runner selection; no comparison result is available.");
  if (prefix === "foundry") parts.push("Same runner class, separate jobs; identical physical hardware is not established. ONNX vs GGUF weights, threads and cache policies differ. This is a contextual comparison, not a speedup verdict.");
  if (prefix === "mlx") parts.push("Synapse CPU vs MLX Metal GPU, different weights and cache policies. Synapse Metal is not measured on this hosted runner; see the dated local Apple GPU comparison below.");
  byId(`${prefix}-selection`).textContent = parts.join(" ");
}

function comparisonControls(prefix, primary, all, run) {
  const update = changed => {
    options(`${prefix}-model`, primary.map(family), value => modelNames[value] ?? value, "qwen2.5-0.5b");
    const model = byId(`${prefix}-model`).value;
    const modelRows = primary.filter(row => family(row) === model);
    if (changed === "model") byId(`${prefix}-runner`).value = "";
    options(`${prefix}-runner`, modelRows.map(runnerClass), value => value, "macOS 15 ARM64");
    const runner = byId(`${prefix}-runner`).value;
    const runnerRows = modelRows.filter(row => runnerClass(row) === runner);
    if (changed === "model" || changed === "runner") byId(`${prefix}-scenario`).value = "";
    options(`${prefix}-scenario`, runnerRows.map(scenarioKey), value => value, "128-token answer · turn 1");
    const scenario = byId(`${prefix}-scenario`).value;
    const primaryScenario = runnerRows.find(row => scenarioKey(row) === scenario)?.model?.scenario_sha256;
    const selected = all.filter(row => family(row) === model && runnerClass(row) === runner && scenarioKey(row) === scenario &&
      (prefix === "metric" || !isSynapse(row) || (primaryScenario && primaryScenario !== "not_reported" && row.model?.scenario_sha256 === primaryScenario)) &&
      (prefix !== "foundry" || isFoundry(row) || isSynapse(row)));
    selectionMessage(selected, prefix);
    renderCharts(prefix, selected);
    renderEvidence(prefix, selected, run);
  };
  const form = byId(`${prefix}-filters`);
  form.hidden = false;
  form.addEventListener("submit", event => event.preventDefault());
  form.addEventListener("change", event => update(event.target.id.replace(`${prefix}-`, "")));
  update();
}

export function renderComparisons(rows, run) {
  const cpu = rows.filter(isCpu);
  if (cpu.length) comparisonControls("metric", cpu, cpu, run);
  const foundry = rows.filter(isFoundry);
  if (foundry.length) comparisonControls("foundry", foundry, rows.filter(row => isFoundry(row) || isSynapse(row)), run);
  else byId("foundry-selection").textContent = "No measured Foundry Local evidence in this run.";
  const mlx = rows.filter(row => row.subject === "SwiftLM/MLX");
  if (mlx.length) comparisonControls("mlx", mlx, rows.filter(row => row.subject === "SwiftLM/MLX" || isSynapse(row)), run);
  else byId("mlx-selection").textContent = "No measured MLX evidence in this run.";
}
