// REQ-WEB-002: render published reports without combining benchmark cohorts.
const repository = "https://github.com/managedcode/Synapse";
const labels = { passed: "Passed", failed: "Failed", success: "Succeeded", failure: "Failed",
  cancelled: "Cancelled", skipped: "Skipped", not_run_missing_hardware: "Not run: hardware unavailable" };
const numeric = ["samples", "output_tokens", "ttft_milliseconds", "decode_tokens_per_second", "wall_milliseconds", "peak_rss_mib"];
const textFields = ["cohort", "runner", "scenario", "subject", "source_artifact", "output_state"];
const byId = id => document.getElementById(id);
const format = value => value === null ? "—" : new Intl.NumberFormat("en", { maximumFractionDigits: 2 }).format(value);
const statusLabel = value => labels[value] ?? value.replaceAll("_", " ");
const validNumber = value => typeof value === "number" && Number.isFinite(value) && value >= 0;
const validDate = value => typeof value === "string" && /^\d{4}-\d{2}-\d{2}T/.test(value) && Number.isFinite(Date.parse(value));
const validText = value => typeof value === "string" && value.length > 0 && value.length <= 2048;
const requireValue = condition => { if (!condition) throw new Error("The published results do not match the supported report format."); };

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function validRun(run) {
  if (run === null) return;
  requireValue(run && Number.isSafeInteger(run.id) && run.id > 0 && Number.isSafeInteger(run.run_number) && run.run_number > 0);
  requireValue(Number.isSafeInteger(run.run_attempt) && run.run_attempt > 0 && /^[a-f0-9]{40}$/.test(run.head_sha));
  requireValue(validDate(run.updated_at) && validText(run.conclusion));
  requireValue(run.html_url === `${repository}/actions/runs/${run.id}`);
}

function validDiagnostics(items) {
  requireValue(Array.isArray(items) && items.length <= 10000);
  requireValue(items.every(validText));
}

function validTest(report) {
  requireValue(report && report.schema_version === 1 && ["runtime_identifier", "scope"].every(key => validText(report[key])));
  requireValue(["passed", "failed", "not_run", "invalid"].includes(report.status));
  requireValue(["total", "passed", "failed", "skipped"].every(key => Number.isSafeInteger(report[key]) && report[key] >= 0));
  requireValue(report.passed + report.failed + report.skipped === report.total);
  requireValue(report.duration_seconds === null || validNumber(report.duration_seconds));
  requireValue(report.status !== "passed" || (report.total > 0 && report.failed === 0));
  validDiagnostics(report.issues);
}

function validRow(row) {
  requireValue(row && textFields.every(key => validText(row[key])));
  requireValue(typeof row.turn === "string" || (Number.isSafeInteger(row.turn) && row.turn >= 0) || row.turn === null);
  requireValue(numeric.every(key => row[key] === null || validNumber(row[key])));
  requireValue(["native_eval", "reported_decode"].includes(row.decode_metric_scope));
  requireValue(["fresh_process", "request"].includes(row.wall_metric_scope));
}

function validate(data) {
  requireValue(data && data.schema_version === 1 && validDate(data.generated_at_utc));
  requireValue(data.verification && data.performance);
  validRun(data.verification.run);
  validRun(data.performance.run);
  requireValue(Array.isArray(data.verification.reports) && data.verification.reports.length <= 1000);
  validDiagnostics(data.verification.missing_artifacts);
  data.verification.reports.forEach(validTest);
  const report = data.performance.report;
  if (report !== null) {
    requireValue(report && report.schema_version === 1 && validDate(report.generated_at_utc));
    requireValue(Number.isSafeInteger(report.complete_artifacts) && report.complete_artifacts >= 0);
    requireValue(Number.isSafeInteger(report.expected_artifacts) && report.expected_artifacts >= report.complete_artifacts);
    validDiagnostics(report.missing_artifacts);
    validDiagnostics(report.invalid_artifacts);
    requireValue(Array.isArray(report.rows) && report.rows.length <= 10000);
    report.rows.forEach(validRow);
  }
  return data;
}

function renderRun(id, run) {
  const container = byId(id);
  container.replaceChildren();
  if (!run) { container.textContent = "No published run is available."; return; }
  const link = element("a", `Run #${run.run_number} · attempt ${run.run_attempt}`);
  link.href = run.html_url;
  const commit = element("a", run.head_sha.slice(0, 8), "metric-commit");
  commit.href = `${repository}/commit/${run.head_sha}`;
  const date = element("time", new Date(run.updated_at).toISOString().replace("T", " ").replace(/\.\d{3}Z$/, " UTC"));
  date.dateTime = run.updated_at;
  container.append(link, element("span", statusLabel(run.conclusion), "metric-run-status"), commit, date);
}

function diagnosticText(item) {
  return item;
}

function diagnostics(container, prefix, items) {
  if (!items.length) return;
  const details = element("details", undefined, "metric-diagnostics");
  details.append(element("summary", `${prefix} (${items.length})`));
  const list = element("ul");
  items.forEach(item => list.append(element("li", diagnosticText(item))));
  details.append(list);
  container.append(details);
}

function renderTests(stream) {
  renderRun("verification-run", stream.run);
  const container = byId("verification-results");
  container.replaceChildren();
  for (const report of stream.reports) {
    const card = element("article", undefined, "verification-card");
    card.append(element("h4", report.runtime_identifier), element("p", report.scope, "metric-scope"));
    card.append(element("p", statusLabel(report.status), "metric-test-status"));
    const counts = element("dl", undefined, "metric-counts");
    for (const key of ["total", "passed", "failed", "skipped"]) {
      const item = element("div");
      item.append(element("dt", key), element("dd", format(report[key])));
      counts.append(item);
    }
    card.append(counts, element("p", report.duration_seconds === null ? "Duration unavailable" : `${format(report.duration_seconds)} seconds`, "metric-note"));
    diagnostics(card, "Reported issues", report.issues);
    container.append(card);
  }
  if (!stream.reports.length) container.append(element("p", "No test report is available for this publication.", "metric-note"));
  const notes = byId("verification-diagnostics");
  notes.replaceChildren();
  diagnostics(notes, "Missing test artifacts", stream.missing_artifacts);
}

function populateFilter(id, rows, field) {
  const select = byId(id);
  select.replaceChildren();
  const all = element("option", `All ${field === "cohort" ? "cohorts" : `${field}s`}`);
  all.value = "";
  select.append(all);
  [...new Set(rows.map(row => row[field]))].sort().forEach(value => {
    const option = element("option", value);
    option.value = value;
    select.append(option);
  });
}

function cell(row, value, note) {
  const container = element("td", value);
  if (note) container.append(element("small", note));
  row.append(container);
}

function renderRow(item) {
  const row = element("tr");
  const subject = element("th", item.subject);
  subject.scope = "row";
  subject.append(element("small", item.source_artifact));
  row.append(subject);
  cell(row, item.cohort);
  cell(row, item.runner);
  cell(row, item.scenario, item.turn === null ? undefined : `Turn ${item.turn}`);
  cell(row, format(item.samples));
  cell(row, format(item.output_tokens), statusLabel(item.output_state));
  cell(row, format(item.ttft_milliseconds));
  cell(row, format(item.decode_tokens_per_second), item.decode_metric_scope === "native_eval" ? "Native evaluation" : "Reported decode");
  cell(row, format(item.wall_milliseconds), item.wall_metric_scope === "fresh_process" ? "Fresh process" : "Request");
  cell(row, format(item.peak_rss_mib));
  return row;
}

function filterRows(rows) {
  const filters = ["cohort", "runner", "scenario"].map(field => [field, byId(`metric-${field}`).value]);
  const selected = rows.filter(row => filters.every(([field, value]) => !value || row[field] === value));
  byId("metric-rows").replaceChildren(...selected.map(renderRow));
  byId("metric-selection").textContent = selected.length ? `${selected.length} of ${rows.length} measurements shown. No cross-cohort ranking is applied.`
    : "No measurements match these filters.";
}

function renderPerformance(stream) {
  renderRun("performance-run", stream.run);
  const notes = byId("performance-diagnostics");
  notes.replaceChildren();
  if (!stream.report) {
    notes.textContent = "No benchmark report is available for this publication. The recorded local measurements below remain separately identified.";
    return;
  }
  const report = stream.report;
  notes.append(element("span", `${report.complete_artifacts} of ${report.expected_artifacts} expected artifacts available. `));
  if (report.missing_artifacts.length || report.invalid_artifacts.length || report.complete_artifacts < report.expected_artifacts) {
    notes.append(element("span", "Results are incomplete; retained rows are diagnostics."));
  }
  diagnostics(notes, "Missing benchmark artifacts", report.missing_artifacts);
  diagnostics(notes, "Invalid benchmark artifacts", report.invalid_artifacts);
  if (!report.rows.length) { notes.append(element("p", "This report contains no measured rows.")); return; }
  for (const field of ["cohort", "runner", "scenario"]) populateFilter(`metric-${field}`, report.rows, field);
  byId("metric-filters").hidden = false;
  byId("metrics-table-wrap").hidden = false;
  byId("metric-filters").addEventListener("change", () => filterRows(report.rows));
  byId("metric-filters").addEventListener("submit", event => event.preventDefault());
  filterRows(report.rows);
}

async function loadMetrics() {
  const status = byId("metrics-status");
  status.textContent = "Loading published GitHub results…";
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);
  try {
    const response = await fetch("data/latest.json", { cache: "no-store", signal: controller.signal });
    if (response.status === 404) {
      status.textContent = "No results have been published yet. Follow the GitHub runs above for the current status.";
      return;
    }
    if (!response.ok) throw new Error("The published results could not be loaded.");
    const data = validate(await response.json());
    renderTests(data.verification);
    renderPerformance(data.performance);
    status.textContent = `Results generated ${new Date(data.generated_at_utc).toISOString().replace("T", " ").replace(/\.\d{3}Z$/, " UTC")}. Test and performance runs keep their own source revisions.`;
  } catch (error) {
    status.textContent = error.name === "AbortError" ? "Loading timed out. Open the JSON or GitHub runs above to inspect the results."
      : `${error.message} Open the JSON or GitHub runs above to inspect the results.`;
  } finally {
    clearTimeout(timeout);
  }
}

if (byId("metrics")) loadMetrics();
