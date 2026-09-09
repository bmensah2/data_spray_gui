// generate_gui_session_report.js
// ABEN Dual RGB Detection System — GUI Session Report → .docx
//
// Consumes the JSON written by core/gui_session_report.py (a GUI
// ARM DETECTION session: operator-entered metadata, full system
// provenance, and the same spray events shown in the Session
// Analysis tab) and produces a publication-quality Word document.
//
// This is deliberately a SEPARATE generator from generate_report_rgb.js
// rather than a reshape into that script's schema: a GUI session's
// data is genuinely different (per-event class/confidence/GPS/pose,
// the model's own class list, camera auto-mode flags, git commit,
// crop/growth-stage/institution) and forcing it into the mission-
// report's cfg/sum shape would mean losing most of that detail. Both
// generators share styling via docx_report_helpers.js so the two
// report types still look consistent.
//
// Usage:
//   node generate_gui_session_report.js --session report.json \
//     --out ABEN_Session_Report.docx

const fs = require("fs");
const { Document, Packer, Paragraph, TextRun, AlignmentType, PageBreak } = require("docx");
const {
  PAGE_WIDTH, PAGE_HEIGHT, MARGIN,
  COLOR_ACCENT, COLOR_OK, COLOR_WARN,
  h1, h2, p, bullet, divider, statLine, simpleTable,
  fmt, pct, docStyles,
} = require("./docx_report_helpers");

function getArg(name, def = null) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : def;
}
const sessionPath = getArg("session");
const outPath     = getArg("out", "ABEN_Session_Report.docx");

if (!sessionPath) {
  console.error("Usage: node generate_gui_session_report.js --session report.json [--out report.docx]");
  process.exit(1);
}

const report = JSON.parse(fs.readFileSync(sessionPath, "utf8"));
const meta  = report.metadata   || {};
const prov  = report.provenance || {};
const stats = report.statistics || {};
const events = report.events    || [];
const warnings = report.warnings || [];

const sw    = prov.software        || {};
const model = prov.model           || {};
const mcls  = prov.model_classes   || {};
const geo   = prov.geometry        || {};
const cams  = prov.cameras         || {};

const children = [];

// ── Title page ──────────────────────────────────────────────
children.push(
  new Paragraph({ spacing: { before: 1600 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Dual RGB Detection System", bold: true, size: 56, color: COLOR_ACCENT })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 100 },
    children: [new TextRun({ text: "Session Report", size: 30 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 400 },
    children: [new TextRun({
      text: `Field: ${meta.field_id || "n/a"}    |    Crop: ${meta.crop || "n/a"}    |    Stage: ${meta.growth_stage || "n/a"}`,
      size: 22, color: "555555" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 100 },
    children: [new TextRun({
      text: `Operator: ${meta.operator || "(not recorded)"}    |    ${report.session_id || ""}`,
      size: 20, color: "888888" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60 },
    children: [new TextRun({
      text: new Date(report.session_start_iso || report.generated_at || Date.now()).toUTCString(),
      size: 20, color: "888888" })] }),
  new Paragraph({ children: [new PageBreak()] }),
);

// ── Warnings banner (up top -- a caveat nobody sees is not a
//    caveat) ─────────────────────────────────────────────────
if (warnings.length > 0) {
  children.push(h1("⚠ Warnings"));
  children.push(p(
    "The following conditions were detected automatically and should be " +
    "reviewed before citing this report:", {}));
  warnings.forEach(w => children.push(bullet(w)));
  children.push(divider());
}

// ── Executive Summary ───────────────────────────────────────
children.push(h1("Executive Summary"));
const n = stats.total_events || 0;
children.push(p(
  `This report documents an ARM DETECTION session of the Dual RGB Detection ` +
  `System operated by ${meta.operator || "an unrecorded operator"}` +
  `${meta.institution ? ` (${meta.institution})` : ""}. ` +
  (n > 0
    ? `Over ${fmt(stats.duration_s, 1)} s, ${n} confirmed weed detection${n === 1 ? "" : "s"} ` +
      `resulted in ${n} spray event${n === 1 ? "" : "s"} ` +
      `(${fmt(stats.events_per_minute, 2)} events/min).`
    : `No spray events were recorded during this session.`)
));
if (mcls.available) {
  children.push(p(
    `The ${model.detection_mode || ""} model (${model.model_path || "unknown"}) was loaded with ` +
    `${mcls.class_count} class${mcls.class_count === 1 ? "" : "es"}: ${Object.values(mcls.classes || {}).join(", ")}.` +
    `${mcls.stub_mode ? "  NOTE: engine was running in STUB MODE (no real inference)." : ""}`,
    mcls.stub_mode ? { color: COLOR_WARN, bold: true } : {}
  ));
}
children.push(p(
  `Pipeline: eMeet C960 4K (Dual RGB) → ${model.detection_mode === "cls" ? "classification" : "YOLO segmentation"} → ` +
  `zone/nozzle geometry → distance-buffered spray decision → Arduino nozzle control.`
));
children.push(divider());

// ── Session Details ─────────────────────────────────────────
children.push(h1("Session Details"));
children.push(simpleTable(
  ["Field", "Value"],
  [
    ["Operator", meta.operator || "(not recorded)"],
    ["Researcher", meta.researcher || "—"],
    ["Institution", meta.institution || "—"],
    ["Field ID", meta.field_id || "—"],
    ["Location", meta.location || "—"],
    ["Crop", meta.crop || "—"],
    ["Growth stage", meta.growth_stage || "—"],
    ["Session ID", report.session_id || "—"],
    ["Start", report.session_start_iso || "—"],
    ["End", report.session_end_iso || "—"],
    ["Wall duration", report.session_wall_duration_s != null ? `${fmt(report.session_wall_duration_s, 1)} s` : "—"],
  ],
  [2500, 6860],
));
if (meta.notes) {
  children.push(p("Notes:", { bold: true }));
  children.push(p(meta.notes));
}
children.push(divider());

// ── System Configuration ────────────────────────────────────
children.push(h1("System Configuration"));
children.push(simpleTable(
  ["Component", "Setting", "Value"],
  [
    ["Model", "Path", model.model_path || "—"],
    ["Model", "Mode", model.detection_mode || "—"],
    ["Model", "Confidence threshold", fmt(model.confidence_threshold, 2)],
    ["Model", "IoU threshold", fmt(model.iou_threshold, 2)],
    ["Model", "Image size", model.imgsz != null ? `${model.imgsz}\u00d7${model.imgsz}` : "—"],
    ["Model", "Device", model.device || "—"],
    ["Model", "File exists on disk", model.model_exists === true ? "yes" : (model.model_exists === false ? "NO — check path" : "—")],
    ["Model", "Modified", model.model_modified || "—"],
    ["Model", "Classes", mcls.available ? `${mcls.class_count}: ${Object.values(mcls.classes || {}).join(", ")}` : "unavailable"],
    ["Model", "Stub mode (no real inference)", mcls.stub_mode ? "YES — results are placeholders" : "no"],
    ["Geometry", "B1 / B2 split", `${geo.b1_split_x ?? "—"} px  /  ${geo.b2_split_x ?? "—"} px`],
    ["Geometry", "Nozzle centers (N1/N2/N2/N3)", `${geo.n1_center_cam1 ?? "—"} / ${geo.n2_center_cam1 ?? "—"} / ${geo.n2_center_cam2 ?? "—"} / ${geo.n3_center_cam2 ?? "—"} px`],
    ["Geometry", "Nozzle Y line", geo.nozzle_y_px != null ? `${geo.nozzle_y_px} px` : "—"],
    ["Geometry", "Camera height", geo.camera_height_m != null ? `${fmt(geo.camera_height_m, 3)} m` : "—"],
    ["Geometry", "GSD", geo.gsd_m_per_px != null ? `${fmt(geo.gsd_m_per_px * 1000, 3)} mm/px` : "—"],
    ["Geometry", "Spray window", `${fmt(geo.min_spray_dist_m, 3)} – ${fmt(geo.max_spray_dist_m, 3)} m`],
    ["Zones", "Debounce threshold", geo.detection_threshold != null ? `${geo.detection_threshold} consecutive frames` : "—"],
    ["Safety", "Never-spray classes", (geo.non_spray_classes || []).join(", ") || "(none configured)"],
  ],
  [2000, 3200, 4160],
));
children.push(divider());

// ── Camera Settings ──────────────────────────────────────────
children.push(h1("Camera Settings"));
if (Object.keys(cams).length === 0) {
  children.push(p("No camera settings were captured for this session.", { color: COLOR_WARN }));
} else {
  const camRows = [];
  for (const [side, c] of Object.entries(cams)) {
    if (!c.available) {
      camRows.push([side.toUpperCase(), "unavailable", c.reason || "unknown"]);
      continue;
    }
    camRows.push([side.toUpperCase(), "Exposure", `${c.exposure ?? "—"}  (auto: ${c.auto_exposure === 3 ? "on" : "off"})`]);
    camRows.push([side.toUpperCase(), "White balance", `${c.wb_temp ?? "—"} K  (auto: ${c.auto_wb === 1 ? "on" : "off"})`]);
    camRows.push([side.toUpperCase(), "Focus", `${c.focus ?? "—"}  (auto: ${c.autofocus === 1 ? "on" : "off"})`]);
    camRows.push([side.toUpperCase(), "Gamma / Gain / Sharpness", `${c.gamma ?? "—"} / ${c.gain ?? "—"} / ${c.sharpness ?? "—"}`]);
    camRows.push([side.toUpperCase(), "Brightness / Contrast / Saturation", `${c.brightness ?? "—"} / ${c.contrast ?? "—"} / ${c.saturation ?? "—"}`]);
  }
  children.push(simpleTable(["Camera", "Setting", "Value"], camRows, [1600, 3200, 4560]));
}
children.push(divider());

// ── Spray Event Log ─────────────────────────────────────────
children.push(h1("Spray Event Log"));
if (events.length === 0) {
  children.push(p("No spray events were recorded this session.", { color: COLOR_WARN }));
} else {
  const rows = events.map(e => [
    e.time || "—",
    e.zone_name || "—",
    e.nozzle || "—",
    (e.classes || []).join(", ") || "—",
    fmt(e.confidence, 2),
    (e.pose_x != null && e.pose_y != null) ? `${fmt(e.pose_x, 2)}, ${fmt(e.pose_y, 2)}` : "—",
    e.gps_valid ? `${fmt(e.gps_lat, 5)}, ${fmt(e.gps_lon, 5)}` : "—",
    e.flagged_cls ? "yes" : "",
  ]);
  children.push(simpleTable(
    ["Time", "Zone", "Nozzle", "Class", "Conf", "Pose (x,y)", "GPS", "Flag"],
    rows,
    [1050, 1200, 850, 1750, 700, 1350, 1450, 970],
  ));
}
children.push(divider());

// ── Statistics ───────────────────────────────────────────────
children.push(h1("Statistics"));
if (n > 0) {
  children.push(h2("Events by Zone"));
  children.push(simpleTable(
    ["Zone", "Count"],
    Object.entries(stats.events_by_zone || {}),
    [4680, 4680],
  ));
  children.push(h2("Events by Nozzle"));
  children.push(simpleTable(
    ["Nozzle", "Count"],
    Object.entries(stats.events_by_nozzle || {}),
    [4680, 4680],
  ));
  children.push(h2("Detections by Class"));
  children.push(simpleTable(
    ["Class", "Count"],
    Object.entries(stats.detections_by_class || {}),
    [4680, 4680],
  ));
  const cs = stats.confidence_stats || {};
  children.push(h2("Confidence"));
  children.push(statLine("Mean", fmt(cs.mean, 3)));
  children.push(statLine("Median", fmt(cs.median, 3)));
  children.push(statLine("Range", `${fmt(cs.min, 3)} – ${fmt(cs.max, 3)}`));
  if (cs.stdev != null) children.push(statLine("Std. dev.", fmt(cs.stdev, 3)));

  const gc = stats.gps_coverage || {};
  children.push(h2("GPS Coverage"));
  children.push(statLine("Events with a valid fix",
    `${gc.events_with_fix ?? 0} / ${n}  (${pct(gc.coverage_pct)})`));

  if (stats.cls_flagged) {
    children.push(h2("CLS-Flagged Events"));
    children.push(statLine("Flagged for review", stats.cls_flagged));
  }
}
children.push(divider());

// ── Software / Reproducibility ──────────────────────────────
children.push(h1("Software & Reproducibility"));
children.push(simpleTable(
  ["Component", "Version"],
  [
    ["Git commit", `${sw.git_commit || "unavailable"}${sw.git_dirty === true ? "  (UNCOMMITTED CHANGES)" : ""}`],
    ["Python", sw.python || "—"],
    ["ultralytics", sw.ultralytics || "—"],
    ["torch", sw.torch || "—"],
    ["CUDA available", String(sw.cuda_available ?? "—")],
    ["GPU", sw.gpu || "—"],
    ["OpenCV", sw.opencv || "—"],
    ["NumPy", sw.numpy || "—"],
    ["Qt", sw.qt || "—"],
    ["Platform", sw.platform || "—"],
  ],
  [3000, 6360],
));
if (sw.git_dirty === true) {
  children.push(p(
    "The working tree had uncommitted changes when this session ran — " +
    "the exact code cannot be fully reconstructed from the git commit alone.",
    { color: COLOR_WARN }
  ));
}
children.push(divider());

// ── Conclusions ──────────────────────────────────────────────
children.push(h1("Conclusions & Next Steps"));
children.push(p(
  n > 0
    ? `This session confirms the complete RGB detection-to-spray pipeline operated correctly end to end: ` +
      `camera capture, ${mcls.stub_mode ? "engine construction (stub mode — not real inference)" : "model inference"}, ` +
      `zone assignment, debounce confirmation, and nozzle actuation across ` +
      `${Object.keys(stats.events_by_zone || {}).length} zone(s).`
    : `This session ran without recording any spray events. Review the Warnings section above ` +
      `and confirm the model, cameras, and zone thresholds are configured as expected before the next run.`
));
children.push(p("Recommended follow-up:"));
if (mcls.stub_mode) {
  children.push(bullet("Engine was in stub mode this session — verify ultralytics is installed and the model path is correct before relying on these results."));
}
if (model.model_exists === false) {
  children.push(bullet(`Model file was not found at ${model.model_path || "the configured path"} — confirm the path before the next session.`));
}
if (!meta.operator) {
  children.push(bullet("No operator name was recorded — confirm the session metadata prompt is being filled in."));
}
children.push(bullet("Cross-reference this report's Spray Event Log against the Session Analysis tab for the same session ID to confirm they agree."));
children.push(bullet("Archive this report alongside the session JSON for full reproducibility (git commit + software versions are recorded above)."));

// ── Build & write ────────────────────────────────────────────
const doc = new Document({
  ...docStyles,
  sections: [{
    properties: {
      page: {
        size: { width: PAGE_WIDTH, height: PAGE_HEIGHT },
        margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN },
      },
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(outPath, buffer);
  console.log(`Report written to: ${outPath}`);
}).catch(err => {
  console.error("Failed to generate report:", err);
  process.exit(1);
});
