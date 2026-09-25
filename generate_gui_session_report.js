// generate_gui_session_report.js
// ABEN RGB Detection Systems (Dual or Triple camera) — GUI Session Report → .docx
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

// core/gui_session_report.py writes session_start_iso/session_end_iso
// via datetime.fromtimestamp() with no explicit timezone -- i.e. the
// machine's OWN local clock, naive (no "Z" or +HH:MM suffix). On the
// Jetson that clock is already set to US Central time (confirmed: a
// real report's "Start" field and its title-page date, the latter run
// through new Date(...).toUTCString(), differed by exactly 5 hours --
// the CDT offset -- meaning the naive string IS already Central time,
// just mislabeled "GMT" by force-converting through toUTCString()).
// Parsing a naive string with `new Date(...)` is itself timezone-
// dependent on whatever machine runs THIS SCRIPT (this sandbox's own
// node reports UTC, not Central -- would silently mis-render if this
// ran here rather than on the Jetson), so this formats the ISO
// string's own Y-M-D/H:M:S components directly instead of round-
// tripping through a Date object at all -- portable regardless of
// which machine the script runs on, always showing exactly what the
// timestamp says, labeled as Central since that's what it is.
// A timestamp that DOES carry an explicit UTC marker (e.g.
// spray_mission_rgb.py's datetime.utcnow().isoformat() + "Z") is
// unambiguous, so that path still safely converts via Intl.
function formatChicago(isoString) {
  if (!isoString) return "—";
  const naive = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$/;
  if (naive.test(isoString)) {
    const [datePart, timePart] = isoString.split("T");
    const [y, mo, d] = datePart.split("-").map(Number);
    const [h, mi, s] = timePart.split(":").map(Number);
    const months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
    const weekday = new Date(Date.UTC(y, mo - 1, d)).toLocaleDateString("en-US", { weekday: "short", timeZone: "UTC" });
    const hh = String(h).padStart(2, "0");
    const mm = String(mi).padStart(2, "0");
    const ss = String(s).padStart(2, "0");
    return `${weekday}, ${String(d).padStart(2, "0")} ${months[mo - 1]} ${y} ${hh}:${mm}:${ss} Central`;
  }
  // Explicit-offset/UTC string -- safe to convert via Intl.
  try {
    const fmtd = new Intl.DateTimeFormat("en-US", {
      timeZone: "America/Chicago", weekday: "short", year: "numeric",
      month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit",
      second: "2-digit", hour12: false, timeZoneName: "short",
    }).format(new Date(isoString));
    return fmtd;
  } catch {
    return isoString;
  }
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

// Camera-count-agnostic title/pipeline text -- geo.system === "triple"
// is the primary signal (set explicitly by core/session_provenance.py's
// capture_triple_geometry()); cams.cam1 is a second, independent check
// for robustness against an older report whose geometry section came
// from a failed capture (geo would be {} / no "system" key) but whose
// cameras section still shows the triple-camera shape (cam1/cam2/cam3
// keys, vs the 2-camera system's left/right).
const isTriple = geo.system === "triple" || cams.cam1 !== undefined;
const systemName = isTriple ? "Triple RGB Detection System" : "Dual RGB Detection System";
const systemNameShort = isTriple ? "Triple RGB Detection" : "Dual RGB Detection";
const pipelineCameraLabel = isTriple ? "eMeet C960 4K (Triple RGB)" : "eMeet C960 4K (Dual RGB)";

const children = [];

// ── Title page ──────────────────────────────────────────────
// Field ID as the main title (operator request) rather than the
// system name -- meta.field_id is what identifies THIS report among
// many from the same system, so it leads; systemName moves to the
// subtitle line, combined with what was a separate "Session Report"
// line. The Field: ... detail line below drops its own "Field:"
// segment since that's now redundant with the title itself.
children.push(
  new Paragraph({ spacing: { before: 1600 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: meta.field_id || "(no field ID recorded)", bold: true, size: 56, color: COLOR_ACCENT })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 100 },
    children: [new TextRun({ text: `${systemName} — Session Report`, size: 30 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 400 },
    children: [new TextRun({
      text: `Crop: ${meta.crop || "n/a"}    |    Stage: ${meta.growth_stage || "n/a"}`,
      size: 22, color: "555555" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 100 },
    children: [new TextRun({
      text: `Operator: ${meta.operator || "(not recorded)"}    |    ${report.session_id || ""}`,
      size: 20, color: "888888" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60 },
    children: [new TextRun({
      text: formatChicago(report.session_start_iso || report.generated_at || new Date().toISOString()),
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
  `This report documents an ARM DETECTION session of the ${systemNameShort} ` +
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
  `Pipeline: ${pipelineCameraLabel} → ${model.detection_mode === "cls" ? "classification" : "YOLO segmentation"} → ` +
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
    ["Start", formatChicago(report.session_start_iso)],
    ["End", formatChicago(report.session_end_iso)],
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
// Zone/nozzle geometry rows differ by system: geo.system === "triple"
// (set by core/session_provenance.py's capture_triple_geometry())
// means a 3-camera, 1:1 camera-to-nozzle layout with no B1/B2 split
// at all -- rendering the 2-camera system's own field names for a
// triple-camera report produced meaningless values on a real
// generated report (600/1700/400/1400px, "N1/N2/N2/N3" with N2
// listed twice), the wrong config object's defaults, not just the
// wrong words. Absent geo.system (older reports, or a genuine
// capture_geometry() failure) falls back to the original 2-camera
// rendering unchanged.
const zoneRows = geo.system === "triple"
  ? [
      ["Geometry", "Cam1/Cam2/Cam3 zone boundaries",
        `${geo.cam1_max_x ?? "—"} / ${geo.cam2_min_x ?? "—"}\u2013${geo.cam2_max_x ?? "—"} / ${geo.cam3_min_x ?? "—"} px`],
      ["Geometry", "Nozzle centers (N1/N2/N3, measured)",
        `${geo.cam1_nozzle_x ?? "—"} / ${geo.cam2_nozzle_x ?? "—"} / ${geo.cam3_nozzle_x ?? "—"} px`],
    ]
  : [
      ["Geometry", "B1 / B2 split", `${geo.b1_split_x ?? "—"} px  /  ${geo.b2_split_x ?? "—"} px`],
      ["Geometry", "Nozzle centers (N1/N2/N2/N3)", `${geo.n1_center_cam1 ?? "—"} / ${geo.n2_center_cam1 ?? "—"} / ${geo.n2_center_cam2 ?? "—"} / ${geo.n3_center_cam2 ?? "—"} px`],
    ];

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
    ...zoneRows,
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
// ── Mission-specific extras (robot kinematics, camera grab
//    performance, inference timing distribution) -- present only
//    when this report came from START MISSION, since ARM DETECTION
//    sessions don't have a "travel toward a target distance" concept
//    or per-frame camera-grab success tracking in the same way. ──
const mx = stats.mission_extras;
if (mx) {
  children.push(h1("Mission Kinematics & Performance"));

  const r = mx.robot || {};
  children.push(h2("Robot"));
  children.push(simpleTable(
    ["Metric", "Value"],
    [
      ["Target distance", `${fmt(r.target_distance_m, 2)} m`],
      ["Distance traveled", `${fmt(r.distance_traveled_m, 2)} m`],
      ["Duration", `${fmt(r.duration_s, 1)} s`],
      ["Commanded speed", `${fmt(r.commanded_speed_mps, 3)} m/s`],
      ["Actual avg speed", `${fmt(r.actual_avg_speed_mps, 3)} m/s`],
      ["Speed accuracy", pct(r.speed_accuracy_pct)],
      ["Ended because", r.abort_reason || "—"],
    ],
    [4680, 4680],
  ));

  const c = mx.camera || {};
  if (Object.keys(c).length) {
    children.push(h2("Camera Grab Performance"));
    children.push(simpleTable(
      ["Metric", "Value"],
      [
        ["Frame grabs attempted", c.grabs_attempted ?? "—"],
        ["Frame grabs succeeded", c.grabs_succeeded ?? "—"],
        ["Success rate", pct(c.success_rate_pct)],
        ["Avg dual-cam sync error", c.avg_sync_error_ms != null ? `${fmt(c.avg_sync_error_ms, 1)} ms` : "—"],
        ["Max dual-cam sync error", c.max_sync_error_ms != null ? `${fmt(c.max_sync_error_ms, 1)} ms` : "—"],
        ["Sync errors >50ms", c.sync_errors_over_50ms ?? "—"],
      ],
      [4680, 4680],
    ));
  }

  const d = mx.detection || {};
  if (d.frames_processed > 0) {
    children.push(h2("Inference Timing"));
    const inf = d.inference_ms || {};
    children.push(simpleTable(
      ["Metric", "Value"],
      [
        ["Frames processed", d.frames_processed],
        ["Achieved FPS", d.achieved_fps != null ? fmt(d.achieved_fps, 2) : "—"],
        ["Inference time (mean/min/max)", inf.mean != null ? `${fmt(inf.mean,1)} / ${fmt(inf.min,1)} / ${fmt(inf.max,1)} ms` : "—"],
        ["Inference p95", inf.p95 != null ? `${fmt(inf.p95, 1)} ms` : "—"],
      ],
      [4680, 4680],
    ));
  }

  const sp = mx.spray || {};
  if (sp.near_misses) {
    children.push(h2("Near Misses"));
    children.push(statLine("Detections that never confirmed",
      `${sp.near_misses} (debounce filter correctly suppressed noise)`));
  }
  children.push(divider());
}

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
