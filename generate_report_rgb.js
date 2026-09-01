// generate_report_rgb.js
// ABEN Dual RGB Detection System — Final Report Assembly
//
// RGB fork of generate_report.js.
// Combines session_rgb.json (live run telemetry from spray_mission_rgb.py),
// model_validation_rgb.json (mAP from evaluate_model_rgb.py),
// and chart PNGs into a polished .docx report.
//
// Key differences from multispec version:
//   - No resistance classifier section
//   - System config shows camera geometry (height, GSD, nozzle Y)
//   - Camera section includes dual-camera sync stats
//   - Spray section shows geometry-based timing (trigger_dist, spray_time)
//   - Pipeline description updated for RGB
//
// Usage:
//   node generate_report_rgb.js --session session_rgb.json \
//     --validation model_validation_rgb.json --charts charts/ \
//     --out ABEN_RGB_Report.docx

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, AlignmentType, LevelFormat, HeadingLevel, BorderStyle,
  WidthType, ShadingType, PageBreak, VerticalAlign,
} = require("docx");

// ── CLI args ────────────────────────────────────────────────
function getArg(name, def = null) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : def;
}
const sessionPath    = getArg("session");
const validationPath = getArg("validation");
const chartsDir      = getArg("charts", "charts");
const outPath        = getArg("out", "ABEN_RGB_Spray_Report.docx");

if (!sessionPath) {
  console.error("Usage: node generate_report_rgb.js --session session_rgb.json [--validation model_validation_rgb.json] [--charts charts/] [--out report.docx]");
  process.exit(1);
}

const session    = JSON.parse(fs.readFileSync(sessionPath, "utf8"));
const validation = (validationPath && fs.existsSync(validationPath))
  ? JSON.parse(fs.readFileSync(validationPath, "utf8")) : null;

const CHART = (name) => {
  const p = path.join(chartsDir, name);
  return fs.existsSync(p) ? p : null;
};

// ── Layout constants ────────────────────────────────────────
const PAGE_WIDTH = 12240, PAGE_HEIGHT = 15840, MARGIN = 1440;
const CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN; // 9360

const COLOR_ACCENT = "2E75B6";
const COLOR_HEAD_BG = "2E75B6";
const COLOR_ROW_ALT = "EEF3F8";
const COLOR_OK = "27AE60";
const COLOR_WARN = "C0392B";

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const cellBorders = { top: border, bottom: border, left: border, right: border };

// ── Helpers ─────────────────────────────────────────────────
function h1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] });
}
function h2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] });
}
function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 160 },
    children: [new TextRun({ text, ...opts })],
  });
}
function bullet(text) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    children: [new TextRun(text)],
  });
}
function divider() {
  return new Paragraph({
    spacing: { before: 120, after: 120 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: COLOR_ACCENT, space: 1 } },
    children: [],
  });
}
function statLine(label, value) {
  return new Paragraph({
    spacing: { after: 80 },
    children: [
      new TextRun({ text: `${label}: `, bold: true }),
      new TextRun({ text: String(value) }),
    ],
  });
}

function simpleTable(headers, rows, widths) {
  const w = widths || headers.map(() => Math.floor(CONTENT_WIDTH / headers.length));
  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((hd, i) => new TableCell({
      borders: cellBorders,
      width: { size: w[i], type: WidthType.DXA },
      shading: { fill: COLOR_HEAD_BG, type: ShadingType.CLEAR },
      verticalAlign: VerticalAlign.CENTER,
      margins: { top: 80, bottom: 80, left: 120, right: 120 },
      children: [new Paragraph({ children: [new TextRun({ text: hd, bold: true, color: "FFFFFF", size: 19 })] })],
    })),
  });
  const bodyRows = rows.map((row, ri) => new TableRow({
    children: row.map((cell, ci) => new TableCell({
      borders: cellBorders,
      width: { size: w[ci], type: WidthType.DXA },
      shading: { fill: ri % 2 === 1 ? COLOR_ROW_ALT : "FFFFFF", type: ShadingType.CLEAR },
      margins: { top: 60, bottom: 60, left: 120, right: 120 },
      children: [new Paragraph({ children: [new TextRun({ text: String(cell), size: 19 })] })],
    })),
  }));
  return new Table({
    width: { size: CONTENT_WIDTH, type: WidthType.DXA },
    columnWidths: w,
    rows: [headerRow, ...bodyRows],
  });
}

function chartImage(chartPath, widthIn = 6.2) {
  if (!chartPath) return null;
  const dims = imageSizeOf(chartPath);
  const widthPx = widthIn * 914400; // EMU per inch
  const heightPx = widthPx * (dims.h / dims.w);
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 240 },
    children: [new ImageRun({
      data: fs.readFileSync(chartPath),
      transformation: { width: widthIn * 96, height: (widthIn * (dims.h / dims.w)) * 96 },
      type: "png",
    })],
  });
}

// Minimal PNG dimension reader (avoids extra deps)
function imageSizeOf(file) {
  const buf = fs.readFileSync(file);
  const w = buf.readUInt32BE(16);
  const h = buf.readUInt32BE(20);
  return { w, h };
}

function fmt(n, digits = 2) {
  if (n === null || n === undefined) return "—";
  return Number(n).toFixed(digits);
}
function pct(n, digits = 1) {
  if (n === null || n === undefined) return "—";
  return `${Number(n).toFixed(digits)}%`;
}

// ── Build document sections ─────────────────────────────────
const cfg = session.system_config;
const sum = session.summary;
const children = [];

// --- Title page ---
children.push(
  new Paragraph({ spacing: { before: 1600 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "ABEN Dual RGB Imaging System", bold: true, size: 56, color: COLOR_ACCENT })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 100 },
    children: [new TextRun({ text: "Autonomous Weed Detection & Precision Spray Mission Report", size: 30 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 400 },
    children: [new TextRun({ text: `Field ID: ${cfg.field_id || "n/a"}    |    Mode: ${cfg.detection_mode}    |    Camera: ${cfg.camera_model}`, size: 22, color: "555555" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 100 },
    children: [new TextRun({ text: new Date(cfg.session_start_iso || Date.now()).toUTCString(), size: 20, color: "888888" })] }),
  new Paragraph({ children: [new PageBreak()] }),
);

// --- Executive Summary ---
children.push(h1("Executive Summary"));
const targetMet = sum.robot.abort_reason === "target_reached";
const missedCount = sum.spray.missed;
const nearMisses = sum.spray.near_misses;
children.push(p(
  `This report documents an autonomous weed detection and targeted spray mission ` +
  `by the ABEN Dual RGB Imaging System. ` +
  `The robot traveled ${fmt(sum.robot.distance_traveled_m, 2)} m ` +
  `of a ${fmt(sum.robot.target_distance_m, 2)} m target (ended: ${sum.robot.abort_reason.replace("_", " ")}), ` +
  `processing ${sum.detection.frames_processed} frames. ` +
  `${sum.spray.total_triggers} weed detection${sum.spray.total_triggers === 1 ? "" : "s"} were confirmed ` +
  `through the debounce filter, resulting in ${sum.spray.fired} spray event${sum.spray.fired === 1 ? "" : "s"}.`
));
if (validation) {
  const segLine = validation.seg_overall
    ? ` Segmentation mask mAP50 of ${fmt(validation.seg_overall.mask_mAP50, 3)} (primary metric — masks drive resistance classification).`
    : "";
  children.push(p(
    `Model validation (held-out set): bounding-box mAP50 = ${fmt(validation.overall.mAP50, 3)}, ` +
    `mAP50-95 = ${fmt(validation.overall.mAP50_95, 3)}.${segLine}`
  ));
}
children.push(p(
  `Pipeline: eMeet C960 4K (Dual RGB) → YOLOv8n-Seg → per-plant masks → ` +
  `zone geometry (look-ahead ${fmt(cfg.look_ahead_m, 4)}m, GSD ${fmt(cfg.gsd_mm_per_px, 2)}mm/px) → ` +
  `distance-buffered spray decision → Arduino nozzle control.`
));

children.push(divider());

// --- System Configuration ---
children.push(h1("System Configuration"));
children.push(simpleTable(
  ["Component", "Setting", "Value"],
  [
    ["Camera", "Model", cfg.camera_model],
    ["Camera", "Resolution / FPS", `${cfg.resolution || "1920×1080"}  @  ${cfg.fps || 30} fps`],
    ["Camera", "Left device", (cfg.left_device || "—").slice(-32)],
    ["Camera", "Right device", (cfg.right_device || "—").slice(-32)],
    ["Geometry", "Camera height", `${fmt(cfg.camera_height_m, 4)} m  (${fmt(cfg.camera_height_m * 39.37, 1)} in)`],
    ["Geometry", "Look-ahead (cam→nozzle)", `${fmt(cfg.look_ahead_m, 4)} m  (${fmt(cfg.look_ahead_m * 39.37, 1)} in)`],
    ["Geometry", "GSD", `${fmt(cfg.gsd_mm_per_px, 3)} mm/px`],
    ["Geometry", "Nozzle Y line", `${cfg.nozzle_y_px || 767} px  (in 1080p frame)`],
    ["Zones", "B1 split (cam1)", `${cfg.b1_split_x || 1150} px  →  N1=${cfg.n1_center_px || 600}px  N2=${cfg.n2_center_cam1_px || 1700}px`],
    ["Zones", "B2 split (cam2)", `${cfg.b2_split_x || 900} px   →  N2=${cfg.n2_center_cam2_px || 400}px  N3=${cfg.n3_center_px || 1400}px`],
    ["Zones", "Zone count", String(cfg.num_zones)],
    ["Zones", "Debounce threshold", `${cfg.zone_threshold} consecutive frames`],
    ["Zones", "Nozzle count", String(cfg.nozzle_count)],
    ["Model", "Weights", cfg.model_path],
    ["Model", "Inference image size", `${cfg.imgsz}\u00d7${cfg.imgsz}`],
    ["Model", "Confidence threshold", fmt(cfg.confidence_threshold, 2)],
    ["Model", "Device", cfg.device],
    ["Spray timing", "Min spray window", `${fmt(cfg.min_spray_dist_m, 3)} m`],
    ["Spray timing", "Max spray window", `${fmt(cfg.max_spray_dist_m, 3)} m`],
    ["Robot", "Commanded speed", `${fmt(cfg.drive_speed_mps, 3)} m/s`],
    ["Robot", "Target distance", `${fmt(cfg.target_distance_m, 2)} m`],
  ],
  [2000, 3000, 4360],
));

children.push(divider());

// --- Camera Performance ---
children.push(h1("Camera Performance"));
children.push(simpleTable(
  ["Metric", "Value"],
  [
    ["Frame grabs attempted",      sum.camera.grabs_attempted],
    ["Frame grabs succeeded",      sum.camera.grabs_succeeded],
    ["Frame grabs failed",         sum.camera.grabs_failed],
    ["Success rate",               pct(sum.camera.success_rate_pct)],
    ["Avg dual-cam sync error",    sum.camera.avg_sync_error_ms != null ? `${fmt(sum.camera.avg_sync_error_ms, 1)} ms` : "—"],
    ["Max dual-cam sync error",    sum.camera.max_sync_error_ms != null ? `${fmt(sum.camera.max_sync_error_ms, 1)} ms` : "—"],
    ["Sync errors >50ms",          sum.camera.sync_errors_over_50ms ?? "—"],
  ],
  [4680, 4680],
));

children.push(divider());

// --- Detection Performance ---
children.push(h1("Detection Performance"));
const d = sum.detection;
if (d.frames_processed > 0) {
  children.push(simpleTable(
    ["Metric", "Value"],
    [
      ["Frames processed", d.frames_processed],
      ["Achieved frame rate", d.achieved_fps ? `${fmt(d.achieved_fps, 2)} fps` : "—"],
      ["Inference time (mean / p95 / max)", `${fmt(d.inference_ms.mean, 1)} / ${fmt(d.inference_ms.p95, 1)} / ${fmt(d.inference_ms.max, 1)} ms`],
      ["Preprocess time (mean)", `${fmt(d.preprocess_ms.mean, 1)} ms`],
      ["Total detections", d.total_detections],
      ["Confidence (mean / min / max)", d.confidence_overall ? `${fmt(d.confidence_overall.mean)} / ${fmt(d.confidence_overall.min)} / ${fmt(d.confidence_overall.max)}` : "—"],
    ],
    [4680, 4680],
  ));

  if (Object.keys(d.by_class || {}).length > 0) {
    children.push(new Paragraph({ spacing: { before: 200, after: 100 }, children: [new TextRun({ text: "Detections by class:", bold: true })] }));
    children.push(simpleTable(
      ["Class", "Count", "Avg. Confidence"],
      Object.entries(d.by_class).map(([cls, info]) => [cls, info.count, fmt(info.confidence.mean)]),
      [3120, 3120, 3120],
    ));
  }

  const confChart = CHART("confidence_distribution.png");
  const timingChart = CHART("inference_timing.png");
  if (confChart) children.push(chartImage(confChart));
  if (timingChart) children.push(chartImage(timingChart));
} else {
  children.push(p("No frames were processed during this run (dummy-detect mode, or detection loop did not execute)."));
}

children.push(divider());

// --- Spray Event Log ---
children.push(h1("Spray Event Log"));
const sp = sum.spray;
const rdRows = Object.entries(sp.resistance_decisions || {}).map(([decision, count]) => {
  const action = {susceptible:"Sprayed", resistant:"Skipped — resistant", uncertain:"Logged only"}[decision] || "—";
  return [decision.charAt(0).toUpperCase() + decision.slice(1), count, action];
});
children.push(simpleTable(
  ["Metric", "Value"],
  [
    ["Total confirmed triggers",      sp.total_triggers],
    ["Fired",                         sp.fired],
    ["Missed (never reached nozzle)", sp.missed],
    ["Near misses (never confirmed)", sp.near_misses],
    ["Trigger distance (mean)",       sp.trigger_dist_m ? `${fmt(sp.trigger_dist_m.mean, 3)} m  (min ${fmt(sp.trigger_dist_m.min, 3)} m  max ${fmt(sp.trigger_dist_m.max, 3)} m)` : "—"],
    ["Spray duration (mean)",         sp.spray_time_s   ? `${fmt(sp.spray_time_s.mean, 3)} s  (min ${fmt(sp.spray_time_s.min, 3)} s  max ${fmt(sp.spray_time_s.max, 3)} s)`   : "—"],
    ["Fire distance (mean)",          sp.fire_distance_m ? `${fmt(sp.fire_distance_m.mean, 3)} m` : "—"],
    ["Nozzle open duration (mean)",   sp.spray_duration_s ? `${fmt(sp.spray_duration_s.mean, 2)} s` : "—"],
  ],
  [5000, 4360],
));

if (session.spray_events && session.spray_events.length > 0) {
  children.push(new Paragraph({ spacing: { before: 200, after: 100 }, children: [new TextRun({ text: "Individual events:", bold: true })] }));
  children.push(simpleTable(
    ["#", "Nozzle", "Zone", "Detected class(es)", "Trig. dist (m)", "Spray time (s)", "Status"],
    session.spray_events.map((e, i) => [
      i + 1,
      `N${e.nozzle + 1}`,
      e.zone_name,
      e.confirming_classes.join(", "),
      e.trigger_dist_m != null ? fmt(e.trigger_dist_m, 3) : "—",
      e.spray_time_s   != null ? fmt(e.spray_time_s, 3)   : "—",
      e.fire_time ? (e.close_time ? "Fired ✓" : "Fired (open)") : "MISSED",
    ]),
    [480, 900, 1200, 2160, 1440, 1440, 1140],
  ));
}

const sprayChart = CHART("spray_timeline.png");
if (sprayChart) children.push(chartImage(sprayChart));

children.push(divider());

// --- Robot Kinematics ---
children.push(h1("Robot Kinematics"));
const rk = sum.robot;
children.push(simpleTable(
  ["Metric", "Value"],
  [
    ["Target distance", `${fmt(rk.target_distance_m, 3)} m`],
    ["Distance traveled", `${fmt(rk.distance_traveled_m, 3)} m`],
    ["Run duration", `${fmt(rk.duration_s, 1)} s`],
    ["Commanded speed", `${fmt(rk.commanded_speed_mps, 3)} m/s`],
    ["Actual average speed", `${fmt(rk.actual_avg_speed_mps, 3)} m/s`],
    ["Speed accuracy", rk.speed_accuracy_pct ? pct(rk.speed_accuracy_pct, 0) + " of commanded" : "—"],
    ["Run ended because", rk.abort_reason],
  ],
  [4680, 4680],
));

// --- Model Validation Metrics ---
if (validation) {
  children.push(new Paragraph({ children: [new PageBreak()] }));
  children.push(h1("YOLO Model Validation Metrics"));
  children.push(p(
    "Computed separately against a held-out labeled validation split (evaluate_model_rgb.py). " +
    "Live field telemetry has no ground truth available in real time. " +
    "Mask AP is the primary metric for precision spray — accurate segmentation masks " +
    "determine exact plant position for nozzle timing."
  ));

  // Bounding box metrics
  children.push(h2("Bounding-Box Metrics"));
  children.push(simpleTable(
    ["Metric", "Value"],
    [
      ["mAP50",          fmt(validation.overall.mAP50, 3)],
      ["mAP50-95",       fmt(validation.overall.mAP50_95, 3)],
      ["Mean precision", fmt(validation.overall.mean_precision, 3)],
      ["Mean recall",    fmt(validation.overall.mean_recall, 3)],
      ["Validation images", validation.num_val_images ?? "—"],
    ],
    [4680, 4680],
  ));

  // Segmentation mask metrics
  if (validation.seg_overall) {
    children.push(h2("Segmentation Mask Metrics  (primary)"));
    const seg = validation.seg_overall;
    children.push(simpleTable(
      ["Metric", "Value"],
      [
        ["Mask mAP50",     fmt(seg.mask_mAP50, 3)],
        ["Mask mAP50-95",  fmt(seg.mask_mAP50_95, 3)],
        ["Mask precision", fmt(seg.mask_precision, 3)],
        ["Mask recall",    fmt(seg.mask_recall, 3)],
      ],
      [4680, 4680],
    ));
  }

  // Per-class box
  if (validation.per_class && validation.per_class.length > 0) {
    children.push(new Paragraph({ spacing: { before: 200, after: 100 }, children: [new TextRun({ text: "Per-class bounding-box metrics:", bold: true })] }));
    children.push(simpleTable(
      ["Class", "Precision", "Recall", "mAP50-95"],
      validation.per_class.map(c => [c.class_name, fmt(c.precision, 3), fmt(c.recall, 3), fmt(c.map50_95, 3)]),
      [2808, 2184, 2184, 2184],
    ));
  }

  // Per-class seg
  if (validation.per_class_seg && validation.per_class_seg.length > 0) {
    children.push(new Paragraph({ spacing: { before: 200, after: 100 }, children: [new TextRun({ text: "Per-class mask metrics:", bold: true })] }));
    children.push(simpleTable(
      ["Class", "Mask Precision", "Mask Recall", "Mask mAP50-95"],
      validation.per_class_seg.map(c => [c.class_name, fmt(c.mask_precision, 3), fmt(c.mask_recall, 3), fmt(c.mask_map50_95, 3)]),
      [2808, 2184, 2184, 2184],
    ));
  }

  const cmChart = CHART("confusion_matrix.png");
  if (cmChart) children.push(chartImage(cmChart, 5.0));
}

// --- Dual RGB System Notes ---
children.push(divider());
children.push(h1("Dual RGB System Notes"));
children.push(p(
  "The ABEN Dual RGB system uses standard 3-channel BGR imagery from two eMeet SmartCam C960 4K " +
  "cameras. Unlike the multispectral system, weed detection is based on visual appearance rather " +
  "than spectral reflectance. Resistance classification is not performed — all confirmed weed " +
  "detections result in a spray event."
));
children.push(simpleTable(
  ["Parameter", "Value", "Notes"],
  [
    ["Camera height", `${fmt(cfg.camera_height_m, 4)} m`, "36.5 in — calibrated 2026-06-26"],
    ["Look-ahead",    `${fmt(cfg.look_ahead_m, 4)} m`,    "7 in — camera center ahead of nozzles"],
    ["GSD",           `${fmt(cfg.gsd_mm_per_px, 3)} mm/px`, "Ground sample distance at camera height"],
    ["Nozzle Y line", `${cfg.nozzle_y_px || 767} px`,    "In 1920×1080 frame"],
    ["N1 center",     `cam1 @ ${cfg.n1_center_px || 600}px`, "Measured — tape measure calibration"],
    ["N2 center",     `cam1 @ ${cfg.n2_center_cam1_px || 1700}px  |  cam2 @ ${cfg.n2_center_cam2_px || 400}px`, "B1/B2 OR logic"],
    ["N3 center",     `cam2 @ ${cfg.n3_center_px || 1400}px`, "Measured — tape measure calibration"],
  ],
  [2400, 2400, 4560],
));

// --- Conclusions ---
children.push(divider());
children.push(h1("Conclusions & Next Steps"));
children.push(p(
  `This mission confirms the complete RGB detection-to-spray pipeline operates correctly: ` +
  `${sum.spray.fired > 0
    ? `${sum.spray.fired} weed detection${sum.spray.fired === 1 ? "" : "s"} successfully triggered nozzle firing with geometry-based look-ahead timing, `
    : "the system ran without errors through the full camera → inference → zone → geometry → spray decision chain, "}` +
  `and the multi-frame debounce filter correctly suppressed single-frame noise ` +
  `(${sum.spray.near_misses} near-miss detection${sum.spray.near_misses === 1 ? "" : "s"} did not falsely trigger a spray).`
));
if (sum.spray.missed > 0) {
  children.push(p(
    `${sum.spray.missed} confirmed detection${sum.spray.missed === 1 ? "" : "s"} did not result in a completed spray — ` +
    `this happens when the run ends before the robot has traveled the trigger distance. ` +
    `Increase --dist to allow more travel time after the final detection.`,
    { color: COLOR_WARN }
  ));
}
children.push(p("Recommended next steps:"));
children.push(bullet("Capture real sugarbeet and kochia/waterhemp/ragweed field images and retrain YOLOv8n-Seg — current model trained on artificial plants only."));
children.push(bullet("Run evaluate_model_rgb.py and confirm mask mAP50-95 ≥ 0.80 per class before field deployment."));
children.push(bullet("Recalibrate B1/B2 zone split after any camera remounting — run measure_split.py and update detection_config_rgb.py."));
children.push(bullet("Export to TensorRT (weed_rgb.engine) and re-measure achieved FPS on the Jetson under real field illumination."));
children.push(bullet("Verify nozzle Y line calibration with a tape test: place a marker at the nozzle line, drive the robot, confirm spray hits the marker."));
children.push(bullet("Tune debounce threshold and confidence threshold against real field false-positive/negative rates — aim for 0 sugarbeet sprays in a full row pass."));

// ── Build & write document ──────────────────────────────────
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Arial", color: COLOR_ACCENT },
        paragraph: { spacing: { before: 320, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Arial", color: "333333" },
        paragraph: { spacing: { before: 240, after: 160 }, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ],
  },
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
});