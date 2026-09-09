// docx_report_helpers.js
// ABEN Dual RGB Detection System — Shared Word-report building blocks
//
// Extracted from generate_report_rgb.js so the mission-report
// generator and generate_gui_session_report.js build visually
// consistent documents (same colors, same table/heading styling)
// from one place instead of two independently-drifting copies.
//
// generate_report_rgb.js still works unmodified -- it defines these
// same helpers inline, so nothing there needs to change. New report
// generators should require() this module instead of re-copying them.

const fs = require("fs");
const {
  Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, AlignmentType, LevelFormat, HeadingLevel, BorderStyle,
  WidthType, ShadingType, VerticalAlign,
} = require("docx");

const PAGE_WIDTH = 12240, PAGE_HEIGHT = 15840, MARGIN = 1440;
const CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN; // 9360

const COLOR_ACCENT = "2E75B6";
const COLOR_HEAD_BG = "2E75B6";
const COLOR_ROW_ALT = "EEF3F8";
const COLOR_OK = "27AE60";
const COLOR_WARN = "C0392B";

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const cellBorders = { top: border, bottom: border, left: border, right: border };

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

// Standard document-level styles/numbering/page setup, so every
// report generator produces the same look without repeating this
// block. Callers spread `...docStyles` into their `new Document({...})`
// call's `styles`/`numbering` keys, and use PAGE_WIDTH/PAGE_HEIGHT/
// MARGIN for `sections[0].properties.page`.
const docStyles = {
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
};

module.exports = {
  PAGE_WIDTH, PAGE_HEIGHT, MARGIN, CONTENT_WIDTH,
  COLOR_ACCENT, COLOR_HEAD_BG, COLOR_ROW_ALT, COLOR_OK, COLOR_WARN,
  h1, h2, p, bullet, divider, statLine, simpleTable, chartImage, imageSizeOf,
  fmt, pct, docStyles,
};
