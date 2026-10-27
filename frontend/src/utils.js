// Group a parallel array of dates + regime labels into contiguous segments,
// e.g. for rendering background ReferenceAreas on a chart.
export function groupIntoSegments(dates, labels) {
  const segments = [];
  let current = null;
  for (let i = 0; i < dates.length; i++) {
    const label = labels[i];
    if (!current || current.label !== label) {
      if (current) segments.push(current);
      current = { label, start: dates[i], end: dates[i] };
    } else {
      current.end = dates[i];
    }
  }
  if (current) segments.push(current);
  return segments;
}

export function fmtPct(x, digits = 1) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return `${(x * 100).toFixed(digits)}%`;
}

export function fmtNum(x, digits = 2) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return x.toFixed(digits);
}
