// Banded contour maps for pages 2.1 and 2.6. The grid extends one cell past the plotted region on
// every side and the bands are clipped to the region, so the outlines d3.contours draws along the
// grid's border fall outside the chart and no frame appears.
const d3 = window.d3;

// g: a d3 selection to draw into (inside the chart's inner area, iw by ih pixels).
// f(x, y): the value to map; regions with f >= threshold are filled, so pass minus a loss to darken low ground.
export function bands(g, { f, xd, yd, iw, ih, n = 140, thresholds, opacity, stroke = true }) {
  const m = n + 2, vals = new Float64Array(m * m);
  const dx = (xd[1] - xd[0]) / n, dy = (yd[1] - yd[0]) / n;
  for (let j = 0; j < m; j++) for (let i = 0; i < m; i++) vals[j * m + i] = f(xd[0] + (i - 0.5) * dx, yd[1] - (j - 0.5) * dy);
  const cont = d3.contours().size([m, m]).thresholds(thresholds)(Array.from(vals));
  const sx = iw / n, sy = ih / n;
  const proj = d3.geoTransform({ point(x, y) { this.stream.point((x - 1) * sx, (y - 1) * sy); } });
  const id = `bands-${Math.random().toString(36).slice(2)}`;
  g.selectAll('*').remove();
  g.append('clipPath').attr('id', id).append('rect').attr('width', iw).attr('height', ih);
  g.append('g').attr('clip-path', `url(#${id})`).selectAll('path').data(cont).join('path').attr('d', d3.geoPath(proj))
    .attr('fill', 'var(--ink)').attr('fill-opacity', opacity)
    .attr('stroke', stroke ? 'var(--quiet)' : 'none').attr('stroke-opacity', 0.35).attr('stroke-width', 0.7);
}
