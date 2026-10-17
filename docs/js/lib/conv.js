// Convolution filters on small images (page 1.4). Pure JavaScript, no DOM, so it runs in Node tests.
// Images are row-major arrays of brightness from 0 to 255; color images have three planes (R, G, B).
// As in neural network libraries, the filter is applied without flipping (cross-correlation),
// and only at positions where it fits inside the image.

export const PRESETS = {
  vertical: { name: 'Vertical edge', k: [-1, 0, 1, -2, 0, 2, -1, 0, 1] },
  horizontal: { name: 'Horizontal edge', k: [-1, -2, -1, 0, 0, 0, 1, 2, 1] },
  blur: { name: 'Blur', k: [1, 1, 1, 1, 1, 1, 1, 1, 1].map((v) => v / 9) },
  sharpen: { name: 'Sharpen', k: [0, -1, 0, -1, 5, -1, 0, -1, 0] },
};

// A test image drawn in code: a square, a disk, and a band running like "/", light on dark.
export function shapes(w = 240, h = 160) {
  const img = new Float64Array(w * h);
  const DARK = 30, LIGHT = 220;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let v = DARK;
      if (x >= 25 && x < 85 && y >= 45 && y < 105) v = LIGHT; // square
      if ((x - 135) ** 2 + (y - 75) ** 2 < 32 ** 2) v = LIGHT; // disk
      if (x >= 180 && x < 228 && y >= 25 && y < 135 && Math.abs(x + y - 270) < 10) v = LIGHT; // band "/"
      img[y * w + x] = v;
    }
  }
  return img;
}

// Applies a 3 by 3 filter k (row-major) to a gray image; returns the (w-2) by (h-2) feature map.
export function conv3(img, w, h, k) {
  const ow = w - 2, oh = h - 2, out = new Float64Array(ow * oh);
  for (let y = 0; y < oh; y++) {
    for (let x = 0; x < ow; x++) {
      let s = 0;
      for (let dy = 0; dy < 3; dy++) for (let dx = 0; dx < 3; dx++) s += k[dy * 3 + dx] * img[(y + dy) * w + x + dx];
      out[y * ow + x] = s;
    }
  }
  return { out, ow, oh };
}

// Applies one color filter f (3 planes of size by size, R, G, B) to a color image given as three
// planes. The ImageNet mean is subtracted first, as the network does. Returns the raw sums.
export const IMAGENET_MEAN = [123.68, 116.78, 103.94];
export function convColor(planes, w, h, f, size) {
  const ow = w - size + 1, oh = h - size + 1, out = new Float64Array(ow * oh);
  for (let c = 0; c < 3; c++) {
    const P = planes[c], F = f[c], mean = IMAGENET_MEAN[c];
    for (let y = 0; y < oh; y++) {
      for (let x = 0; x < ow; x++) {
        let s = 0;
        for (let dy = 0; dy < size; dy++) {
          const row = (y + dy) * w + x, frow = dy * size;
          for (let dx = 0; dx < size; dx++) s += F[frow + dx] * (P[row + dx] - mean);
        }
        out[y * ow + x] += s;
      }
    }
  }
  return { out, ow, oh };
}

export const sum = (k) => k.reduce((a, b) => a + b, 0);
