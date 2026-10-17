// Starting queries, keys, and values for page 3.4 (shared with tools/test-page-3-4.mjs).
export const START = {
  q: [[0.8, 0.9], [0.9, -0.4], [-0.3, 1.0], [1.0, -0.8]],
  k: [[1.6, 0.2], [-0.1, 1.5], [-1.5, -0.3], [0.3, -1.5]],
  v: [[-1.5, 1.0], [1.5, 1.2], [1.2, -1.3], [-1.3, -1.1]],
};
export const LIMIT = 3; // queries and keys stay inside [-3, 3] in each coordinate
// The worked example in the static figure: position 3 of three tokens.
export const FIGURE = {
  q: [0.6, 1.6],
  k: [[1.6, 0.2], [-0.1, 1.5], [-1.5, -0.3]],
  v: [[-1.5, 1.0], [1.5, 1.2], [1.2, -1.3]],
};
