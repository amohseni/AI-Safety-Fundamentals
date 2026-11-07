// Default training settings for the probes on page 6.3, shared by the page and its tests.
export const PROBE_DEFAULTS = {
  linear: { l2: 0.01, steps: 300, lr: 0.05 },
  mlp: { hidden: 64, l2: 0.0001, steps: 400, lr: 0.01, seed: 1 },
};
