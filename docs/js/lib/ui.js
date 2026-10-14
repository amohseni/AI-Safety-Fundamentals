// Small control builders. Every control shows its plain name and current value.

export function slider(parent, { name, min, max, step, value, format = (v) => v, onInput }) {
  const wrap = document.createElement('label');
  wrap.className = 'ctl';
  wrap.innerHTML = `<span class="ctl-head"><span class="ctl-name">${name}</span><output></output></span>`;
  const input = document.createElement('input');
  Object.assign(input, { type: 'range', min, max, step, value });
  const out = wrap.querySelector('output');
  const update = () => { out.textContent = format(Number(input.value)); };
  input.addEventListener('input', () => { update(); onInput && onInput(Number(input.value)); });
  wrap.append(input);
  parent.append(wrap);
  update();
  return {
    el: input,
    get value() { return Number(input.value); },
    set(v) { input.value = v; update(); },
  };
}

export function segmented(parent, { name, options, value, onChange }) {
  const wrap = document.createElement('div');
  wrap.className = 'ctl';
  wrap.setAttribute('role', 'group');
  wrap.setAttribute('aria-label', name);
  wrap.innerHTML = `<span class="ctl-head"><span class="ctl-name">${name}</span></span>`;
  const seg = document.createElement('span');
  seg.className = 'seg';
  const buttons = options.map(([key, label]) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = label;
    b.dataset.key = key;
    b.addEventListener('click', () => { set(key); onChange && onChange(key); });
    seg.append(b);
    return b;
  });
  const set = (key) => buttons.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.key === key)));
  set(value);
  wrap.append(seg);
  parent.append(wrap);
  return { set };
}

export function button(parent, label, onClick, cls = '') {
  const b = document.createElement('button');
  b.type = 'button';
  b.textContent = label;
  if (cls) b.className = cls;
  b.addEventListener('click', onClick);
  parent.append(b);
  return b;
}

export const fmt = (d) => (v) => (Number.isFinite(v) ? v.toFixed(d) : '∞');

export function reducedMotion() {
  return window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

// Reads a CSS custom property, so canvas drawings follow the light or dark theme.
export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
