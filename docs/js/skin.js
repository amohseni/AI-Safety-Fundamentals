// Skins: alternative visual designs for the same pages, kept in css/skins/. The default is the
// Tufte grid in site.css and needs no skin file. A skin is chosen with ?skin=<name> on any address
// or from skins.html, and is remembered in this browser. This script runs before the page is drawn,
// so charts that measure their type read the skin's sizes and faces.
(function () {
  var SKINS = [['tufte', 'Tufte'], ['mohseni', 'Your website'], ['swiss', 'Swiss'], ['interface', 'Interface'], ['folio', 'Folio'], ['voices', 'Two voices']];
  var q = new URLSearchParams(location.search).get('skin'), name = q, bar = null;
  try { if (q) localStorage.setItem('primer-skin', q); name = localStorage.getItem('primer-skin'); bar = localStorage.getItem('primer-skin-bar'); } catch (e) { /* storage unavailable */ }
  if (!SKINS.some(function (s) { return s[0] === name; })) name = 'tufte';
  var base = document.currentScript.src.replace(/js\/skin\.js.*$/, '');
  document.documentElement.dataset.skin = name;
  if (name !== 'tufte') document.write('<link rel="stylesheet" href="' + base + 'css/skins/' + name + '.css">');
  if (!bar) return;
  // A reader who has opened skins.html gets a bar for switching skins on any page.
  document.addEventListener('DOMContentLoaded', function () {
    var el = document.createElement('div');
    el.setAttribute('role', 'group'); el.setAttribute('aria-label', 'Skin');
    el.style.cssText = 'position:fixed;left:50%;bottom:12px;transform:translateX(-50%);z-index:50;display:flex;flex-wrap:wrap;justify-content:center;gap:4px;padding:6px 8px;max-width:calc(100vw - 24px);' +
      'background:var(--paper);border:1px solid var(--quiet);border-radius:6px;font:13px/1.2 system-ui,sans-serif;';
    SKINS.concat([['', 'Hide']]).forEach(function (s) {
      var b = document.createElement('button');
      b.type = 'button'; b.textContent = s[1];
      b.style.cssText = 'font:inherit;padding:4px 8px;border-radius:4px;border:1px solid ' + (s[0] === name ? 'var(--ink)' : 'transparent') + ';background:transparent;color:var(--ink);cursor:pointer;';
      if (s[0] === name) b.setAttribute('aria-pressed', 'true');
      b.addEventListener('click', function () {
        try { if (s[0]) localStorage.setItem('primer-skin', s[0]); else localStorage.removeItem('primer-skin-bar'); } catch (e) { /* ignore */ }
        var u = new URL(location.href); u.searchParams.delete('skin'); location.replace(u.toString());
      });
      el.appendChild(b);
    });
    document.body.appendChild(el);
  });
})();
