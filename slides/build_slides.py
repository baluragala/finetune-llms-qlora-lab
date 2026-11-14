"""
build_slides.py: turn the deck source (slides/src/) into ONE self-contained presenter page.

    python3 slides/build_slides.py      ->  writes slides/index.html

Source layout (edit these, then rebuild):
    slides/src/deck.json          title, slide order, sections, fonts
    slides/src/deck.css           shared styles for every slide
    slides/src/slides/<id>.html   one <section> per slide; an optional <aside> holds speaker notes

The output needs no server and no build tools: open it in any browser, or via GitHub Pages.
Keys: → / Space / PageDown = next (reveals checkpoint answers first) · ← = back · N = notes ·
F = fullscreen · Home / End = first / last. Clicking the slide also advances.
"""
import html
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")

deck = json.load(open(os.path.join(SRC, "deck.json"), encoding="utf-8"))
css = open(os.path.join(SRC, "deck.css"), encoding="utf-8").read()

# Section titles for the menu: map a slide id to the section description it starts.
starts = {s["start"]: s["description"] for s in deck.get("sections", {}).values()}

sections = []
menu = []
for n, sid in enumerate(deck["order"], start=1):
    raw = open(os.path.join(SRC, "slides", f"{sid}.html"), encoding="utf-8").read().strip()
    if not raw.startswith("<section"):
        raise SystemExit(f"{sid}.html must contain exactly one <section>")
    sections.append(raw)
    h = re.search(r"<h[12][^>]*>(.*?)</h[12]>", raw, re.S)
    title = re.sub(r"<[^>]+>", "", h.group(1)).strip() if h else sid
    group = starts.get(sid)
    if group:
        menu.append(f'<p class="menu-group">{html.escape(group.split(":")[0].rstrip("."))}</p>')
    menu.append(f'<button data-go="{n - 1}"><span>{n}</span> {html.escape(title)}</button>')

fonts = "&".join(
    f["href"].split("?", 1)[1].replace("&display=swap", "")
    for f in deck.get("faces", {}).values() if "href" in f
)
font_link = f"https://fonts.googleapis.com/css2?{fonts}&display=swap" if fonts else ""

page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(deck["title"])}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="{font_link}">
<style>
/* ---------- presenter shell ---------- */
html, body {{ margin: 0; height: 100%; background: #0E1626; overflow: hidden; }}
#viewport {{ position: fixed; inset: 0 0 56px 0; display: flex; align-items: center; justify-content: center; }}
#canvas {{ width: 1920px; height: 1080px; position: relative; transform-origin: center center; flex: none;
           box-shadow: 0 10px 60px rgba(0,0,0,.45); }}
#canvas > section {{ position: absolute; inset: 0; box-sizing: border-box; overflow: hidden; display: none !important; }}
#canvas > section.active {{ display: flex !important; }}
#canvas > section aside {{ display: none; }}
#canvas h1, #canvas h2, #canvas h3, #canvas p, #canvas ul, #canvas ol, #canvas table {{ margin: 0; }}
#canvas table {{ border-collapse: collapse; }}
#canvas th, #canvas td {{ padding: .35em .6em; text-align: left; border-bottom: 1px solid rgba(20,33,61,.15); }}
#canvas ol, #canvas ul {{ padding-left: 1.4em; }}

#bar {{ position: fixed; left: 0; right: 0; bottom: 0; height: 56px; display: flex; align-items: center; gap: 8px;
        padding: 0 16px; background: #0E1626; color: #D5DCEA; font: 15px/1 'IBM Plex Sans', Arial, sans-serif; }}
#bar button {{ background: #1E2D4F; color: #F7F5EF; border: 1px solid #34497A; border-radius: 8px;
              padding: 8px 14px; font: inherit; cursor: pointer; }}
#bar button:hover {{ background: #2B3F6A; }}
#count {{ min-width: 72px; text-align: center; }}
#progress {{ position: fixed; left: 0; bottom: 56px; height: 4px; background: #F2A766; transition: width .25s; }}
#hint {{ margin-left: auto; opacity: .7; }}
#notes {{ position: fixed; left: 16px; right: 16px; bottom: 72px; max-height: 32vh; overflow: auto; display: none;
          background: #F7F5EF; color: #14213D; border-radius: 12px; padding: 16px 20px;
          font: 17px/1.5 'IBM Plex Sans', Arial, sans-serif; box-shadow: 0 8px 40px rgba(0,0,0,.4); }}
#notes.show {{ display: block; }}
#notes b {{ color: #B5541A; }}
#menu {{ position: fixed; top: 0; left: 0; bottom: 56px; width: min(420px, 90vw); overflow: auto; display: none;
         background: #14213D; padding: 12px; box-sizing: border-box; }}
#menu.show {{ display: block; }}
#menu button {{ display: block; width: 100%; text-align: left; background: none; border: 0; color: #D5DCEA;
               padding: 8px 10px; border-radius: 6px; font: 15px/1.3 'IBM Plex Sans', Arial, sans-serif; cursor: pointer; }}
#menu button:hover, #menu button.here {{ background: #1E2D4F; color: #fff; }}
#menu button span {{ display: inline-block; width: 28px; opacity: .6; }}
.menu-group {{ color: #F2A766; font: 600 12px/1 'IBM Plex Sans', Arial, sans-serif; letter-spacing: 2px;
               text-transform: uppercase; margin: 14px 10px 6px; }}
@media (max-width: 640px) {{ #hint {{ display: none; }} #bar {{ gap: 4px; padding: 0 8px; }} #bar button {{ padding: 8px 10px; }} }}

/* ---------- deck styles (slides/src/deck.css) ---------- */
{css}
</style>
</head>
<body>
<div id="viewport"><div id="canvas">
{chr(10).join(sections)}
</div></div>
<div id="progress"></div>
<nav id="menu" aria-label="Slides">{''.join(menu)}</nav>
<div id="notes" aria-live="polite"></div>
<div id="bar">
  <button id="menuBtn" title="All slides (M)">☰ Slides</button>
  <button id="prev" title="Previous (←)">‹</button>
  <span id="count"></span>
  <button id="next" title="Next (→ / Space)">›</button>
  <button id="notesBtn" title="Speaker notes (N)">Notes</button>
  <button id="fsBtn" title="Fullscreen (F)">⛶</button>
  <span id="hint">→ next · ← back · N notes · F fullscreen · click a checkpoint to reveal the answer</span>
</div>
<script>
(() => {{
  const slides = [...document.querySelectorAll('#canvas > section')];
  const canvas = document.getElementById('canvas');
  const notes = document.getElementById('notes');
  const menu = document.getElementById('menu');
  let i = 0;

  function fit() {{
    const vp = document.getElementById('viewport').getBoundingClientRect();
    const s = Math.min(vp.width / 1920, vp.height / 1080);
    canvas.style.transform = 'scale(' + s + ')';
  }}
  const frags = s => [...s.querySelectorAll('.fragment')];

  function show(n, revealAll) {{
    i = Math.max(0, Math.min(slides.length - 1, n));
    slides.forEach((s, k) => s.classList.toggle('active', k === i));
    frags(slides[i]).forEach(f => f.classList.toggle('visible', !!revealAll));
    document.getElementById('count').textContent = (i + 1) + ' / ' + slides.length;
    document.getElementById('progress').style.width = (100 * i / (slides.length - 1)) + '%';
    const a = slides[i].querySelector('aside');
    notes.innerHTML = '<b>Speaker notes · slide ' + (i + 1) + '</b><br>' + (a ? a.innerHTML : '(none)');
    menu.querySelectorAll('button').forEach(b => b.classList.toggle('here', +b.dataset.go === i));
    history.replaceState(null, '', '#' + (i + 1));
  }}
  function next() {{
    const hidden = frags(slides[i]).filter(f => !f.classList.contains('visible'));
    if (hidden.length) hidden[0].classList.add('visible'); else show(i + 1);
  }}
  function prev() {{
    const shown = frags(slides[i]).filter(f => f.classList.contains('visible'));
    if (shown.length) shown[shown.length - 1].classList.remove('visible'); else show(i - 1, true);
  }}
  const fs = () => document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen?.();

  document.getElementById('next').onclick = next;
  document.getElementById('prev').onclick = prev;
  document.getElementById('notesBtn').onclick = () => notes.classList.toggle('show');
  document.getElementById('menuBtn').onclick = () => menu.classList.toggle('show');
  document.getElementById('fsBtn').onclick = fs;
  menu.onclick = e => {{ const b = e.target.closest('button'); if (b) {{ show(+b.dataset.go); menu.classList.remove('show'); }} }};
  document.getElementById('viewport').onclick = () => {{ if (menu.classList.contains('show')) menu.classList.remove('show'); else next(); }};
  document.addEventListener('keydown', e => {{
    if (['ArrowRight', ' ', 'PageDown'].includes(e.key)) {{ e.preventDefault(); next(); }}
    else if (['ArrowLeft', 'PageUp'].includes(e.key)) {{ e.preventDefault(); prev(); }}
    else if (e.key === 'Home') show(0);
    else if (e.key === 'End') show(slides.length - 1);
    else if (e.key.toLowerCase() === 'n') notes.classList.toggle('show');
    else if (e.key.toLowerCase() === 'm') menu.classList.toggle('show');
    else if (e.key.toLowerCase() === 'f') fs();
    else if (e.key === 'Escape') {{ menu.classList.remove('show'); notes.classList.remove('show'); }}
  }});
  // Swipe on touch screens.
  let x0 = null;
  addEventListener('touchstart', e => {{ x0 = e.touches[0].clientX; }}, {{ passive: true }});
  addEventListener('touchend', e => {{
    if (x0 === null) return; const dx = e.changedTouches[0].clientX - x0; x0 = null;
    if (Math.abs(dx) > 50) {{ dx < 0 ? next() : prev(); }}
  }});
  addEventListener('resize', fit);
  addEventListener('hashchange', () => show((parseInt(location.hash.slice(1), 10) || 1) - 1));
  fit();
  show((parseInt(location.hash.slice(1), 10) || 1) - 1);
}})();
</script>
</body>
</html>
"""

out = os.path.join(HERE, "index.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(page)
print(f"Wrote {out}  ({len(deck['order'])} slides, {len(page) // 1024} KB)")
