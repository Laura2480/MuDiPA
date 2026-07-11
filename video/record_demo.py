"""
Automated screencast recorder for the MuDiPA demo video (staged & slow-paced).

Records the browser viewport to WEBM and paces each scene to the exact duration
of its narration segment (video/audio/sN_*.mp3) so the ffmpeg mux lines up.

Staging (per reviewer feedback: let actions breathe, show accept/reject, use a
pre-annotated graph, zoom out for overview then zoom in to work):
  S1 intro     load a 9-EDU dialogue, LOAD GOLD (rich graph), zoom OUT (overview)
  S2 link      CLEAR arcs, zoom IN, select an EDU, suggest -> HOVER the accept/
               reject chip (hold so the viewer reads it) -> click accept -> arc
  S3 relation  open the codebook + relation suggestion (ranked labels)
  S4 reasoning hover the arc -> click the (star) -> hold on the rationale panel
  S5 multimodal switch to draddp, loop-play the clips
  S6 close     LOAD GOLD again (full graph) + zoom OUT + export dropdown

Prereqs: app :5050 + DDPE :8092 running; `playwright install chromium` done.
Run:     .venv/Scripts/python.exe video/record_demo.py   -> video/raw/*.webm
"""
import time
from playwright.sync_api import sync_playwright

APP = "http://localhost:5050"
VIEWPORT = {"width": 1920, "height": 1080}
DEMO_DIALOGUE_INDEX = 20          # Dialogue 21 (9 EDU, clean multi-party trade)
MULTIMODAL_DATASET = "draddp"

SCENE_SECS = {"s1_intro": 22, "s2_link": 18, "s3_relation": 18,
              "s4_reasoning": 28, "s_sub": 14, "s5_multimodal": 24, "s6_close": 13}

# Injected fake cursor (Playwright doesn't render the OS pointer) + click ripple.
CURSOR_JS = r"""
if (!window.__curInit) { window.__curInit = 1;
  var svg = "data:image/svg+xml;utf8," + encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" width="30" height="30" viewBox="0 0 24 24">'
    + '<path d="M4 2 L4 21 L9.2 15.6 L12.6 22.5 L15.7 21.1 L12.3 14.4 L19.5 14.4 Z"'
    + ' fill="#111" stroke="#fff" stroke-width="1.4" stroke-linejoin="round"/></svg>');
  var add = function(){
    if (document.getElementById('__cur__')) return;
    var cur = document.createElement('img'); cur.id = '__cur__'; cur.src = svg;
    cur.style.cssText = 'position:fixed;left:-60px;top:-60px;width:30px;height:30px;'
      + 'z-index:2147483647;pointer-events:none;filter:drop-shadow(0 1px 2px rgba(0,0,0,.45));'
      + 'transition:left .05s linear,top .05s linear;';
    (document.body || document.documentElement).appendChild(cur);
    addEventListener('mousemove', function(e){ cur.style.left=e.clientX+'px'; cur.style.top=e.clientY+'px'; }, true);
    addEventListener('mousedown', function(e){
      var r = document.createElement('div');
      r.style.cssText = 'position:fixed;left:'+(e.clientX-17)+'px;top:'+(e.clientY-17)+'px;'
        + 'width:34px;height:34px;border:3px solid #1a6fb0;border-radius:50%;'
        + 'z-index:2147483646;pointer-events:none;opacity:.85;transition:transform .5s ease-out,opacity .5s;';
      (document.body || document.documentElement).appendChild(r);
      requestAnimationFrame(function(){ r.style.transform='scale(2)'; r.style.opacity='0'; });
      setTimeout(function(){ r.remove(); }, 520);
    }, true);
  };
  if (document.body) add(); else addEventListener('DOMContentLoaded', add);
}
"""

# Opaque veil that hides the login form during setup (pointer-events:none so the
# recorder can still click through it). Removed the instant scene 1 begins.
VEIL_JS = r"""
if (!window.__veilInit) { window.__veilInit = 1;
  var add = function(){
    if (document.getElementById('__veil__')) return;
    var v = document.createElement('div'); v.id = '__veil__';
    v.style.cssText = 'position:fixed;inset:0;background:#eef1ee;z-index:2147483640;pointer-events:none;';
    (document.body || document.documentElement).appendChild(v);
  };
  if (document.body) add(); else addEventListener('DOMContentLoaded', add);
}
"""


def hold(page, scene, t0):
    left = SCENE_SECS[scene] - (time.time() - t0)
    if left > 0:
        page.wait_for_timeout(int(left * 1000))


def wait(page, ms):
    page.wait_for_timeout(ms)


def node_center(page, i):
    b = page.evaluate(
        "(i)=>{const n=document.querySelectorAll('.node')[i];if(!n)return null;"
        "const r=n.getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2];}", i)
    return b


def hover_click(page, selector, hold_ms=2500, timeout=4000):
    """Move the cursor onto an element, PAUSE (so the viewer sees it), then click."""
    loc = page.locator(selector).first
    loc.wait_for(state="visible", timeout=timeout)
    box = loc.bounding_box()
    if box:
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=14)
    wait(page, hold_ms)
    loc.click(timeout=timeout)


def zoom(page, which, n=1):
    for _ in range(n):
        try:
            page.get_by_title(which).click(); wait(page, 350)
        except Exception:
            pass


def pan(page, dx, steps=25):
    """Ctrl+drag the canvas horizontally to slide the view (dx<0 reveals the right).
    Drag on an EMPTY strip low on the canvas so we never grab an arc/node."""
    cx, cy = VIEWPORT["width"] // 2, int(VIEWPORT["height"] * 0.85)
    page.keyboard.down("Control")
    page.mouse.move(cx, cy, steps=4); page.mouse.down()
    page.mouse.move(cx + dx, cy, steps=steps)
    page.mouse.up(); page.keyboard.up("Control")


def canvas_center_x(page):
    """x of the visible canvas centre (the .canvas-wrap viewport, excluding the left
    dialogue list and any right panel)."""
    return page.evaluate("()=>{const el=document.querySelector('.canvas-wrap');"
                         "if(!el)return null;const b=el.getBoundingClientRect();return b.x+b.width/2;}")


def center_on(page, idxs):
    """Ctrl+drag so the mean x of the given node indices lands on the canvas centre."""
    cs = [c for c in (node_center(page, i) for i in idxs) if c]
    cxk = canvas_center_x(page)
    if not cs or cxk is None:
        return
    dx = cxk - sum(c[0] for c in cs) / len(cs)
    if abs(dx) > 8:
        pan(page, dx); wait(page, 500)


def main():
    import os
    os.makedirs("video/raw", exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(viewport=VIEWPORT, record_video_dir="video/raw",
                                  record_video_size=VIEWPORT)
        t_rec = time.time()                        # recording starts at context creation
        ctx.add_init_script(CURSOR_JS)             # visible cursor on every page
        ctx.add_init_script(VEIL_JS)               # hide the login form during setup
        page = ctx.new_page()
        page.on("dialog", lambda d: d.accept())   # auto-accept confirm() (clear arcs / load gold)
        page.goto(APP)
        # --- setup (login + dialogue pick): hidden by the veil, trimmed from the video ---
        page.get_by_text("Explore as guest", exact=False).click()
        wait(page, 4000)
        page.get_by_role("combobox").first.select_option(str(DEMO_DIALOGUE_INDEX))
        wait(page, 3500)
        with open("video/_trim.txt", "w") as f:    # build_video trims this many seconds
            f.write(f"{max(0.0, time.time() - t_rec + 0.2):.2f}")
        page.evaluate("() => document.getElementById('__veil__') && document.getElementById('__veil__').remove()")

        # ===== S1: intro — show the pre-annotated graph, slide across it + zoom =====
        t = time.time()
        try:
            page.get_by_text("load gold", exact=True).click()   # rich draft graph
        except Exception as e:
            print("load gold:", e)
        wait(page, 2200)
        zoom(page, "zoom out", 2)           # pull back to see the whole structure
        wait(page, 1200)
        pan(page, -560)                     # slide across the dialogue (reveal the right side)
        wait(page, 1400)
        pan(page, 560)                      # slide back to the start (returns exactly)
        wait(page, 1400)
        hold(page, "s1_intro", t)

        # ===== S2: keep the graph PRE-LABELLED; remove only the demo arc, then
        #          re-suggest it so the surrounding labels give the suggester context =====
        t = time.time()
        try:
            page.get_by_title("reset zoom").click()      # back to 100% (a zoom-in vs the S1 overview)
        except Exception:
            pass
        wait(page, 500)
        center_on(page, [1])                             # centre the worked node (1) in the canvas
        wait(page, 700)
        n0, n1 = node_center(page, 0), node_center(page, 1)
        if n0 and n1:                                     # delete the existing 0->1 arc (✕ on hover)
            page.mouse.move((n0[0] + n1[0]) / 2, (n0[1] + n1[1]) / 2, steps=10); wait(page, 800)
            try:
                page.locator('.arc-del').first.click(timeout=3000)
            except Exception as e:
                print("del arc:", e)
            wait(page, 1200)
        n1 = node_center(page, 1)           # target EDU 1 ("yep plenty")
        if n1:
            page.mouse.move(*n1, steps=12); wait(page, 400); page.mouse.click(*n1)
        wait(page, 900)
        try:
            page.locator('.node-bulb').first.click(timeout=4000)   # 💡 suggest parents
        except Exception as e:
            print("bulb:", e)
        wait(page, 1800)                    # the amber chip (score + accept/reject) appears
        try:
            hover_click(page, '.pc-yes', hold_ms=3200)   # HOVER accept ~3s, then click ✓
        except Exception as e:
            print("accept:", e)
        wait(page, 1500)                    # let the arc settle
        hold(page, "s2_link", t)

        # ===== S3: LABEL the link — relation ranking + assign a relation =====
        t = time.time()
        n0, n1 = node_center(page, 0), node_center(page, 1)
        mid = ((n0[0] + n1[0]) / 2, (n0[1] + n1[1]) / 2) if (n0 and n1) else None
        if mid:
            page.mouse.click(*mid); wait(page, 1200)       # select arc -> relation picker opens
        try:
            page.locator('[data-tip="relation suggestion"]').first.click(timeout=4000)   # rank labels
        except Exception as e:
            print("rel sugg:", e)
        wait(page, 3000)                                   # show the ranked SDRT labels
        # assign 'Elaboration' (a plausible label the reasoning engine will vet & correct)
        try:
            page.locator('.arc-rel-picker').get_by_text("Elaboration", exact=True).first.click(timeout=3000)
        except Exception:
            try:
                page.locator('.arp-scores').first.locator('button, [role=button], div').first.click(timeout=3000)
            except Exception as e:
                print("assign:", e)
        wait(page, 2500)                                   # keep the assigned label on screen
        hold(page, "s3_relation", t)

        # ===== S4: the reasoning engine VETS the relation (disagrees -> suggests the correct one) =====
        t = time.time()
        try:
            page.locator('[data-tip="explain this relation"]').first.click(timeout=4000)   # ✦ relation
        except Exception as e:
            print("rel explain:", e)
        wait(page, 12000)                                  # relation rationale renders + read
        hold(page, "s4_reasoning", t)

        # ===== S_sub: sub-dialogues — show the discourse threads =====
        t = time.time()
        # close the relation-explanation panel (its full-screen scrim blocks the threads button)
        for _ in range(3):
            try:
                if page.locator('.canvas-scrim').count():
                    page.locator('.canvas-scrim').first.click(position={"x": 30, "y": 30}, timeout=1500)
                    wait(page, 300)
                else:
                    break
            except Exception:
                break
        try:
            page.keyboard.press("Escape")                  # belt-and-suspenders
        except Exception:
            pass
        wait(page, 400)
        try:
            page.get_by_title("reset zoom").click()
        except Exception:
            pass
        zoom(page, "zoom out", 3)                          # shrink so the thread paths fit the frame
        wait(page, 500)
        try:
            page.get_by_text("threads", exact=False).first.click(timeout=4000)    # DISCOURSE PATHS panel (canvas narrows)
        except Exception as e:
            print("threads:", e)
        wait(page, 1200)
        center_on(page, list(range(9)))                    # re-centre the graph in the now-narrower canvas
        wait(page, 1000)
        for rx, ry in [(1740, 210), (1740, 260), (1740, 160)]:  # hover each thread -> path lights up
            page.mouse.move(rx, ry, steps=10); wait(page, 2300)
        try:
            page.get_by_text("threads", exact=False).first.click(timeout=2500)   # toggle the drawer shut
        except Exception as e:
            print("threads close:", e)
        wait(page, 600)
        hold(page, "s_sub", t)

        # ===== S5: multimodal (the star) — switch, loop-play the clips =====
        t = time.time()
        try:
            page.get_by_title("choose dataset").click(); wait(page, 700)
            page.get_by_text(MULTIMODAL_DATASET, exact=False).first.click(timeout=4000)
            wait(page, 4500)
            page.evaluate("() => document.querySelectorAll('video').forEach(v=>{"
                          "v.muted=true; v.loop=true; try{v.currentTime=0.5;}catch(e){} v.play().catch(()=>{});})")
            wait(page, 3000)
        except Exception as e:
            print("multimodal:", e)
        hold(page, "s5_multimodal", t)

        # ===== S6: close — full graph again + zoom out + export =====
        t = time.time()
        try:
            page.get_by_role("combobox").first.select_option(str(DEMO_DIALOGUE_INDEX))
            wait(page, 2500)
            page.get_by_text("load gold", exact=True).click(); wait(page, 1500)
            zoom(page, "zoom out", 3); wait(page, 800)
            page.get_by_role("combobox", name="export").click()
        except Exception as e:
            print("close:", e)
        hold(page, "s6_close", t)

        path = page.video.path()
        ctx.close(); browser.close()
        print("RAW VIDEO ->", path)


if __name__ == "__main__":
    main()
