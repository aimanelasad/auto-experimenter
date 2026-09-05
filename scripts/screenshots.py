"""Capture the screenshots for the submission document from a running app.

    streamlit run app.py --server.port 8511 --server.headless true   # in another shell
    python scripts/screenshots.py --url http://127.0.0.1:8511
    python scripts/screenshots.py --url https://auto-experimenter.streamlit.app/

Uses Playwright with the locally installed Microsoft Edge (no browser download).
Works both for a local server and for Streamlit Community Cloud, which embeds
the app in an iframe.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent.parent / "submission" / "screenshots"
SCROLL_MAIN = "(y) => { const m = document.querySelector('[data-testid=\"stMain\"]'); if (m) m.scrollTo(0, y); }"
SCROLL_BY = "(dy) => { const m = document.querySelector('[data-testid=\"stMain\"]'); if (m) m.scrollBy(0, dy); }"


def app_context(page, timeout_s: float = 90.0):
    """Return the frame that holds the Streamlit app (an iframe on Community Cloud, else the page)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        for frame in page.frames:
            try:
                if frame.locator("[data-testid='stMain']").count() > 0:
                    return frame
            except Exception:  # noqa: BLE001 - frame may be navigating
                pass
        time.sleep(1.0)
    raise TimeoutError("no Streamlit main container found in any frame")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8511")
    parser.add_argument("--live-wait", type=float, default=6.0, help="seconds into the live run before the shot")
    parser.add_argument("--expect", default="Winner:", help="text that must be visible before the first shot")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1.5)
        page.goto(args.url, wait_until="networkidle", timeout=120_000)
        app = app_context(page)
        app.get_by_text(args.expect, exact=False).first.wait_for(timeout=120_000)
        time.sleep(3.0)  # let plotly finish drawing
        page.screenshot(path=str(OUT / "01_overview.png"))

        def shoot_section(heading: str, name: str, offset: int = 24) -> None:
            el = app.get_by_role("heading", name=heading).first
            el.scroll_into_view_if_needed()
            app.evaluate(SCROLL_BY, -offset)
            time.sleep(1.2)
            page.screenshot(path=str(OUT / name))

        shoot_section("Hypothesis trail", "02_trail.png")
        shoot_section("Conclusion", "04_narration.png")
        shoot_section("Retention actions for the top decile", "05_retention.png")

        app.evaluate(SCROLL_MAIN, 0)
        time.sleep(0.5)
        app.get_by_role("button", name="Run the loop now", exact=False).click()
        time.sleep(args.live_wait)
        app.evaluate(SCROLL_MAIN, 0)
        page.screenshot(path=str(OUT / "03_live_run.png"))
        app.get_by_text("Done in", exact=False).first.wait_for(timeout=300_000)
        time.sleep(2.0)
        page.screenshot(path=str(OUT / "06_live_done.png"))
        browser.close()
    for f in sorted(OUT.glob("0*.png")):
        print("written", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
