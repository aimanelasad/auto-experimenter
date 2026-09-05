"""Capture the screenshots for the submission document from a running app.

    streamlit run app.py --server.port 8511 --server.headless true   # in another shell
    python scripts/screenshots.py --url http://127.0.0.1:8511

Uses Playwright with the locally installed Microsoft Edge (no browser download).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent.parent / "submission" / "screenshots"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8511")
    parser.add_argument("--live-wait", type=float, default=6.0, help="seconds into the live run before the shot")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1.5)
        page.goto(args.url, wait_until="networkidle")
        page.get_by_text("Winner:", exact=False).first.wait_for(timeout=60_000)
        time.sleep(2.5)  # let plotly finish drawing
        page.screenshot(path=str(OUT / "01_overview.png"))

        # Streamlit scrolls its main container, not the window.
        scroll_main = "(y) => { const m = document.querySelector('[data-testid=\"stMain\"]'); if (m) m.scrollTo(0, y); }"

        def shoot_section(heading: str, name: str, offset: int = 24) -> None:
            el = page.get_by_role("heading", name=heading).first
            el.scroll_into_view_if_needed()
            page.evaluate("(dy) => { const m = document.querySelector('[data-testid=\"stMain\"]'); if (m) m.scrollBy(0, dy); }", -offset)
            time.sleep(1.0)
            page.screenshot(path=str(OUT / name))

        shoot_section("Hypothesis trail", "02_trail.png")
        shoot_section("Conclusion", "04_narration.png")
        shoot_section("Retention actions for the top decile", "05_retention.png")

        page.evaluate(scroll_main, 0)
        time.sleep(0.5)
        page.get_by_role("button", name="Run the loop now", exact=False).click()
        time.sleep(args.live_wait)
        page.evaluate(scroll_main, 0)
        page.screenshot(path=str(OUT / "03_live_run.png"))
        page.get_by_text("Done in", exact=False).first.wait_for(timeout=180_000)
        time.sleep(1.5)
        page.screenshot(path=str(OUT / "06_live_done.png"))
        browser.close()
    for f in sorted(OUT.glob("0*.png")):
        print("written", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
