"""Converts report.md to report.html and then to PDF with a headless Chromium browser.

Run from the repository root:  python report/build_pdf.py
Set CHROME=/path/to/chrome if Chrome/Brave/Chromium is not found automatically.
"""
import os
import shutil
import subprocess
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
OUT_PDF = HERE / "Rishant_Ranjan_S24BCAU0036_CSET485_Assignment1.pdf"

CSS = """
@page { size: A4; margin: 18mm 17mm 18mm 17mm; }
body { font-family: "Georgia", "Times New Roman", serif; font-size: 10.5pt; line-height: 1.45; color: #111; }
h1 { font-size: 19pt; margin: 0 0 2pt 0; }
h2 { font-size: 14pt; margin-top: 18pt; border-bottom: 1px solid #bbb; padding-bottom: 2pt; break-after: avoid; }
h3 { font-size: 11.5pt; margin-top: 13pt; break-after: avoid; }
p, li { text-align: left; }
table { border-collapse: collapse; margin: 6pt 0 10pt 0; font-size: 9pt; break-inside: avoid; }
th, td { border: 1px solid #c9c9c9; padding: 3pt 6pt; vertical-align: top; }
th { background: #f1f1ef; text-align: left; }
code { font-family: Menlo, Consolas, monospace; font-size: 8.8pt; background: #f4f4f2; padding: 0 2pt; }
img { display: block; max-width: 100%; margin: 6pt auto 10pt auto; break-inside: avoid; }
img[alt="DP vs EO trade-off"] { max-width: 78%; }
blockquote { margin: 6pt 0 6pt 0; padding: 4pt 10pt; border-left: 3px solid #2a78d6; background: #f6f8fb; }
hr { border: none; border-top: 1px solid #ddd; margin: 14pt 0; }
"""


def find_browser():
    candidates = [os.environ.get("CHROME"),
                  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                  "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
                  "/Applications/Chromium.app/Contents/MacOS/Chromium",
                  shutil.which("google-chrome"), shutil.which("chromium"), shutil.which("chromium-browser")]
    for c in candidates:
        if c and Path(c).exists():
            return c
    raise SystemExit("No Chromium-based browser found; set CHROME=/path/to/browser")


def main():
    md_text = (HERE / "report.md").read_text()
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])
    html = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>CSET485 Assignment 1 - Rishant Ranjan</title><style>{CSS}</style></head>"
            f"<body>{body}</body></html>")
    html_path = HERE / "report.html"
    html_path.write_text(html)
    subprocess.run([find_browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={OUT_PDF}", html_path.as_uri()],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    print("wrote", OUT_PDF)


if __name__ == "__main__":
    main()
