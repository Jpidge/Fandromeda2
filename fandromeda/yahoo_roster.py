"""User-directed local Yahoo roster capture.

This helper never reads, stores, or submits Yahoo credentials. It opens a
visible persistent Playwright browser profile; the user completes login and
any multi-factor prompt directly in Yahoo before confirming the capture.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
import shutil
import time


def _clean_visible_text(text: str) -> str:
    """Normalize browser-visible text without altering the roster content."""

    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    return "\n".join(lines).strip() + "\n"


def _roster_row_count(text: str) -> int:
    """Return Yahoo-style team/position lines found in a roster export.

    The dashboard's Yahoo parser expects lines such as ``TB - QB``.  Requiring
    several of those lines prevents an advertisement or a Yahoo navigation
    shell from ever being mistaken for a roster capture.
    """

    return len(
        re.findall(
            r"(?mi)^[A-Z]{2,4}\s*-\s*(?:QB|RB|WR|TE|K|DEF)\s*$",
            text,
        )
    )


def _is_yahoo_roster_text(text: str) -> bool:
    # Headers are useful but not required: Yahoo's whole-page copy can omit
    # them while still preserving the slot/player/TEAM - POSITION structure
    # consumed by ``parse_yahoo_roster``.
    return _roster_row_count(text) >= 8


def _windows_clipboard_text() -> str:
    """Read a user-confirmed Windows clipboard copy without extra packages."""

    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            return root.clipboard_get()
        finally:
            root.destroy()
    except Exception as exc:
        raise RuntimeError(
            "FANDROMEDA could not read the Windows clipboard. "
            "After copying the Yahoo roster, retry or install a standard "
            "Python build with Tk support."
        ) from exc


def _copy_active_yahoo_page(page) -> str:
    """Copy the active Yahoo page through Chrome's normal user clipboard path."""

    page.bring_to_front()
    page.evaluate("() => window.focus()")
    page.keyboard.press("Control+A")
    page.wait_for_timeout(200)
    page.keyboard.press("Control+C")
    page.wait_for_timeout(400)
    return _windows_clipboard_text()


def _native_copy_yahoo_window() -> str:
    """Use opt-in Windows input to copy the visible Yahoo Chrome window.

    Yahoo's roster component can ignore browser-automation key events even
    though it responds to real keyboard shortcuts.  This function is kept
    separate and opt-in because it activates a desktop window and replaces the
    user's clipboard, just as a user pressing Ctrl+A/Ctrl+C would.
    """

    try:
        import pyautogui
        import pygetwindow as gw
    except ImportError as exc:
        raise RuntimeError(
            "Native Yahoo copy needs PyAutoGUI. Install it with: "
            "python -m pip install pyautogui pygetwindow"
        ) from exc

    windows = [
        window
        for window in gw.getAllWindows()
        if window.title
        and "yahoo" in window.title.lower()
        and "chrome" in window.title.lower()
    ]
    if len(windows) != 1:
        titles = [window.title for window in windows]
        raise RuntimeError(
            "Native Yahoo copy requires exactly one visible Yahoo Chrome "
            "window. Close other Yahoo Chrome windows, then retry. Found: "
            + repr(titles)
        )

    window = windows[0]
    if window.isMinimized:
        window.restore()
    window.activate()
    time.sleep(0.8)
    pyautogui.hotkey("ctrl", "a")
    time.sleep(0.2)
    pyautogui.hotkey("ctrl", "c")
    time.sleep(0.5)
    return _windows_clipboard_text()


def save_roster_capture(
    visible_text: str,
    roster_path: Path,
    export_directory: Path,
) -> tuple[Path, Path | None]:
    """Archive a capture and safely replace the working roster text file."""

    cleaned = _clean_visible_text(visible_text)
    if len(cleaned.strip()) < 80:
        raise RuntimeError(
            "Yahoo returned too little visible text to safely replace the roster."
        )
    if not _is_yahoo_roster_text(cleaned):
        raise RuntimeError(
            "Yahoo returned readable text, but it did not resemble a league "
            "roster export (expected multiple 'TEAM - POSITION' rows). "
            "The working roster was left unchanged."
        )

    export_directory.mkdir(parents=True, exist_ok=True)
    roster_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    captured_path = export_directory / f"yahoo_roster_{stamp}.txt"
    captured_path.write_text(cleaned, encoding="utf-8")

    backup_path = None
    if roster_path.exists():
        backup_path = export_directory / f"my_roster_before_yahoo_{stamp}.txt"
        shutil.copy2(roster_path, backup_path)
    roster_path.write_text(cleaned, encoding="utf-8")
    return captured_path, backup_path


def _visible_roster_text(pages) -> tuple[str, str]:
    """Choose the largest visible Yahoo content block across tabs and frames.

    Yahoo can place the roster table in a different tab or an embedded frame.
    Browser ``inner_text`` reads off-screen DOM text too, so a user does not
    need to scroll through a long roster table before capture.
    """

    candidates: list[tuple[str, str]] = []
    diagnostics: list[str] = []
    yahoo_pages = [
        page
        for page in pages
        if "football.fantasysports.yahoo.com" in page.url.lower()
    ]
    if not yahoo_pages:
        attached_pages = [page.url for page in pages]
        raise RuntimeError(
            "No Yahoo Fantasy Football tab is visible to the attached Chrome "
            "session. Capture was cancelled without changing the roster. "
            "Tabs visible through the attached Chrome session: "
            + repr(attached_pages)
        )

    for page in yahoo_pages:
        # A CDP-attached tab can briefly report a URL before its renderer is
        # ready. Bringing it forward and allowing one render turn avoids
        # treating that transient state as an empty Yahoo page.
        try:
            page.bring_to_front()
            page.wait_for_timeout(750)
        except Exception as exc:
            diagnostics.append(f"page activation failed ({type(exc).__name__})")

        for frame in page.frames:
            # Prefer direct DOM text first. It works for Yahoo layouts whose
            # roster component is nested in an open shadow tree, where a
            # conventional locator can return no usable visible text.
            try:
                text = frame.evaluate(
                    "() => document.body ? document.body.innerText : ''"
                )
                if text and _is_yahoo_roster_text(text):
                    source = f"{page.url} :: {frame.url} :: document.body.innerText"
                    candidates.append((text, source))
                elif text and text.strip():
                    diagnostics.append(
                        f"non-roster text from {frame.url or 'top'} "
                        f"({_roster_row_count(text)} roster rows)"
                    )
            except Exception as exc:
                diagnostics.append(
                    f"frame text failed ({frame.url or 'top'}; {type(exc).__name__}: {exc})"
                )

            for selector in ("main", "[role='main']", "#Main", "body"):
                try:
                    locator = frame.locator(selector)
                    if not locator.count():
                        continue
                    text = locator.first.inner_text(timeout=5_000)
                    if _is_yahoo_roster_text(text):
                        source = f"{page.url} :: {frame.url} :: {selector}"
                        candidates.append((text, source))
                    elif text.strip():
                        diagnostics.append(
                            f"non-roster {selector} from {frame.url or 'top'} "
                            f"({_roster_row_count(text)} roster rows)"
                        )
                except Exception as exc:
                    # A cross-origin or transient frame should not prevent
                    # checking the remaining Yahoo frames/pages.
                    diagnostics.append(
                        f"{selector} failed ({frame.url or 'top'}; {type(exc).__name__})"
                    )
                    continue

    if not candidates:
        attached_pages = [page.url for page in pages]
        detail = "; ".join(diagnostics[-6:]) or "no DOM text was exposed"
        raise RuntimeError(
            "No Yahoo roster-shaped content was found to capture. "
            "Tabs visible through the attached Chrome session: "
            + repr(attached_pages)
            + ". Diagnostic detail: "
            + detail
        )
    return max(candidates, key=lambda candidate: len(candidate[0].strip()))


def _open_yahoo_roster_tab(context, roster_url: str):
    """Open the requested roster in a dedicated tab, tolerating ad redirects."""

    errors: list[str] = []
    for attempt in range(1, 4):
        page = context.new_page()
        try:
            # ``commit`` avoids treating a late advertising frame as a reason
            # to abandon the legitimate top-level Yahoo response.
            page.goto(roster_url, wait_until="commit", timeout=60_000)
            page.wait_for_timeout(2_000)
            if "football.fantasysports.yahoo.com" in page.url.lower():
                return page
            errors.append(f"attempt {attempt} ended at {page.url}")
        except Exception as exc:
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
        try:
            page.close()
        except Exception:
            pass

    raise RuntimeError(
        "Yahoo roster navigation was diverted before the roster could load. "
        "Attempts: " + " | ".join(errors)
    )


def capture_yahoo_roster(
    roster_url: str,
    roster_path: Path,
    profile_directory: Path,
    export_directory: Path,
    attach_port: int | None = None,
    auto_copy: bool = False,
    native_copy: bool = False,
    navigate: bool = False,
    confirm: bool = True,
) -> tuple[Path, Path | None, str]:
    """Open Yahoo visibly, wait for user login, and capture roster text.

    The function is deliberately interactive. It never attempts password
    autofill, CAPTCHA bypass, two-factor bypass, or background scraping.
    """

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is not installed. Run: python -m pip install playwright"
        ) from exc

    attached_browser = attach_port is not None
    profile_directory.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        if attached_browser:
            browser = playwright.chromium.connect_over_cdp(
                f"http://127.0.0.1:{attach_port}"
            )
            if not browser.contexts:
                raise RuntimeError("No Chrome browsing context is available to attach.")
            context = browser.contexts[0]
            if navigate:
                page = _open_yahoo_roster_tab(context, roster_url)
            else:
                page = next(
                    (
                        candidate
                        for candidate in reversed(context.pages)
                        if "yahoo" in candidate.url.lower()
                    ),
                    context.pages[-1] if context.pages else context.new_page(),
                )
            print("\nAttached to your manually launched Chrome browser.")
        else:
            try:
                context = playwright.chromium.launch_persistent_context(
                    user_data_dir=str(profile_directory),
                    channel="chrome",
                    headless=False,
                    viewport={"width": 1440, "height": 1050},
                )
            except Exception as exc:
                raise RuntimeError(
                    "FANDROMEDA could not launch installed Google Chrome. "
                    "Install/update Chrome, then retry --refresh-yahoo-roster."
                ) from exc
            page = context.pages[0] if context.pages else context.new_page()
            page.goto(roster_url, wait_until="domcontentloaded", timeout=60_000)
            print("\nA visible Yahoo browser is open.")

        if confirm:
            print("Log in yourself, then confirm the requested league roster week is visible.")
            input("When the roster is fully visible, press Enter here to capture it: ")
        else:
            print("Waiting for the requested Yahoo roster week to render...")
            page.wait_for_timeout(3_000)
        try:
            visible_text, source = _visible_roster_text(context.pages)
        except RuntimeError as browser_error:
            # Yahoo's current roster page sometimes renders its table inside a
            # component whose text is available to a human copy operation but
            # not to CDP/Playwright DOM queries.  This fallback remains
            # user-directed: the user explicitly selects and copies only the
            # visible Yahoo roster before FANDROMEDA reads the clipboard.
            print("\nYahoo did not expose roster text to the browser connection.")
            if native_copy:
                print("Copying the Yahoo Chrome window with native Windows input...")
                visible_text = _native_copy_yahoo_window()
                source = "native Windows Ctrl+A/Ctrl+C clipboard copy"
            elif auto_copy:
                print("Copying the active Yahoo page through Chrome...")
                visible_text = _copy_active_yahoo_page(page)
                source = "automated Chrome Ctrl+A/Ctrl+C clipboard copy"
            else:
                input(
                    "Click inside the Yahoo roster table, press Ctrl+A then Ctrl+C, "
                    "and press Enter here to capture that copy: "
                )
                visible_text = _windows_clipboard_text()
                source = "user-confirmed Windows clipboard copy"
            if not _is_yahoo_roster_text(visible_text):
                raise RuntimeError(
                    str(browser_error)
                    + " Clipboard text also did not resemble a Yahoo roster; "
                    "the working roster was left unchanged."
                )
        if not attached_browser:
            context.close()

    captured_path, backup_path = save_roster_capture(
        visible_text,
        roster_path,
        export_directory,
    )
    return captured_path, backup_path, source
