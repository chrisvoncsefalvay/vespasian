"""Exercise Space login with local OAuth fixtures in Chromium and WebKit.

Run from a checkout with the Space dependencies and Playwright installed:
    python -m playwright install chromium webkit
    python scripts/check_space_oauth.py

The test starts an isolated HTTPS app and a cross-site iframe. OpenSSL creates
a temporary test certificate. The OAuth provider and account catalogue are
fixtures; Gradio's routes, signed session cookies, UI and callbacks are real.
No user credentials, Hub requests, model inference or publication are used.
This is not a live Hugging Face token-exchange test.

VESPASIAN_TEST_APP_PYTHON can select an existing Space Python environment.
VESPASIAN_TEST_CHROMIUM_EXECUTABLE and VESPASIAN_TEST_WEBKIT_EXECUTABLE can
select browser installations matching the active Playwright version.
"""

import json
import os
from pathlib import Path
import re
import socket
import ssl
import subprocess
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import time
from urllib.parse import urlparse
import urllib.request

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]

LAUNCHER = r"""
import os
import sys
import time
from functools import wraps
from unittest.mock import patch

os.environ.update(
    VESPASIAN_SPACE_SKIP_MODEL_LOAD="1", GRADIO_ANALYTICS_ENABLED="False",
    HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", SYSTEM="spaces",
    SPACE_ID="test/vespasian", OAUTH_CLIENT_ID="test-client",
    OAUTH_CLIENT_SECRET="test-placeholder",
    OAUTH_SCOPES="openid profile read-repos gated-repos write-repos",
    OPENID_PROVIDER_URL="https://huggingface.co",
)
os.environ.pop("SPACE_HOST", None)
sys.path.insert(0, "spaces/zerogpu")

from fastapi.responses import HTMLResponse, RedirectResponse
from authlib.integrations.base_client.errors import MismatchingStateError
import gradio as gr

class Provider:
    async def authorize_redirect(self, request, redirect_uri):
        request.session["fixture_state"] = "test"
        return RedirectResponse(str(redirect_uri))

    async def authorize_access_token(self, request):
        if request.cookies.get("fixture_mismatch") == "1" or request.session.pop("fixture_state", None) != "test":
            raise MismatchingStateError()
        return {
            "access_token": "fixture-only-not-a-credential",
            "scope": "openid profile",
            "expires_at": int(time.time()) + 3600,
            "userinfo": {
                "name": "Fixture User", "preferred_username": "fixture-user" + ("-long" * 10 if request.cookies.get("fixture_long_username") == "1" else ""),
                "profile": "https://huggingface.co/fixture-user", "picture": "",
            },
        }

class OAuthFixture:
    huggingface = Provider()

    def register(self, **kwargs):
        pass

# The guide opens before this result arrives, reproducing the label race.
original_check = gr.LoginButton._check_login_status
@wraps(original_check)
def delayed_check(self, request):
    time.sleep(0.75)
    return original_check(self, request)
gr.LoginButton._check_login_status = delayed_check

with patch("authlib.integrations.starlette_client.OAuth", OAuthFixture):
    import app
    app.hub.account_datasets = lambda token: ("fixture-user", []) if token else (None, [])
    app.gr.set_static_paths(paths=[app.ASSET_DIR])
    demo = app.demo
    demo.queue(max_size=16, default_concurrency_limit=4).launch(
        app_kwargs=app.oauth_guard.app_kwargs(),
        css=app.CSS, theme=app.THEME, js=app.JS, show_error=True,
        server_name="127.0.0.1", server_port=int(sys.argv[1]),
        allowed_paths=[str(app.ASSET_DIR)], ssl_certfile=sys.argv[2],
        ssl_keyfile=sys.argv[3], ssl_verify=False, prevent_thread_lock=True,
    )
    @demo.app.get("/test-oauth-iframe")
    async def authenticated_iframe():
        return HTMLResponse(
            '<iframe src="/" sandbox="allow-scripts allow-same-origin allow-forms '
            'allow-popups allow-popups-to-escape-sandbox"></iframe>'
        )
    while True:
        time.sleep(1)
"""


def unused_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def blocked_functions(config):
    publication_buttons = {
        component["id"] for component in config["components"]
        if component.get("props", {}).get("value") in (
            "Save edited model to Hub", "Save trajectory to Hub",
            "Save latest/best patches privately",
        )
    }
    return {
        dependency["id"] for dependency in config["dependencies"]
        if dependency.get("api_name") in ("run", "advance")
        or any(target[0] in publication_buttons for target in dependency.get("targets", []))
    }


def check_header(page, engine, signed_in, username):
    """Check the real Gradio controls, without contacting the OAuth provider."""
    page.locator(".vs-guide-top [data-guide-close]").click()
    login = page.locator("#vs-login")
    expect(login).to_have_attribute("aria-label", f"Sign out of Hugging Face ({username})" if signed_in else "Sign in with Hugging Face")
    for width in (320, 390, 1280):
        page.set_viewport_size({"width": width, "height": 900})
        theme = page.locator(".vs-header-links [data-vs-theme-toggle]")
        help_control = page.locator(".vs-header-links [data-vs-help]")
        expect(help_control).to_have_attribute("aria-label", "Quick guide")
        for control in (theme, help_control, login):
            box = control.bounding_box()
            assert box and box["width"] >= 44 and 44 <= box["height"] <= 48, box
            assert box["x"] >= 0 and box["x"] + box["width"] <= width, box
            control.focus()
            expect(control).to_be_focused()
        assert theme.inner_text() == ""
        assert help_control.inner_text() == ""
        original = page.locator("html").get_attribute("data-vs-theme")
        theme.press("Enter")
        expect(page.locator("html")).to_have_attribute("data-vs-theme", "light" if original == "dark" else "dark")
        assert page.evaluate("localStorage.getItem('vespasian.theme')") != original
        page.screenshot(animations="disabled", path=f"/tmp/vespasian-header-{engine}-{'signed-in' if signed_in else 'anonymous'}-{width}-alternate.png")
        theme.press("Enter")
        help_control.press("Enter")
        expect(page.locator("#vs-guide[open]")).to_be_visible()
        page.keyboard.press("Escape")
        expect(help_control).to_be_focused()
        expect(page.locator("#vs-guide")).not_to_be_visible()
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
        page.screenshot(animations="disabled", path=f"/tmp/vespasian-header-{engine}-{'signed-in' if signed_in else 'anonymous'}-{width}.png")
    if not signed_in:
        page.locator('#vs-setup-tabs button[data-tab-id="editor"]').click()
        effort = page.locator("#vs-reasoning-effort")
        expect(effort).to_be_visible()
        expect(effort.locator("input")).to_have_value("low")
        effort.locator("input").click()
        effort.get_by_role("option", name="high", exact=True).click()
        expect(effort.locator("input")).to_have_value("high")
        page.locator('#vs-setup-tabs button[data-tab-id="target"]').click()
    page.set_viewport_size({"width": 390, "height": 844})
    page.locator(".vs-header-links [data-vs-help]").click()


def exercise_browsers(origin, wrapper, config):
    receipts, forbidden = [], []
    forbidden_functions = blocked_functions(config)
    with sync_playwright() as playwright:
        for engine in ("chromium", "webkit"):
            options = {"headless": True}
            if executable := os.environ.get(f"VESPASIAN_TEST_{engine.upper()}_EXECUTABLE"):
                options["executable_path"] = executable
            browser = getattr(playwright, engine).launch(**options)
            try:
                cases = (
                    (False, False, False, False), (False, True, False, False),
                    (True, False, False, False), (True, True, False, False),
                    (True, True, True, False), (True, True, False, True),
                )
                for embedded, guide, interrupted, blocked_popup in cases:
                    # HTTPS exceptions apply only to this disposable local test context.
                    context = browser.new_context(
                        viewport={"width": 390, "height": 844}, ignore_https_errors=True,
                    )
                    username = "fixture-user"
                    if not embedded and not guide:
                        username += "-long" * 10
                        context.add_cookies([{"name": "fixture_long_username", "value": "1", "url": origin}])
                    errors = []

                    def guard(route):
                        request = route.request
                        if request.url == "https://huggingface.co/api/spaces/test/vespasian":
                            route.fulfill(json={
                                "id": "test/vespasian", "author": "test", "sha": "fixture",
                                "cardData": {"title": "Vespasian"}, "likes": 0,
                            })
                            return
                        if urlparse(request.url).hostname not in {"127.0.0.1", "localhost"}:
                            route.abort()
                            return
                        if re.search(r"/gradio_api/(queue/join|run/|call/)", request.url):
                            try:
                                body = request.post_data_json or {}
                            except Exception:
                                body = {}
                            if body.get("fn_index") in forbidden_functions:
                                forbidden.append(request.url)
                                route.abort()
                                return
                        route.continue_()

                    context.route("**/*", guard)
                    page = context.new_page()
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    try:
                        page.goto(wrapper if embedded else origin, wait_until="domcontentloaded")
                        frame = page.frame_locator("iframe") if embedded else page
                        expect(frame.locator("#vs-guide[open]")).to_be_visible(timeout=30000)
                        if not embedded and not guide:
                            check_header(page, engine, False, username)
                        def start_login():
                            if guide:
                                frame.locator("[data-guide-login]").click()
                            else:
                                frame.locator(".vs-guide-top [data-guide-close]").click()
                                frame.locator("#vs-login").click()
                        if embedded and blocked_popup:
                            frame.locator("html").evaluate("() => { window.open = () => null }")
                            start_login()
                            expect(frame.locator("#vs-login-help")).to_contain_text("standalone tab")
                            with context.expect_page() as popup:
                                frame.locator("#vs-login-help a").click()
                            page = popup.value
                            page.on("pageerror", lambda error: errors.append(str(error)))
                            expect(page.locator("#vs-guide[open]")).to_be_visible()
                            page.locator(".vs-guide-top [data-guide-close]").click()
                            page.locator("#vs-login").click()
                        elif embedded:
                            if interrupted:
                                context.route(
                                    origin + "/login/huggingface?*",
                                    lambda route: route.fulfill(
                                        content_type="text/html", body="<h1>Sign-in paused</h1>",
                                    ),
                                    times=1,
                                )
                            with context.expect_page() as popup:
                                start_login()
                            expect(frame.locator("#vs-login-help")).to_contain_text("standalone tab")
                            # A rapid repeated tap must not start a second login tab.
                            pages_before = len(context.pages)
                            frame.locator("#vs-login").evaluate("el => el.click()")
                            page.wait_for_timeout(200)
                            assert len(context.pages) == pages_before
                            assert page.url == wrapper + "/"
                            if interrupted:
                                expect(popup.value.locator("h1")).to_have_text("Sign-in paused")
                                popup.value.close()
                                expect(frame.locator("#vs-login")).to_have_text("Not logged in")
                                # The deliberate retry follows the double-tap pause.
                                page.wait_for_timeout(1500)
                                with context.expect_page() as popup:
                                    frame.locator("#vs-login").click()
                            page = popup.value
                            page.on("pageerror", lambda error: errors.append(str(error)))
                        else:
                            start_login()
                        expect(page.locator("#vs-login")).to_contain_text(
                            username, timeout=30000,
                        )
                        expect(page.locator("#vs-profile .vs-profile")).to_have_text("")
                        assert page.url.startswith(origin), page.url
                        if guide and not interrupted and not blocked_popup:
                            expect(page.locator('[data-guide-tab="1"]')).to_have_attribute(
                                "aria-selected", "true",
                            )
                            assert "__vs_guide_resume" not in urlparse(page.url).query

                        # A fresh welcome must update when the delayed native status arrives.
                        page.evaluate("localStorage.removeItem('vespasian.guide.v2')")
                        page.reload(wait_until="domcontentloaded")
                        expect(page.locator("#vs-guide[open]")).to_be_visible()
                        expect(page.locator("#vs-login")).to_have_text(username)
                        expect(page.locator("[data-guide-login]")).to_have_text("Continue signed in")
                        if not embedded and not guide:
                            check_header(page, engine, True, username)
                        page.locator("[data-guide-login]").click()
                        expect(page.locator('[data-guide-tab="1"]')).to_have_attribute(
                            "aria-selected", "true",
                        )
                        expect(page.locator("#vs-login")).to_have_text(username)
                        page.locator(".vs-guide-top [data-guide-close]").click()
                        if not embedded and not guide:
                            # Same-origin fixture ensures a signed-in iframe in both
                            # engines, including WebKit's strict cookie mode.
                            page.goto(origin + "/test-oauth-iframe")
                            signed_frame = page.frame_locator("iframe")
                            expect(signed_frame.locator("#vs-login")).to_contain_text(
                                username, timeout=30000,
                            )
                            if signed_frame.locator("#vs-guide[open]").count():
                                signed_frame.locator(".vs-guide-top [data-guide-close]").click()
                            pages_before = len(context.pages)
                            signed_frame.locator("#vs-login").click()
                            expect(signed_frame.locator("#vs-login")).to_have_text(
                                "Not logged in", timeout=30000,
                            )
                            assert len(context.pages) == pages_before
                            page.goto(origin)
                        else:
                            page.locator("#vs-login").click()
                        expect(page.locator("#vs-login")).to_have_text(
                            "Not logged in", timeout=30000,
                        )
                        if not embedded and not guide:
                            await_starts = []
                            context.on("request", lambda request: await_starts.append(request.url)
                                       if urlparse(request.url).path == "/login/huggingface" else None)
                            context.add_cookies([{
                                "name": "fixture_mismatch", "value": "1", "url": origin,
                                "secure": True, "sameSite": "None",
                            }])
                            page.goto(origin + "/login/huggingface")
                            expect(page.locator("h1")).to_have_text("Sign-in could not finish")
                            assert len(await_starts) == 3  # third request is stopped before OAuth
                            page.reload(wait_until="domcontentloaded")
                            expect(page.locator("h1")).to_have_text("Sign-in could not finish")
                            assert len(await_starts) == 4
                            # Dismissal/new-tab recovery never automatically starts OAuth.
                            with context.expect_page() as recovered:
                                page.get_by_role("link", name="Open standalone workbench").click()
                            recovery = recovered.value
                            expect(recovery.locator("#vs-login")).to_have_text("Not logged in")
                            assert len(await_starts) == 4
                            page.go_back(wait_until="domcontentloaded")
                            assert len(await_starts) == 4
                            context.clear_cookies(name="fixture_mismatch")
                            if recovery.locator("#vs-guide[open]").count():
                                recovery.locator(".vs-guide-top [data-guide-close]").click()
                            recovery.locator("#vs-login").click()
                            expect(recovery.locator("#vs-login")).to_have_text(username)
                            recovery.close()
                        assert not errors, errors
                        assert not forbidden, forbidden
                        receipts.append({
                            "engine": engine, "embedded": embedded,
                            "entry": "guide" if guide else "account",
                            "cancelled_popup_retry": "passed" if interrupted else "not applicable",
                            "blocked_popup_fallback": "passed" if blocked_popup else "not applicable",
                            "rapid_repeat_suppressed": "passed" if embedded and not blocked_popup else "not applicable",
                            "state_failure_recovery": "passed" if not embedded and not guide else "not applicable",
                            "login": "passed", "connected_profile": "passed",
                            "delayed_guide_status": "passed",
                            "continue_without_logout": "passed", "logout": "passed",
                        })
                    finally:
                        context.close()
            finally:
                browser.close()
    return receipts


def main():
    python = os.environ.get("VESPASIAN_TEST_APP_PYTHON", sys.executable)
    port = unused_port()
    origin = f"https://127.0.0.1:{port}"

    class WrapperHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = (
                f'<iframe src="{origin}" style="width:100%;height:800px" '
                'sandbox="allow-downloads allow-forms allow-modals allow-pointer-lock '
                'allow-popups allow-popups-to-escape-sandbox allow-same-origin '
                'allow-scripts allow-storage-access-by-user-activation"></iframe>'
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    wrapper_server = ThreadingHTTPServer(("127.0.0.1", 0), WrapperHandler)
    wrapper = f"http://localhost:{wrapper_server.server_port}"
    Thread(target=wrapper_server.serve_forever, daemon=True).start()
    try:
        with tempfile.TemporaryDirectory(prefix="vespasian-oauth-") as directory:
            cert, key = Path(directory) / "cert.pem", Path(directory) / "key.pem"
            subprocess.run([
                "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                "-keyout", str(key), "-out", str(cert), "-days", "1",
                "-subj", "/CN=127.0.0.1", "-addext", "subjectAltName=IP:127.0.0.1",
            ], check=True, capture_output=True)
            log_path = Path(directory) / "server.log"
            with log_path.open("w") as log:
                server = subprocess.Popen(
                    [python, "-c", LAUNCHER, str(port), str(cert), str(key)],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                )
                try:
                    tls = ssl.create_default_context(cafile=str(cert))
                    for _ in range(90):
                        try:
                            with urllib.request.urlopen(origin + "/config", timeout=1, context=tls) as response:
                                config = json.load(response)
                            break
                        except Exception:
                            if server.poll() is not None:
                                raise RuntimeError("OAuth fixture failed:\n" + log_path.read_text()[-5000:])
                            time.sleep(0.5)
                    else:
                        raise RuntimeError("OAuth fixture did not start")
                    print(json.dumps(exercise_browsers(origin, wrapper, config), indent=2))
                finally:
                    server.terminate()
                    try:
                        server.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait()
    finally:
        wrapper_server.shutdown()
        wrapper_server.server_close()


if __name__ == "__main__":
    main()

