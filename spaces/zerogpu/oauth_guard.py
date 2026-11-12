"""Stop Gradio state-mismatch recovery from repeatedly starting new logins.

Gradio retries a failed state check and eventually redirects to the standalone
Space host. That fallback is still the same host for first-party login, so a
persistent cookie failure otherwise keeps restarting OAuth indefinitely.
"""

from urllib.parse import parse_qs

from starlette.middleware import Middleware
from starlette.responses import HTMLResponse


class OAuthRetryGuard:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"] == "/login/huggingface":
            query = parse_qs(scope.get("query_string", b"").decode("latin-1"), keep_blank_values=True)
            attempts = query.get("_nb_redirects", ["0"])
            # Invalid or ambiguous counters cannot be allowed to restart login.
            try:
                retries = int(attempts[0]) if len(attempts) == 1 else -1
            except ValueError:
                retries = -1
            if retries < 0 or retries >= 2:
                response = HTMLResponse(
                    "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
                    "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
                    "<title>Sign-in could not finish</title>"
                    "<h1>Sign-in could not finish</h1>"
                    "<p>The sign-in session could not be verified. "
                    "Automatic retries have stopped.</p>"
                    "<p>Close other sign-in tabs and open the standalone workbench. "
                    "If Hugging Face reports too many attempts, wait before trying again.</p>"
                    "<p><a href=\"/\" target=\"_blank\" rel=\"noopener noreferrer\">"
                    "Open standalone workbench</a></p></html>",
                    status_code=400,
                    headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"},
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def app_kwargs():
    return {"middleware": [Middleware(OAuthRetryGuard)]}
