"""Exercise real Gradio state validation without any provider requests."""
import os
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.responses import RedirectResponse, Response
from fastapi.testclient import TestClient
from authlib.integrations.base_client.errors import MismatchingStateError

import oauth_guard


class Provider:
    starts = 0

    async def authorize_redirect(self, request, redirect_uri):
        self.starts += 1
        return RedirectResponse(str(redirect_uri))

    async def authorize_access_token(self, request):
        raise MismatchingStateError()


class OAuthRetryGuardTests(unittest.TestCase):
    def client(self, guarded=True):
        import gradio.oauth as oauth
        provider = Provider()
        registry = type("Registry", (), {"huggingface": provider, "register": lambda *a, **k: None})
        app = FastAPI(**(oauth_guard.app_kwargs() if guarded else {}))
        with patch.dict(os.environ, {"SPACE_HOST": "testserver"}), \
             patch.object(oauth, "OAUTH_CLIENT_ID", "fixture"), \
             patch.object(oauth, "OAUTH_CLIENT_SECRET", "fixture"), \
             patch.object(oauth, "OAUTH_SCOPES", "openid profile"), \
             patch.object(oauth, "OPENID_PROVIDER_URL", "https://provider.invalid"), \
             patch("gradio.oauth.get_space", return_value="test/space"), \
             patch("authlib.integrations.starlette_client.OAuth", registry):
            oauth.attach_oauth(app)
        return TestClient(app, base_url="https://testserver", follow_redirects=False), provider

    def test_real_state_mismatch_stops_and_fresh_retry_is_possible(self):
        client, provider = self.client()
        with patch.dict(os.environ, {"SPACE_HOST": "testserver"}):
            response = client.get("/login/huggingface")
            for _ in range(8):
                if not response.is_redirect:
                    break
                response = client.get(response.headers["location"])
            self.assertEqual(response.status_code, 400)
            self.assertEqual(provider.starts, 2)
            self.assertNotIn("location", response.headers)
            self.assertIn("Automatic retries have stopped", response.text)
            self.assertEqual(response.headers["cache-control"], "no-store")
            # Back to the stopped URL cannot silently start another flow.
            self.assertEqual(client.get(response.url).status_code, 400)
            self.assertEqual(provider.starts, 2)
            self.assertEqual(client.get("/login/huggingface").status_code, 307)
            self.assertEqual(provider.starts, 3)

    def test_unguarded_gradio_reproduces_unbounded_state_mismatch(self):
        client, provider = self.client(guarded=False)
        with patch.dict(os.environ, {"SPACE_HOST": "testserver"}):
            response = client.get("/login/huggingface")
            for _ in range(14):
                self.assertTrue(response.is_redirect)
                response = client.get(response.headers["location"])
            self.assertTrue(response.is_redirect)
            self.assertGreater(provider.starts, 5)

    def test_invalid_and_excessive_retry_values_stop_before_provider(self):
        client, provider = self.client()
        for query in ("", "-1", "2", "99", "oops", "1&_nb_redirects=0", "1&_nb_redirects="):
            with self.subTest(query=query):
                response = client.get("/login/huggingface?_nb_redirects=" + query)
                self.assertEqual(response.status_code, 400)
        self.assertEqual(provider.starts, 0)

    def test_error_page_does_not_reflect_callback_parameters(self):
        client, provider = self.client()
        response = client.get("/login/huggingface", params={
            "_nb_redirects": "99", "_target_url": "//evil.invalid",
            "code": "private-code", "state": "private-state",
            "error_description": "<script>alert(1)</script>",
        })
        self.assertEqual(response.status_code, 400)
        for private in ("evil.invalid", "private-code", "private-state", "<script>"):
            self.assertNotIn(private, response.text)
        self.assertNotIn("location", response.headers)
        self.assertEqual(provider.starts, 0)

    def test_successful_callback_is_not_blocked_by_retry_counter(self):
        client, provider = self.client()
        provider.authorize_access_token = AsyncMock(return_value={
            "access_token": "fixture-only", "userinfo": {"name": "Fixture"},
        })
        response = client.get("/login/callback?_nb_redirects=99&_target_url=/")
        self.assertEqual(response.status_code, 307)
        self.assertEqual(response.headers["location"], "/")
        self.assertIn("session=", response.headers["set-cookie"])
        provider.authorize_access_token.assert_awaited_once()
        self.assertEqual(provider.starts, 0)

    def test_provider_rate_limit_is_not_retried_or_rewritten(self):
        client, provider = self.client()
        provider.authorize_redirect = AsyncMock(return_value=Response(
            "Too many attempts", status_code=429, headers={"Retry-After": "300"},
        ))
        response = client.get("/login/huggingface")
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["retry-after"], "300")
        self.assertEqual(response.text, "Too many attempts")
        provider.authorize_redirect.assert_awaited_once()

    def test_other_paths_are_unchanged(self):
        client, provider = self.client()
        self.assertEqual(client.get("/missing?_nb_redirects=99").status_code, 404)
        self.assertEqual(provider.starts, 0)


if __name__ == "__main__":
    unittest.main()
