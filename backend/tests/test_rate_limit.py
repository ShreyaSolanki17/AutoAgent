"""Rate limiter tests — sliding window logic and global config."""

import unittest

from backend.middleware.safety import RateLimiter, get_rate_limiter


class TestRateLimiter(unittest.TestCase):

    def setUp(self):
        self.limiter = RateLimiter(max_requests=5, window_seconds=10)

    def test_allows_under_limit(self):
        allowed, info = self.limiter.is_allowed("user1")
        self.assertTrue(allowed)
        self.assertEqual(info["remaining"], 4)

    def test_allows_multiple_under_limit(self):
        for i in range(4):
            allowed, info = self.limiter.is_allowed("user1")
            self.assertTrue(allowed)
        _, info = self.limiter.is_allowed("user1")
        self.assertEqual(info["remaining"], 0)

    def test_blocks_at_limit(self):
        for i in range(5):
            self.limiter.is_allowed("user1")
        allowed, info = self.limiter.is_allowed("user1")
        self.assertFalse(allowed)
        self.assertEqual(info["remaining"], 0)

    def test_separate_keys_independent(self):
        for i in range(5):
            self.limiter.is_allowed("user1")
        allowed, _ = self.limiter.is_allowed("user2")
        self.assertTrue(allowed)

    def test_rate_limit_info_has_required_keys(self):
        _, info = self.limiter.is_allowed("user1")
        self.assertIn("remaining", info)
        self.assertIn("reset_after", info)
        self.assertIn("limit", info)

    def test_reset_after_positive_when_blocked(self):
        for i in range(5):
            self.limiter.is_allowed("user1")
        _, info = self.limiter.is_allowed("user1")
        self.assertGreater(info["reset_after"], 0)

    def test_limit_matches_config(self):
        _, info = self.limiter.is_allowed("user1")
        self.assertEqual(info["limit"], 5)


class TestGlobalRateLimiter(unittest.TestCase):
    """Verify the global rate limiter has the expected config."""

    def test_global_limiter_30per60(self):
        rl = get_rate_limiter()
        self.assertEqual(rl.max_requests, 30)
        self.assertEqual(rl.window_seconds, 60)
