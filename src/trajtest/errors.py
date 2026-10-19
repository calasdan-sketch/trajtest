"""Errors raised by trajtest."""


class EvalFailure(AssertionError):
    """Raised by Expect.assert_all() when one or more assertions fail.

    Carries the full fail-collected report: every assertion outcome plus a
    trajectory transcript, so eval authors debug in one pass.
    """

    def __init__(self, report: str):
        super().__init__(report)
        self.report = report


class MaxStepsExceeded(Exception):
    """Raised inside run() when the agent exceeds its step budget."""


class UnmockedRequestError(Exception):
    """Raised by FakeHTTP when no mock matches a request (strict mode)."""
