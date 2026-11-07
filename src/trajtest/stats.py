"""Pass-rate statistics for stochastic agents (pairs with sample())."""

from dataclasses import dataclass, field

from trajtest.errors import EvalFailure


@dataclass
class PassRate:
    passed: int
    total: int
    failures: list = field(default_factory=list)  # [(index, report)]

    @property
    def rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def __str__(self) -> str:
        return f"{self.passed}/{self.total} passed ({self.rate:.0%})"


def pass_rate(trajs, build) -> PassRate:
    """Evaluate `build(traj) -> Expect` on every trajectory; count passes.

    An Expect with no assertions counts as a failure, same as assert_all().
    """
    trajs = list(trajs)
    failures = []
    for i, t in enumerate(trajs):
        violations = build(t).check()
        bad = [v for v in violations if not v.ok]
        if bad:
            failures.append((i, "\n".join(v.line() for v in bad)))
    return PassRate(len(trajs) - len(failures), len(trajs), failures)


def assert_pass_rate(trajs, build, min_rate: float = 1.0) -> PassRate:
    """Raise EvalFailure unless at least `min_rate` of runs pass.

    The default of 1.0 means every sampled run must pass: for safety
    properties (never_called a destructive tool) anything lower is a bug,
    so you must opt in to a lower bar explicitly.
    """
    if not 0.0 <= min_rate <= 1.0:
        raise ValueError("min_rate must be within [0, 1]")
    result = pass_rate(trajs, build)
    if result.total == 0:
        raise EvalFailure("trajtest: assert_pass_rate got zero trajectories")
    if result.rate + 1e-12 < min_rate:
        first = result.failures[0]
        raise EvalFailure(
            f"trajtest pass rate {result} is below required {min_rate:.0%}\n"
            f"failing runs: {[i for i, _ in result.failures]}\n"
            f"first failure (run {first[0]}):\n{first[1]}"
        )
    return result
