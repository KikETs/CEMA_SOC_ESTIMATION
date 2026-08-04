from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LeakageGuard:
    fold: str
    test_profile: str
    test_files: set[Path]
    events: list[dict] = field(default_factory=list)

    def record_fit(self, stage: str, paths: list[Path]) -> None:
        resolved = {Path(path).resolve() for path in paths}
        overlap = resolved & self.test_files
        profile_hits = [path for path in resolved if self.test_profile.upper() in path.stem.upper()]
        passed = not overlap and not profile_hits
        event = {
            "stage": stage,
            "files": [str(path) for path in sorted(resolved)],
            "test_overlap": [str(path) for path in sorted(overlap)],
            "test_profile_name_hits": [str(path) for path in sorted(profile_hits)],
            "passed": passed,
        }
        self.events.append(event)
        if not passed:
            raise RuntimeError(f"Held-out leakage in fold={self.fold}, stage={stage}: {event}")

    def to_dict(self) -> dict:
        return {
            "fold": self.fold,
            "test_profile": self.test_profile,
            "test_files": [str(path) for path in sorted(self.test_files)],
            "fit_events": self.events,
            "leakage_count": sum(not event["passed"] for event in self.events),
            "status": "PASS" if all(event["passed"] for event in self.events) else "FAIL",
        }
