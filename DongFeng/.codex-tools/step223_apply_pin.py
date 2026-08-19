"""Minimal DataContainer.pin_memory for STEP-223 candidate.

Applied only during candidate A/B; restore from backup after.
"""

from __future__ import annotations

from pathlib import Path

MARKER = "    def pin_memory(self):"
SNIPPET = '''
    def pin_memory(self):
        """Pin CPU tensor payload so DataLoader pin_memory=True takes effect."""
        if self.cpu_only:
            return self

        def _pin(obj):
            if isinstance(obj, torch.Tensor):
                return obj.pin_memory()
            if isinstance(obj, list):
                return [_pin(x) for x in obj]
            if isinstance(obj, tuple):
                return tuple(_pin(x) for x in obj)
            return obj

        self._data = _pin(self._data)
        return self
'''

def apply(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        raise RuntimeError("pin_memory already present")
    if "class DataContainer:" not in text:
        raise RuntimeError("DataContainer class missing")
    path.write_text(text.rstrip() + "\n" + SNIPPET + "\n", encoding="utf-8")


def main() -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("target", type=Path)
    args = p.parse_args()
    apply(args.target)
    print("applied", args.target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
