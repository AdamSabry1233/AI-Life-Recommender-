# Test Suite

Layout:

```
tests/
├── conftest.py              # shared fixtures + sys.path injection for src/
├── unit/                    # fast, isolated (no network, no real model load)
│   ├── agents/              # mirrors src/agents/
│   ├── test_engine.py
│   └── test_recommender.py
├── integration/             # multi-module, mocked LLM + small fixture model
├── baselines/               # Cornac / LightFM baseline model tests
└── fixtures/                # tiny CSVs + checkpoint stubs (to be added)
```

## Running

All routing goes through the central delegator at `tools/test_runner.py`. Run it
with no argument for everything, or pass a subset name:

```bash
python tools/test_runner.py              # all
python tools/test_runner.py fast         # excludes @pytest.mark.slow
python tools/test_runner.py agents       # just src/agents/ coverage
python tools/test_runner.py habit        # just the habit agent
python tools/test_runner.py engine
python tools/test_runner.py recommender
python tools/test_runner.py integration
python tools/test_runner.py baselines

python tools/test_runner.py --watch              # filesystem auto-rerun
python tools/test_runner.py --changed            # only tests affected by git diff vs main
python tools/test_runner.py agents --watch       # scope + watch combined

python tools/test_runner.py --list               # show all subset names
python tools/test_runner.py --markers            # show pytest markers
```

## Markers

| Marker | Meaning |
|---|---|
| `unit` | Fast, isolated, no external deps |
| `integration` | Multi-module, mocked LLM |
| `slow` | Excluded from the `fast` subset |
| `agents` | Anything in `src/agents/` |
| `engine` | ChatEngine / classification routing |
| `recommender` | MF inference |
| `memory` | FAISS / embedder |
| `baselines` | Cornac / LightFM / other library baselines |

## Writing new tests

- Put the test file next to an existing one in the matching subdirectory.
- Add the appropriate markers at module scope:
  ```python
  import pytest
  pytestmark = [pytest.mark.unit, pytest.mark.agents]
  ```
- Use fixtures from `conftest.py` rather than building ad-hoc mocks.
- Never hit a real LLM or Redis from tests; mock `ChatOllama`/`ChatGroq` at the boundary.
