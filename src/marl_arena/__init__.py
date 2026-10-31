"""The arena engine, shared by both experiment trees.

There is one copy of this package. ``ctde_arena/src/marl_arena/__init__.py`` is a shim that points its
import path here and names ``ctde_arena/`` as the project directory, so experiment 2 reads its own
``.env``, ``data/`` and ``requirements.txt`` while running exactly this code (audit item A-11).
"""

from pathlib import Path

# The experiment tree whose .env, data/ and requirements.txt this process reads. Importing the package
# directly selects the repository root; the experiment-2 shim defines its own value instead.
PROJECT_DIR = Path(__file__).resolve().parents[2]

__all__ = []
