"""anemic-blood.

OpenMP note -- do not reorder or remove anything in this file.

torch and LightGBM each ship their own copy of the OpenMP runtime
(`libomp.dylib`). On macOS/arm64 the two copies fight, and the process dies with
**SIGSEGV (exit 139) and no traceback**. It is nondeterministic and surfaces in
whichever library happens to spawn a thread pool first, so the same bug appears
as a crash in `LGBMClassifier.fit`, or inside a torchvision v2 transform, or
during imports. It looks like a data bug. It is not.

Two lines are needed, and both must run before torch or lightgbm load:

1. `OMP_NUM_THREADS=1` stops the duplicate runtimes contending. Measured cost on
   this project: none -- 50 img/s single-threaded vs 47 img/s default, because
   the real work sits on MPS and the threads were only adding contention.
2. Importing lightgbm ahead of torch. Necessary but NOT sufficient on its own:
   it protects `LGBMClassifier.fit` and still leaves torchvision transforms
   crashing, which is how this was originally misdiagnosed as fixed.

`KMP_DUPLICATE_LIB_OK=TRUE` is the usual internet advice. Avoid it -- it works by
letting two OpenMP runtimes coexist and is documented as capable of producing
silently wrong results.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import lightgbm as _lightgbm  # noqa: E402,F401  (must precede torch -- see above)
