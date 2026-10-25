# %%
import os

import pytest
# %%
torch = pytest.importorskip("torch")
torch.cuda.is_available()
# %%
