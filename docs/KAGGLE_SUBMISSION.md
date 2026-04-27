# Kaggle Submission Path

BirdCLEF 2026 accepts notebook/code submissions, not direct CSV/script uploads.

Known working direction:

```python
import os
from kaggle.api.kaggle_api_extended import KaggleApi

os.environ["KAGGLE_API_TOKEN"] = "<KGAT token from local credentials/vault>"
api = KaggleApi()
api.authenticate()

push_result = api.kernels_push("/path/to/kernel-folder")
# inspect push_result/version, then wait for the kernel to complete

api.competition_submit_code(
    file_name="submission.csv",
    message="submission description",
    competition="birdclef-2026",
    kernel="yourslewis/kernel-slug",
    kernel_version=<completed_version>,
)
```

Notes:

- Kaggle API `1.6.17` authenticated via legacy basic auth failed with `401 Unauthorized` for the current `KGAT_*` token.
- Kaggle API `2.1.0` authenticated successfully with `KAGGLE_API_TOKEN` and could list BirdCLEF submissions.
- A real submission still requires a runnable kernel that produces `submission.csv`.
