# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
"""Convert a DeepReview HTML report into canonical Issue records.

DeepReview emits a free-form HTML-ish fragment (finding cards, a verdict
banner, and a prose preamble) whose exact shape varies run to run. Rather than
parse that brittly, an auxiliary LLM call rewrites it into the same normalized
text shape that AiCodeReview produces — a ``> ``-quoted code block followed by a
prose paragraph per finding — which the existing :func:`parse_issues` already
understands. This keeps the embedded/judged issue text identical in form to the
AiCodeReview eval path.
"""

from __future__ import annotations

import hashlib
import logging
import warnings
from pathlib import Path

from .parse import Issue, parse_issues

logger = logging.getLogger(__name__)

_CONVERT_PROMPT = """\
You are normalizing a Linux kernel patch code-review report into a strict,
machine-parseable issue list. The report below was produced by an AI reviewer
as an HTML fragment (finding cards, a verdict banner, and some prose preamble).

Extract every DISTINCT actionable finding (severity [BUG], [CONCERN], [MINOR],
or [NIT]). The report repeats findings between the per-commit cards and the
summary/verdict banner — emit each finding exactly ONCE.

For each finding, output, in order:
  1. A quoted code block: the specific code or diff snippet the finding is
     about, with EVERY line prefixed by "> ". Use the snippet shown in the
     finding (e.g. a <pre> block or quoted code) if present. If the finding
     shows no snippet, quote the cited location and construct instead, e.g.:
         > File: drivers/spi/spi-geni-qcom.c, line ~1067
         > mas->dev_data->resources_init(&mas->se);
  2. A single blank line.
  3. A prose paragraph describing the defect: start with the severity tag and a
     short title, then explain the problem. Do NOT include validation-gate
     trace text such as "(Gate 1: ...)" or step-completion records.
  4. A blank line separating it from the next finding.

Rules:
- Do not invent findings or details not present in the report.
- Quote-block lines MUST start with "> ". Prose lines MUST NOT start with "> ".
- Output ONLY the normalized findings. No preamble, no closing remarks.
- If the report contains no actionable findings, output exactly: No issues found.

Report to normalize:

<report>
{report}
</report>
"""


def convert_deepreview_to_issues(
    review_file: Path | None,
    *,
    model: str,
    api_base: str | None,
    cache_dir: Path,
) -> list[Issue]:
    """Convert a deepreview.txt into Issue records via an auxiliary LLM call.

    Returns an empty list when *review_file* is ``None`` or missing (DeepReview
    found no issues). The normalized text is written next to *review_file* as
    ``converted.txt`` for debuggability, and the LLM response is cached by
    content hash so re-runs are free.
    """
    if review_file is None or not review_file.exists():
        return []

    report = review_file.read_text(encoding="utf-8")
    if not report.strip():
        return []

    normalized = _normalize(report, model=model, api_base=api_base, cache_dir=cache_dir)

    converted_file = review_file.with_name("converted.txt")
    converted_file.write_text(normalized, encoding="utf-8")

    return parse_issues(converted_file)


def _normalize(report: str, *, model: str, api_base: str | None, cache_dir: Path) -> str:
    prompt = _CONVERT_PROMPT.format(report=report)

    cache_path = _cache_path(prompt, model, cache_dir)
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    import httpx
    import litellm

    from patchwise.utils.decorators import retry

    warnings.filterwarnings("ignore", message="Unverified HTTPS request")
    litellm.client_session = httpx.Client(verify=False)

    @retry(
        max_retries=10,
        exceptions=(
            litellm.Timeout,
            litellm.RateLimitError,
            litellm.InternalServerError,
            litellm.OpenAIError,
        ),
    )
    def _call() -> object:
        return litellm.completion(
            model=model,
            api_base=api_base,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )

    response = _call()
    normalized = (response.choices[0].message.content or "").strip()

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(normalized, encoding="utf-8")
    return normalized


def _cache_path(prompt: str, model: str, cache_dir: Path) -> Path:
    key = hashlib.sha256((model + "\x00" + prompt).encode()).hexdigest()
    return cache_dir / f"convert-{key}.txt"
