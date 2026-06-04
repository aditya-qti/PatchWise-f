# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--model", default=None, help="--model passed to patchwise AiCodeReview")
    parser.addoption("--provider", default=None, help="--provider passed to patchwise AiCodeReview")
    parser.addoption("--embedding-model", default=None, help="litellm model for embeddings (default: text-embedding-3-small)")
    parser.addoption("--embedding-provider", default=None, help="api_base for litellm embedding calls")
    parser.addoption("--reviews-dir", default=None, help="use existing review .txt files from this directory instead of running patchwise (deepreview.txt with --deep-review)")
    parser.addoption("--deep-review", action="store_true", default=False, help="run patchwise --deep-review (DeepReview) and convert its output to issues instead of AiCodeReview")
    parser.addoption("--converter-model", default=None, help="LLM that converts DeepReview output into issues (default: judge model)")
    parser.addoption("--converter-provider", default=None, help="api_base for the converter LLM (default: judge provider)")
