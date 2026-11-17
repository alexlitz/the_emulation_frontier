"""Shared fixtures: the zoo pages' session page, relation and headless browser (tests/zoo_harness.py).

Tests that import or run code downloaded from the Hub or GitHub (exec_module on a snapshot's modeling files,
trust_remote_code=True, third-party plugins) are marked remote_code and skipped unless TEF_ALLOW_REMOTE_CODE=1.
"""
import os

import pytest

from tests.zoo_harness import browser, open_page, truth, zoo  # noqa: F401

# Tests that execute remote code. Names are matched without their parametrize suffix.
REMOTE_CODE_TESTS = {
    "test_diffusion_real.py": {
        "test_real_diffusion_encoder_is_bidirectional",
        "test_wider_host_reproduces_real_diffusion_forward_exactly",
    },
    "test_universal_elephantine.py": {
        "test_edgerock_memory_and_engram_blocks_fp64",
        "test_ffn_block_carries_motif3_polynorm",
        "test_same_host_carries_agnes_parallel_ffn",
        "test_same_host_carries_aliceai_attnres_sigmoid_router",
        "test_same_host_carries_allgdn_engram",
        "test_same_host_carries_ceno",
        "test_same_host_carries_dqwen35_diffusion",
        "test_same_host_carries_edgerock_fixed_memory",
        "test_same_host_carries_gigachat35",
        "test_same_host_carries_gka_hybrid_qwen3",
        "test_same_host_carries_glm_superexpert_own_code",
        "test_same_host_carries_gpts3_gdn2_full_attnres",
        "test_same_host_carries_granite_switch_alora",
        "test_same_host_carries_hymba_parallel_kv_sharing_meta_tokens",
        "test_same_host_carries_hymba_window_with_meta_token_sinks",
        "test_same_host_carries_jet_nemotron",
        "test_same_host_carries_lfm2_moe_and_lfm3",
        "test_same_host_carries_lfm3_rd_unrolled",
        "test_same_host_carries_mini_k3",
        "test_same_host_carries_motif3_whole_model",
        "test_same_host_carries_phi4flash_sambay",
        "test_same_host_carries_plamo2",
        "test_same_host_carries_pseudobrain_vllm_plugin",
        "test_same_host_carries_qwen35_moe_and_memt",
        "test_same_host_carries_youtu_kda_mla",
        "test_ssm_block_carries_gated_kalmanet",
        "test_ssm_block_carries_mamba3_mimo",
        "test_ssm_block_carries_nemotron_flash_deltanet",
        "test_ssm_block_carries_nemotron_flash_mamba2",
        "test_ssm_block_carries_rwkv5",
        "test_ssm_block_carries_rwkv6_ddlerp",
        "test_ssm_block_carries_rwkv7_with_v_first",
    },
}


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: a real-checkpoint check (skipped when the checkpoint is not cached)")
    config.addinivalue_line("markers", "remote_code: runs code downloaded from the Hub / GitHub "
                                       "(skipped unless TEF_ALLOW_REMOTE_CODE=1)")


def pytest_collection_modifyitems(config, items):
    allow = os.environ.get("TEF_ALLOW_REMOTE_CODE") == "1"
    skip = pytest.mark.skip(reason="executes remote code; set TEF_ALLOW_REMOTE_CODE=1 to run")
    for item in items:
        names = REMOTE_CODE_TESTS.get(os.path.basename(str(item.fspath)), ())
        if item.originalname in names or item.name in names:
            item.add_marker(pytest.mark.remote_code)
            if not allow:
                item.add_marker(skip)
