import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from config_loader import Config
import download_models


class ConfigEnvironmentOverrideTests(unittest.TestCase):
    def test_environment_api_keys_override_yaml_values(self):
        config_text = """
llm:
  api_key: yaml-llm-key
embeddings:
  api_key: yaml-embedding-key
reranker:
  api_key: yaml-reranker-key
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text(config_text, encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "QA_LLM_API_KEY": "env-llm-key",
                    "QA_EMBEDDINGS_API_KEY": "env-embedding-key",
                    "QA_RERANKER_API_KEY": "env-reranker-key",
                },
                clear=False,
            ):
                config = Config(str(config_path))

        self.assertEqual("env-llm-key", config.llm_config["api_key"])
        self.assertEqual("env-embedding-key", config.embeddings_config["api_key"])
        self.assertEqual("env-reranker-key", config.reranker_config["api_key"])


class ModelDownloadTests(unittest.TestCase):
    def test_modelscope_download_excludes_unused_onnx_artifacts(self):
        snapshot_download = Mock(return_value="models/bge-m3")
        snapshot_module = types.ModuleType("modelscope.hub.snapshot_download")
        snapshot_module.snapshot_download = snapshot_download

        with patch.dict(
            sys.modules,
            {"modelscope.hub.snapshot_download": snapshot_module},
        ):
            self.assertTrue(download_models.try_modelscope("BAAI/bge-m3", "models/bge-m3"))

        self.assertEqual(
            download_models.MODEL_FILE_PATTERNS,
            snapshot_download.call_args.kwargs["allow_patterns"],
        )

    @patch("download_models._download_file")
    def test_hf_mirror_downloads_only_required_model_files(self, download_file):
        self.assertTrue(download_models.try_huggingface("BAAI/bge-m3", "models/bge-m3"))

        downloaded_names = [call.args[0].rsplit("/", 1)[-1] for call in download_file.call_args_list]
        self.assertEqual(download_models.MODEL_FILES["BAAI/bge-m3"], downloaded_names)

    @patch("download_models._download_file")
    def test_hf_mirror_retries_an_interrupted_file_download(self, download_file):
        model_files = download_models.MODEL_FILES["BAAI/bge-reranker-v2-m3"]
        download_file.side_effect = [
            download_models.requests.ConnectionError("connection reset"),
            *([None] * len(model_files)),
        ]

        self.assertTrue(
            download_models.try_huggingface(
                "BAAI/bge-reranker-v2-m3", "models/bge-reranker-v2-m3"
            )
        )
        self.assertEqual(len(model_files) + 1, download_file.call_count)
