# Copyright 2021 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Project hooks."""
from typing import Any, Dict, Iterable, Optional
import logging
import time
import click

from kedro.config import ConfigLoader, TemplatedConfigLoader
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog

class ProjectHooks:
    def _parse_cli_global(self, click_ctx):
        globals_dict = {}
        global_params_list = [
            "section",
            "start_distance",
            "end_distance",
            "batch_size",
            "model",
            "run_id",
            "seed",
            "total_len",
            "input_len",
            "max_epochs",
            "loss",
            "train_checkpoint",
            "test_checkpoint",
        ]
        if click_ctx.params.get("params") is None:
            return {}
        for param_name in global_params_list:
            param = click_ctx.params.get("params").get(param_name)
            if param is not None:
                globals_dict[param_name] = param
        return globals_dict

    def register_config_loader(
        self,
        conf_paths: Iterable[str],
        env: str,
        extra_params: Dict[str, Any],
    ) -> ConfigLoader:
        click_ctx = click.get_current_context(silent=True)
        globals_dict = self._parse_cli_global(click_ctx)
        return TemplatedConfigLoader(
            conf_paths, globals_pattern="globals.yml", globals_dict=globals_dict
        )

    def register_catalog(
        self,
        catalog: Optional[Dict[str, Dict[str, Any]]],
        credentials: Dict[str, Dict[str, Any]],
        load_versions: Dict[str, str],
        save_version: str,
    ) -> DataCatalog:
        return DataCatalog.from_config(
            catalog, credentials, load_versions, save_version
        )

    @hook_impl
    def before_dataset_loaded(self, dataset_name: str) -> None:
        start = time.time()
        logging.info("Loading dataset %s started at %0.3f", dataset_name, start)

    @hook_impl
    def after_dataset_loaded(self, dataset_name: str, data: Any) -> None:
        end = time.time()
        logging.info("Loading dataset %s ended at %0.3f", dataset_name, end)
