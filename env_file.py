# coding: utf-8
"""Load KEY=VALUE lines from the .env file next to this script.

Used for WANDB_API_KEY: wandb reads it from the environment, so putting it in
.env lets run_training.bat log in without a separate ``wandb login``. No
python-dotenv dependency -- the format needed here is one line per key.

Variables already set in the real environment win over .env, and empty values
are skipped, so the placeholder ``WANDB_API_KEY=`` changes nothing.
"""

import os
import os.path as osp

ENV_PATH = osp.join(osp.dirname(osp.abspath(__file__)), '.env')


def load_env_file(path=ENV_PATH):
    """Return the names of the variables loaded from ``path``."""
    loaded = []
    if not osp.exists(path):
        return loaded
    with open(path, encoding='utf-8-sig') as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            if key.lower().startswith('export '):
                key = key[7:].strip()
            value = value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value
                loaded.append(key)
    return loaded
