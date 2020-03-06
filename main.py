import yaml
import logging
import json

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)

# for debugging
if __name__ == "__main__":
    tmp_path = "tmp.yaml"

    args = Config(tmp_path)

