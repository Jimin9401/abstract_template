import yaml
import logging
import json

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


class Config:
    def __init__(self, file):
        if file.endswith(".json"):
            with open(file, "r", encoding="utf-8") as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                setattr(self, key, value)

        elif file.endswidth(".yaml"):
            with open(file, "r") as f:
                data = yaml.load(f.read(), Loader=yaml.FullLoader)
            self.__dict__ = data

    @staticmethod
    def save(config: dict, path):
        if path.endswith(".json"):
            with open(path, "w") as out_file:
                json.dump(config, out_file)
        elif path.endswith(".yaml"):

            with open(path, "w") as out_file:
                yaml.dump(config, out_file)


# for debugging


if __name__ == "__main__":
    # tmp_dict={"tmp_path":"data/cache/train.pkl","model_check_point":"gpt-2"}
    output_path = "tmp.yaml"

    # Config.save_as_yaml(tmp_dict,output_path)

    args = Config(output_path)

