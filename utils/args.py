import argparse
import os




class ExperimentArgument:
    def __init__(self):

        data = {}
        parser=self.get_args()
        args=parser.parse_args()

        data.update(vars(args))
        self.data = data
        self.set_savename()
        self.__dict__ = data

    def get_args(self):

        parser=argparse.ArgumentParser(description="OOD analysis")
        parser.add_argument("--root", type=str,required=True)
        parser.add_argument("--seed", type=int, default=777)
        parser.add_argument("--dataset", choices=["product","movie"],type=str,required=True)
        parser.add_argument("--model", choices=["transformer","logistic","tree"],type=str,default="transformer")
        parser.add_argument("--checkpoint_dir",  type=str, default="checkpoints")

        parser.add_argument("--do_train",action="store_true")
        parser.add_argument("--do_eval", action="store_true")
        parser.add_argument("--do_test", action="store_true")
        parser.add_argument("--evaluation_training", action="store_true")

        return parser

    def set_savename(self):
        self.data["checkpoint_name"]=os.path.join(self.data["checkpoint_dir"],self.data["savename"])
        if not os.path.isdir(self.data["checkpoint_name"]):
            os.makedirs(self.data["checkpoint_name"])

        if self.data["do_test"]:
            if not os.path.isdir(self.data["save_example_path"]):
                os.makedirs(self.data["save_example_path"])
