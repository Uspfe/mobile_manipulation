import argparse

import mm_control.MPC as MPC
from mm_utils import parsing

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to config file.")
    parser.add_argument(
        "-n",
        "--name",
        default="",
        required=False,
        help="Name Identifier for acados ocp. This will overwrite the name in the config file if provided.",
    )

    args = parser.parse_args()

    config = parsing.load_config(args.config)
    if args.name != "":
        config["controller"]["acados"]["name"] = args.name

    config["controller"]["acados"]["cython"]["enabled"] = True
    config["controller"]["acados"]["cython"]["recompile"] = True

    ctrl_config = config["controller"]
    control_class = getattr(MPC, ctrl_config["type"], None)
    if control_class is None:
        raise ValueError(f"Unknown controller type: {ctrl_config['type']}")
    controller = control_class(ctrl_config)
