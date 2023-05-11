import argparse
import os

from config import get_cfg_defaults


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help=".yml config path",
    )
    return parser.parse_args()


def main(cfg):
    pass


if __name__ == "__main__":
    args = parse_arguments()
    cfg = get_cfg_defaults()

    if os.path.exists(args.config):
        cfg.merge_from_file(args.config)

    print(cfg)
    main(cfg)