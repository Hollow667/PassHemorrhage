import os, sys
import time
from tqdm import tqdm
import argparse

def get_parser():

    parser = argparse.ArgumentParser(description="PAIB")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "-pg",
        "--passgen",
        action="store_true",
        help="Generate passwords using keywords given by the user",
    )

    group.add_argument(
        "-op",
        "--osint_passgen",
        action="store_true",
        help="helps to collect victim's info through inbuilt osint tool and generate passwords using the words collected",
    )
    group.add_argument(
        "-pt",
        "--pretrained",
        action="store_true",
        help="Generate password on a pretrained model using this command",
    )

    return parser

def main():

    parser = get_parser()
    args = parser.parse_args()

    if args.passgen:
        os.system("python3 train.py")
    elif args.osint_passgen:
        print("[+] Tool Developement in progress")
    elif args.pretrained:
        os.system("python3 sample.py")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
