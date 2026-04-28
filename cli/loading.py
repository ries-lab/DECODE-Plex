import argparse
from decode.utils.notebooks import load_examples


def main():
    parser = argparse.ArgumentParser("Load example notebooks.")
    parser.add_argument("path", metavar="N", type=str, help="Destination Path")

    load_examples(parser.parse_args().path)


if __name__ == "__main__":
    main()
