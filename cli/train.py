import os

os.environ.setdefault("MPLBACKEND", "Agg")

from decode.neuralfitter.train.train import train


def main():
    train()


if __name__ == "__main__":
    main()
