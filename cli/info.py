from decode.utils.bookkeeping import decode_state


def main():
    v = decode_state()
    print(f"DECODE version: {v}")


if __name__ == "__main__":
    main()
