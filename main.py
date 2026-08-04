import sys


def main() -> None:
    print("prompte:", sys.argv[1:])
    for arg in sys.argv[1:]:
        print(arg)


if __name__ == "__main__":
    main()
