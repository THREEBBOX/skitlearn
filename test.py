import argparse
parse=argparse.ArgumentParser()
parse.add_argument(
    "--add1",
    default=10,
    help="the add1",
    type=int
)
parse.add_argument(
    "--add2",
    default=10,
    help="the add2",
    type=int
)
args=parse.parse_args()
if __name__ == '__main__':
    a=args.add1
    b=args.add2
    print(a*b)
