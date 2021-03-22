#!/usr/bin/python3

import sys

def win_probability(r_a, r_b, den):
    """Standard logistic calculator, per
       https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details
    """
    q_a = 10 ** (r_a / den)
    q_b = 10 ** (r_b / den)
    return q_a / (q_a + q_b)


def main(argv):
    if len(argv) != 3 and len(argv) != 4:
        print(("Usage: {0} <in:elo_A> <in:elo_B>").format(argv[0]))
        sys.exit(1)
    elo_A = float(argv[1])
    elo_B = float(argv[2])
    den = 240
    if len(argv) > 3:
        den = float(argv[3])
    p = win_probability(elo_A, elo_B, den)
    print('%4d vs %4d: %.5f' % (elo_A, elo_B, p))
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv)
