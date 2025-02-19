#!/usr/bin/python
#
# 学習データから使わないデータ(例: True Positive/Negative)をフィルタするプログラム

def str_to_bool(s):
    s = s.lower()
    if s == 'true' or s == 'yes':
        return True
    else:
        return False

if __name__ == '__main__':
    import argparse
    import sys
    import csv

    parser = argparse.ArgumentParser()
    args, remains = parser.parse_known_args(sys.argv[1:])
    for key, value in vars(args).items():
        # TODO:
        pass

    for name in remains:
        with open(name) as f:
            for row in csv.reader(f, delimiter="\t"):
                want = str_to_bool(row[1])
                got = str_to_bool(row[5])
                if want == got:
                    continue
                print('\t'.join(row))

