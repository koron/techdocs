#!/usr/bin/python
#
# 13-inference.py の出力TSVを集計して accuracy, precision, recall, F-measure の各種評価値を求める

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
        print(f"Aggregate: {name}")
        Tp = 0; Tn = 0; Fp = 0; Fn = 0
        with open(name) as f:
            for row in csv.reader(f, delimiter="\t"):
                want = str_to_bool(row[1])
                got = str_to_bool(row[4])
                if want == got:
                    if want:
                        Tp += 1
                    else:
                        Tn += 1
                else:
                    if want:
                        Fn += 1
                    else:
                        Fp += 1
        print(f"  Tp={Tp} Tn={Tn} Fp={Fp} Fn={Fn}")
        accuracy = (Tp + Tn) / (Tp + Tn + Fp + Fn)
        print(f"  accuracy:  {accuracy}")
        precision = 0
        if Tp + Fp > 0:
            precision = Tp / (Tp + Fp)
            print(f"  precision: {precision}")
        if Tp + Fn > 0:
            recall = Tp / (Tp + Fn)
            print(f"  recall:    {recall}")
        if precision != 0 and recall != 0:
            fmeasure = 2 * precision * recall / (precision + recall)
            print(f"  F-measure: {fmeasure}")
