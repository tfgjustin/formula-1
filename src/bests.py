#!/usr/bin/python

from __future__ import print_function

import csv
import sys


_DEFAULT_NUM_TO_PRINT=30
_DEFAULT_ALLOW_DUPES=0


def LoadMetrics(infile, aggregation, metric_id, values):
  with open(infile, 'r') as intsv:
    tsvreader = csv.reader(intsv, delimiter='\t')
    for row in tsvreader:
      if row[0] == aggregation and row[1] == metric_id:
        values.append(row)


def PrintHeader(tsvwriter):
  tsvwriter.writerow(['Rank', '%-30s' % 'Driver', 'Start', 'End', 'Value'])


def PrintResults(values, outfile, params):
  num_to_print = _DEFAULT_NUM_TO_PRINT
  allow_dupes = _DEFAULT_ALLOW_DUPES
  # Possibly override the 
  if len(params) > 0:
    num_to_print = int(params[0])
    if len(params) > 1:
      allow_dupes = params[1]
  seen_drivers = set()
  num_printed = 1
  with open(outfile, 'w') as outtsv:
    tsvwriter = csv.writer(outtsv, delimiter='\t')
    PrintHeader(tsvwriter)
    for row in sorted(values, key=lambda x: float(x[-1]), reverse=True):
      if not allow_dupes:
        if row[2] in seen_drivers:
          continue
        seen_drivers.add(row[2])
      tsvwriter.writerow(['%4d' % num_printed, '%-30s' % row[3]] + row[4:])
      if num_printed >= num_to_print:
        return
      num_printed += 1


def main(argv):
  if len(argv) < 5 or len(argv) > 7:
    print(('Usage: %s <in:metrics_tsv> <in:aggregation> <in:metric_id> '
           '<out:rankings_tsv> <in:number_to_print> [in:allow_dupes]'
          ) % (argv[0])
         )
    sys.exit(1)
  values = list()
  LoadMetrics(argv[1], argv[2], argv[3], values)
  PrintResults(values, argv[4], argv[5:])


if __name__ == "__main__":
  main(sys.argv)
