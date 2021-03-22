#!/usr/bin/python

import csv
import math
import sys

from collections import defaultdict


_DEFAULT_NUM_TO_PRINT = 30
_DEFAULT_ALLOW_DUPES = 0


def LoadMetrics(infile, aggregation, metric_id, values):
    with open(infile, 'r') as input_tsv:
        tsv_reader = csv.reader(input_tsv, delimiter='\t')
        for row in tsv_reader:
            if row[0] == aggregation and row[1] == metric_id:
                values.append(row)


def PrintHeader(tsv_writer):
    tsv_writer.writerow(['Rank', '%-35s' % 'Entry', 'Start', 'End', 'Value'])


def in_seen_range(team_uuid, start_year, end_year, seen_ranges):
    if team_uuid not in seen_ranges:
        return False
    for team_range in seen_ranges[team_uuid]:
        years = team_range.split(':')
        start_range = int(years[0])
        end_range = int(years[1])
        if start_range <= start_year <= end_range:
            return True
        elif start_range <= end_year <= end_range:
            return True
    return False


def PrintResults(values, outfile, params):
    num_to_print = _DEFAULT_NUM_TO_PRINT
    allow_dupes = _DEFAULT_ALLOW_DUPES
    # Possibly override the 
    if len(params) > 0:
        num_to_print = int(params[0])
        if len(params) > 1:
            allow_dupes = int(params[1])
    seen_ranges = dict()
    seen = set()
    rankings = defaultdict(list)
    count = 0
    last_rating = None
    for row in sorted(values, key=lambda x: float(x[-1]), reverse=True):
        team_uuid = row[2]
        start_year = int(row[4])
        end_year = int(row[5])
        if not allow_dupes:
            if team_uuid in seen:
                continue
            seen.add(team_uuid)
        elif allow_dupes == 1:
            if in_seen_range(team_uuid, start_year, end_year, seen_ranges):
                continue
            if team_uuid in seen_ranges:
                seen_ranges[team_uuid].add('%d:%d' % (start_year, end_year))
            else:
                seen_ranges[team_uuid] = set(['%d:%d' % (start_year, end_year)])
        rating = math.floor(float(row[-1]))
        row[-1] = rating
        rankings[rating].append(row)
        count += 1
        if last_rating is None:
            last_rating = rating
        elif rating == last_rating:
            continue
        elif count >= num_to_print:
            break

    rank = 1
    with open(outfile, 'w') as out_tsv:
        tsv_writer = csv.writer(out_tsv, delimiter='\t')
        PrintHeader(tsv_writer)
        for rating in sorted(rankings.keys(), reverse=True):
            rows = rankings[rating]
            prefix = ' '
            if len(rows) > 1:
                prefix = 'T'
            for row in rows:
                tsv_writer.writerow(['%s%4d' % (prefix, rank), '%-35s' % row[3]] + row[4:])
            rank += len(rows)


def main(argv):
    if len(argv) < 5 or len(argv) > 7:
        print(('Usage: %s <in:metrics_tsv> <in:aggregation> <in:metric_id> '
               '<out:rankings_tsv> <in:number_to_print> [in:allow_dupes]') % (argv[0]))
        sys.exit(1)
    values = list()
    LoadMetrics(argv[1], argv[2], argv[3], values)
    PrintResults(values, argv[4], argv[5:])


if __name__ == "__main__":
    main(sys.argv)
