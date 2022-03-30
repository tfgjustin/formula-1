import csv
import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
import sys

from collections import defaultdict


def select_reference_events(events, year):
    year_events = sorted(events[year])
    num_events = len(year_events)
    start_idx = math.ceil(3. * num_events / 4)
    return year_events[start_idx:-1]


def select_test_events(events, year):
    year_events = sorted(events[year])
    num_events = len(year_events)
    start_idx = math.ceil(num_events / 2)
    return year_events[0:14]


def load_events(filename, reference_events, test_events, results):
    all_events = defaultdict(set)
    with open(filename, 'r') as infile:
        reader = csv.DictReader(infile, delimiter='\t')
        for row in reader:
            if 'RaceID' not in row or 'EloPost' not in row:
                continue
            if not row['RaceID'].endswith('-R'):
                continue
            event_id = row['RaceID']
            year = event_id[1:5]
            if int(year) < 1991 or int(year) > 2021:
                continue
            all_events[year].add(event_id)
            rating = float(row['EffectPost'])
            key = None
            if 'TeamUUID' in row:
                key = row['TeamUUID']
            elif 'DriverID' in row:
                key = row['DriverID']
            else:
                continue
            results[key][event_id] = rating
    for year, events in all_events.items():
        reference_events[year] = select_reference_events(all_events, year)
        test_events[year] = select_test_events(all_events, year)


def process_events(reference_events, test_events, results):
    # Create [year][entrant] = reference_score
    eoy_reference_ratings = defaultdict(dict)
    for year, events in reference_events.items():
        for entrant, event_ratings in results.items():
            ratings = [event_ratings[event_id] for event_id in events if event_id in event_ratings]
            if len(ratings) < 3:
                continue
            target = sum(ratings) / len(ratings)
            if target < 50:
                continue
            eoy_reference_ratings[year][entrant] = target
    # Create [round][current, reference] array
    regression = defaultdict(list)
    errors = defaultdict(list)
    for year, events in test_events.items():
        for entrant, event_ratings in results.items():
            for event_id in events:
                if event_id not in event_ratings:
                    continue
                if entrant not in eoy_reference_ratings[year]:
                    continue
                current = event_ratings[event_id]
                target = eoy_reference_ratings[year][entrant]
                round_key = event_id[6:-2]
                regression[round_key].append([current, target])
                errors[round_key].append(abs(current - target))
    for f1_round, data in regression.items():
        dataframe = pd.DataFrame(data)
        x = pd.DataFrame(dataframe[0])
        y = dataframe[1]
        model = sm.OLS(y, x).fit()
        print('R%s\t%.3f\t%.3f' % (f1_round, model.rsquared, np.std(errors[f1_round])))


def main(argv):
    if len(argv) != 3:
        print('Usage: %s <in:ratings_tsv> <out:convergence_data>' % (argv[0]))
        sys.exit(1)
    reference_events = defaultdict(set)
    test_events = defaultdict(set)
    results = defaultdict(dict)
    load_events(argv[1], reference_events, test_events, results)
    process_events(reference_events, test_events, results)


if __name__ == "__main__":
    main(sys.argv)
