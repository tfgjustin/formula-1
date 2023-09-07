#!/usr/bin/python3

from collections import defaultdict
from numpy import mean
from scipy.stats import hmean

import csv
import sys


def ImportDrivers(intsv, drivers):
    """Creates a mapping from driver ID to the name.
    """
    with open(intsv, 'r') as infile:
        tsvreader = csv.DictReader(infile, delimiter='\t')
        for row in tsvreader:
            if 'driver_id' not in row or 'driver_name' not in row:
                continue
            drivers[row['driver_id']] = row['driver_name']


def ImportRaces(intsv, races):
    """Creates a mapping for season ~> round ~> race ID

    Input row:
        S1950E0043DR  1950  1 1950-05-13  1950 British Grand Prix
    Output value:
        [1950][1] = S1950E0043DR
    """
    with open(intsv, 'r') as infile:
        tsvreader = csv.DictReader(infile, delimiter='\t')
        for row in tsvreader:
            races[row['season']][row['stage']] = row['event_id']


def LoadFlatData(filename, metrics, dnfs, data):
    """Loads flattened data.

    Format:
        RaceID	DriverID	Placed	#Drivers	DnfReason [Metric0]Pre	[Metric0]Post

    Output dictionary:
        [driver][year][race][metric] = value
    """
    with open(filename, 'r') as infile:
        tsvreader = csv.DictReader(infile, delimiter='\t')
        metrics.update([x[:-4] for x in tsvreader.fieldnames if x.endswith('Post')])
        for row in tsvreader:
            driver_id = row['DriverID']
            year = row['RaceID'][1:5]
            race_id = row['RaceID']
            if race_id[-2:] != 'RA':
                continue
            if row['DnfReason'] != '-':
                dnfs[driver_id][race_id] = 1
            for metric in metrics:
                data[driver_id][year][race_id][metric] = float(row[metric + 'Post'])


def MinPctRaces(year):
    """The minimum pct of events a driver has to participate in to get a rating.
    """
    y = int(year)
    if y < 1980:
        return 0.4
    elif y < 2000:
        return 0.6
    else:
        return 0.7


def PrintOneYearPeak(data, metrics, races, drivers, tsvwriter):
    """Print the peak value a driver had in a year.
    """
    for driver, year_dict in data.items():
        for year, race_dict in year_dict.items():
            if len(race_dict) < (len(races[year]) * MinPctRaces(year)):
                continue
            if len(race_dict) <= 2:
                continue
            for metric in metrics:
                max_val = max([x[metric] for x in race_dict.values()])
                tsvwriter.writerow(['Peak', metric, driver, drivers[driver], year, year, '%.4f' % max_val])


def PrintOneYearOverall(data, metrics, races, drivers, tsvwriter):
    """Calculates the harmonic mean of the season peak, average, and end rating.
    """
    for driver, year_dict in data.items():
        for year, race_dict in year_dict.items():
            if len(race_dict) < (len(races[year]) * MinPctRaces(year)):
                continue
            sorted_races = sorted(race_dict.keys())
            for metric in metrics:
                ratings = [x[metric] for x in race_dict.values()]
                min_val = min(ratings)
                if min_val < 0:
                    continue
                max_val = max(ratings)
                season_mean = mean(ratings)
                season_end = race_dict[sorted_races[-1]][metric]
                overall = hmean([max_val, season_mean, season_end])
                tsvwriter.writerow(['Overall', metric, driver, drivers[driver], year, year, '%.4f' % overall])


def PrintNYearsAverage(data, metrics, dnfs, races, drivers, tag, num_years, tsvwriter):
    """Calculates the average rating over an N-year window.
    """
    for driver, year_dict in data.items():
        driver_dnf = set()
        if driver in dnfs:
            driver_dnf = dnfs[driver]
        for start_year in sorted(year_dict.keys()):
            num_completed = 0
            num_required = 0
            end_year = int(start_year) + num_years - 1
            if str(end_year) not in year_dict:
                continue
            values = []
            skip = False
            for y in range(int(start_year), end_year + 1):
                year = str(y)
                if year not in year_dict:
                    skip = True
                    break
                # Championship races are initially all the races in the year in which the
                # driver received a rating.
                champ_races = {race_id: rating for race_id, rating in year_dict[year].items()}
                # Now filter down to the ones where the driver finished.
                champ_races = {race_id: rating for race_id, rating in champ_races.items() if race_id not in driver_dnf}
                num_completed += len(champ_races)
                num_required += (len(races[year]) * (MinPctRaces(year)))
                # print('NY:%d\tD:%s\tY:%s\tCR:%d\tNC:%d\tNR:%.1f' % (
                #     num_years, driver, year, len(champ_races), num_completed, num_required
                # ))
                if len(champ_races) <= 3:
                    skip = True
                    break
                values.extend(champ_races.values())
            if not values or skip or num_completed < num_required:
                continue
            for metric in metrics:
                tsvwriter.writerow(
                    [tag, metric, driver, drivers[driver], start_year, end_year,
                     '%.4f' % mean([v[metric] for v in values])]
                )


def PrintHeaders(tsvwriter):
    tsvwriter.writerow(['Aggregate', 'Metric', 'DriverID', 'DriverName', 'StartYear', 'EndYear', 'Value'])


def main(argv):
    nested_dict = lambda: defaultdict(nested_dict)
    if len(argv) != 5:
        print(('Usage: %s <in:drivers_tsv> <in:races_tsv> <in:flat_ratings_tsv> '
               '<out:metrics_tsv>') % (argv[0]))
        sys.exit(1)
    drivers = dict()
    ImportDrivers(argv[1], drivers)
    races = nested_dict()
    ImportRaces(argv[2], races)

    metrics = set()
    data = nested_dict()
    dnfs = nested_dict()
    LoadFlatData(argv[3], metrics, dnfs, data)

    with open(argv[4], 'w') as outfile:
        tsvwriter = csv.writer(outfile, delimiter='\t')
        PrintHeaders(tsvwriter)
        PrintOneYearPeak(data, metrics, races, drivers, tsvwriter)
        PrintOneYearOverall(data, metrics, races, drivers, tsvwriter)
        for num_years in [1, 3, 5, 7]:
            tag = 'Range%dy' % num_years
            PrintNYearsAverage(data, metrics, dnfs, races, drivers, tag, num_years, tsvwriter)


if __name__ == "__main__":
    main(sys.argv)
