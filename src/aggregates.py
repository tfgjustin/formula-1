#!/usr/bin/python

from __future__ import print_function

from collections import defaultdict
from scipy import mean
from scipy.stats import hmean

import csv
import sys


def ImportDrivers(intsv, drivers):
  """Creates a mapping from driver ID to the name.
  """
  with open(intsv, 'r') as infile:
    tsvreader = csv.DictReader(infile, delimiter='\t')
    for row in tsvreader:
      if 'DriverID' not in row or 'DriverName' not in row:
        continue
      drivers[row['DriverID']] = row['DriverName']


def ImportRaces(intsv, races):
  """Creates a mapping for season ~> round ~> race ID

  Input row:
    S1950E0043DR    1950    1       1950-05-13      1950 British Grand Prix
  Output value:
    [1950][1] = S1950E0043DR
  """
  with open(intsv, 'r') as infile:
    tsvreader = csv.DictReader(infile, delimiter='\t')
    for row in tsvreader:
      races[row['Season']][row['Round']] = row['RaceID']



def LoadFlatData(filename, metrics, dnfs, data):
  """Loads flattened data.

  Format:
    RaceID	DriverID	Placed	#Drivers	[Metric0]Pre	[Metric0]Post

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
      if row['Placed'] == 'dnf':
        dnfs[driver_id][race_id] = 1
      placed = row['Placed']
      for metric in metrics:
        data[driver_id][year][race_id][metric] = float(row[metric + 'Post'])


def MinPctEvents(year):
  y = int(year)
  if y < 1970:
    return 0.5
  else:
    return 1.2


def PrintOneYearPeak(data, metrics, races, drivers, tsvwriter):
  """Print the peak value a driver had in a year.
  """
  for driver,year_dict in data.items():
    for year,race_dict in year_dict.items():
      if len(race_dict) < (len(races[year]) * MinPctEvents(year)):
        continue
      for metric in metrics:
        max_val = max([x[metric] for x in race_dict.values()])
        tsvwriter.writerow(['Peak', metric, driver, drivers[driver], year, year,
                            '%.4f' % max_val])
#        print('Peak,%s,%s,%s,%s,%s,%.4f' % (metric, driver, drivers[driver], year, year, max_val))


def PrintOneYearOverall(data, metrics, races, drivers, tsvwriter):
  """Calculates the harmonic mean of the season peak, average, and end rating.
  """
  for driver,year_dict in data.items():
    for year,race_dict in year_dict.items():
      if len(race_dict) < (len(races[year]) * MinPctEvents(year)):
        continue
      sorted_races = sorted(race_dict.keys())
      for metric in metrics:
        ratings = [x[metric] for x in race_dict.values()]
        max_val = max(ratings)
        season_mean = mean(ratings)
        season_end = race_dict[sorted_races[-1]][metric]
        overall = hmean([max_val, season_mean, season_end])
        tsvwriter.writerow(['Overall', metric, driver, drivers[driver], year,
                            year, '%.4f' % overall])
#        print('Overall,%s,%s,%s,%s,%s,%.4f' % (metric, driver, drivers[driver],
#              year, year, overall))


def PrintNYearsAverage(data, metrics, dnfs, races, drivers, tag, num_years,
                       tsvwriter):
  for driver,year_dict in data.items():
    driver_dnf = set()
    if driver in dnfs:
      driver_dnf = dnfs[driver]
    for start_year in sorted(year_dict.keys()):
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
        champ_races = {race_id:rating for race_id,rating in year_dict[year].items()}
#        print('DNFed:%s:%s %s' % (driver, year, ' '.join([race_id for race_id in champ_races.keys() if race_id in driver_dnf])))
        champ_races = {race_id:rating for race_id,rating in champ_races.items() if race_id not in driver_dnf}
        if len(champ_races) < (len(races[year]) * MinPctEvents(year)):
          skip = True
          break
        values.extend(champ_races.values())
      if skip or len(values) == 0:
        continue
      for metric in metrics:
        tsvwriter.writerow(
          [tag, metric, driver, drivers[driver], start_year, end_year,
           '%.4f' % mean([v[metric] for v in values])]
        )
#        print('%s,%s,%s,%s,%s,%s,%.4f' % (tag, metric, driver, drivers[driver],
#          start_year, end_year, mean([v[metric] for v in values])))


def PrintHeaders(tsvwriter):
  tsvwriter.writerow(['Aggregate', 'Metric', 'DriverID', 'DriverName',
                      'StartYear', 'EndYear', 'Value'])


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
      PrintNYearsAverage(data, metrics, dnfs, races, drivers, tag, num_years,
                         tsvwriter)


if __name__ == "__main__":
  main(sys.argv)
