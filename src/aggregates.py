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
    csvreader = csv.DictReader(infile, delimiter='\t')
    for row in csvreader:
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
    csvreader = csv.DictReader(infile, delimiter='\t')
    for row in csvreader:
      races[row['Season']][row['Round']] = row['RaceID']


# RaceID,DriverID,Placed,NumDrivers,[Metric0]Pre,[Metric0]Post,[Metric1]Pre,[Metric1]Post
def LoadData(filename, metrics, dnfs, data):
  # [driver][year][race][metric] = value
  with open(filename, 'r') as infile:
    csvreader = csv.DictReader(infile, delimiter=',')
    metrics.update([x[:-4] for x in csvreader.fieldnames if x.endswith('Post')])
    for row in csvreader:
      driver_id = row['DriverID']
      year = row['RaceID'][1:5]
      race_id = row['RaceID']
      if row['Placed'] == 'dnf':
        dnfs[driver_id][race_id] = 1
      placed = row['Placed']
      for metric in metrics:
        data[driver_id][year][race_id][metric] = float(row[metric + 'Post'])


def MinPctRaces(year):
  y = int(year)
  if y < 1970:
    return 0.5
  else:
    return 1.2

# Output
# StatType,MetricType,Driver ID,Driver Name,Year Start,Year End,MetricValue
def PrintOneYearPeak(data, metrics, races, drivers):
  for driver,year_dict in data.items():
    for year,race_dict in year_dict.items():
      if len(race_dict) < (len(races[year]) * MinPctRaces(year)):
        continue
      for metric in metrics:
        max_val = max([x[metric] for x in race_dict.values()])
        print('Peak,%s,%s,%s,%s,%s,%.4f' % (metric, driver, drivers[driver], year, year, max_val))


def PrintOneYearOverall(data, metrics, races, drivers):
  """Calculates the harmonic mean of the season peak, average, and end rating.
  """
  for driver,year_dict in data.items():
    for year,race_dict in year_dict.items():
      if len(race_dict) < (len(races[year]) * MinPctRaces(year)):
        continue
      sorted_races = sorted(race_dict.keys())
      for metric in metrics:
        ratings = [x[metric] for x in race_dict.values()]
        max_val = max(ratings)
        season_mean = mean(ratings)
        season_end = race_dict[sorted_races[-1]][metric]
        overall = hmean([max_val, season_mean, season_end])
        print('Overall,%s,%s,%s,%s,%s,%.4f' % (metric, driver, drivers[driver],
              year, year, overall))


def PrintNYearsAverage(data, metrics, dnfs, races, drivers, tag, num_years):
  for driver,year_dict in data.items():
    driver_dnf = set()
    if driver in dnfs:
      driver_dnf = dnfs[driver]
    # TODO: Figure out metrics for this
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
        if len(champ_races) < (len(races[year]) * MinPctRaces(year)):
          skip = True
          break
        values.extend(champ_races.values())
      if skip or len(values) == 0:
        continue
      for metric in metrics:
        print('%s,%s,%s,%s,%s,%s,%.4f' % (tag, metric, driver, drivers[driver],
          start_year, end_year, mean([v[metric] for v in values])))


def main(argv):
  nested_dict = lambda: defaultdict(nested_dict)
  if len(argv) != 4:
    print('Usage: %s <drivers.tsv> <races.tsv> <ratings.tsv>' % (argv[0]))
    sys.exit(1)
  drivers = dict()
  ImportDrivers(argv[1], drivers)
  races = nested_dict()
  ImportRaces(argv[2], races)

  metrics = set()
  data = nested_dict()
  dnfs = nested_dict()
  LoadData(argv[3], metrics, dnfs, data)

  PrintOneYearPeak(data, metrics, races, drivers)
  PrintOneYearOverall(data, metrics, races, drivers)
  for num_years in [1, 3, 5, 7]:
    tag = 'Range%dy' % num_years
    PrintNYearsAverage(data, metrics, dnfs, races, drivers, tag, num_years)


if __name__ == "__main__":
  main(sys.argv)
