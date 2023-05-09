#!/usr/bin/python

from __future__ import print_function

import binascii
import csv
import datetime
import sys

from sklearn.preprocessing import LabelEncoder


# List of columns that are valid for races and then results
_VALID_RACE_COLUMNS = ['season', 'stage', 'date', 'race']
_VALID_RESULT_COLUMNS = _VALID_RACE_COLUMNS + ['finish', 'start', 'driver_ID']
# Time delta to calculate qualifying day from a race day
_ONE_DAY_DELTA = datetime.timedelta(days=1)


def MakeFullRaceID(year, event_id, event_type):
  if event_type not in ['QU', 'RA']:
    return None
  return 'S%sE%04dD%s' % (year, event_id, event_type)


def MakeFullRaceName(year, race_tag, suffix):
  return '%s %s%s' % (year, race_tag.replace('_', ' '), suffix)


def QualifyingForDate(race_date):
  rd = datetime.date.fromisoformat(race_date)
  qd = rd - _ONE_DAY_DELTA
  return qd.isoformat()


def ConvertDrivers(intsv, outtsv, drivers):
  """Get the list of drivers, LabelEncode them, and write to the specified TSV.
  """
  _FIELDS = ['DriverID', 'DriverName']
  names = dict()
  with open(intsv, 'r') as infile:
    tsvreader = csv.DictReader(infile, delimiter='\t')
    for row in tsvreader:
      if 'driver_ID' not in row or 'driver_name' not in row:
        continue
      # names['firstname_lastname'] = 'firstname lastname'
      names[row['driver_ID']] = row['driver_name']
  driver_tags = list(names.keys())
  name_encoder = LabelEncoder()
  name_encoder.fit(driver_tags)
  for string_key,name in names.items():
    driver_id = name_encoder.transform([string_key])[0]
    # drivers['firstname_lastname'] = 123
    drivers[string_key] = driver_id
  with open(outtsv, 'w') as outfile:
    tsvwriter = csv.DictWriter(outfile, delimiter='\t', fieldnames=_FIELDS)
    tsvwriter.writeheader()
    for driver_id in sorted(drivers.keys()):
      outdict = dict.fromkeys(_FIELDS)
      outdict['DriverID'] = drivers[driver_id]
      outdict['DriverName'] = names[driver_id]
      tsvwriter.writerow(outdict)


def IsValidRaceRow(row):
  for h in _VALID_RACE_COLUMNS:
    if h not in row:
      return False
  if 'Indianapolis' in row['race']:
    return False
  return True


def ConvertRaces(intsv, outtsv, races):
  """Get the list of races, write to a TSV, and store the set of races.
  """
  tmpraces = dict()
  with open(intsv, 'r') as infile:
    tsvreader = csv.DictReader(infile, delimiter='\t')
    for row in tsvreader:
      if not IsValidRaceRow(row):
        continue
      tmpraces[row['date']] = {h:row[h] for h in _VALID_RACE_COLUMNS}
  _FIELDS = ['RaceID', 'Season', 'Round', 'Date', 'RaceName']
  with open(outtsv, 'w') as outfile:
    tsvwriter = csv.DictWriter(outfile, delimiter='\t', fieldnames=_FIELDS)
    tsvwriter.writeheader()
    race_id = 1
    for date in sorted(tmpraces.keys()):
      row = tmpraces[date]
      # Each race has a qualifying session and a race. Qualifying happens the
      # day before the race.
      outdict = dict.fromkeys(_FIELDS)
      # Qualifying first
      # S1950E0042DQ
      outdict['RaceID'] = MakeFullRaceID(row['season'], race_id, 'QU')
      outdict['Season'] = row['season']
      outdict['Round'] = row['stage']
      outdict['Date'] = QualifyingForDate(date)
      outdict['RaceName'] = MakeFullRaceName(row['season'], row['race'],
                                             ' – Qualifying')
      races[row['date'] + 'QU'] = race_id
      tsvwriter.writerow(outdict)
      race_id += 1
      # Now the race itself
      outdict['RaceID'] = MakeFullRaceID(row['season'], race_id, 'QU')
      outdict['Date'] = date
      outdict['RaceName'] = MakeFullRaceName(row['season'], row['race'], '')
      races[row['date'] + 'RA'] = race_id
      tsvwriter.writerow(outdict)
      race_id += 1


def IsValidResultRow(row):
  for h in _VALID_RESULT_COLUMNS:
    if h not in row:
      return False
  if 'Indianapolis' in row['race']:
    return False
  return True


def ConvertResults(intsv, drivers, races, outtsv):
  """Convert the races from the racing-reference TSV to our TSV format.
  """
  _FIELDS = ['Year', 'RaceID', 'RaceType', 'DriverID', 'Result']
  with open(intsv, 'r') as infile:
    tsvreader = csv.DictReader(infile, delimiter='\t')
    with open(outtsv, 'w') as outfile:
      tsvwriter = csv.DictWriter(outfile, delimiter='\t', fieldnames=_FIELDS)
      tsvwriter.writeheader()
      for row in tsvreader:
        if not IsValidResultRow(row):
          continue
        # Qualifying
        outdict = dict.fromkeys(_FIELDS)
        outdict['Year'] = row['season']
        outdict['RaceID'] = races[row['date'] + 'QU']
        outdict['RaceType'] = 'QU'
        outdict['DriverID'] = drivers[row['driver_ID']]
        # If we can't convert a result to an int it's because they didn't start
        # (or finish) the race successfully.
        try:
          outdict['Result'] = int(row['start'])
        except:
          continue
        tsvwriter.writerow(outdict)
        # Race
        outdict['RaceID'] = races[row['date'] + 'RA']
        outdict['RaceType'] = 'RA'
        try:
          outdict['Result'] = int(row['finish'])
        except:
          continue
        tsvwriter.writerow(outdict)


def main(argv):
  if len(argv) != 5:
    print(('Usage: %s <in:reference_tsv> <out:drivers_tsv> <out:races_tsv> '
           '<out:results_tsv>') % (argv[0]))
    sys.exit(1)
  drivers = dict()
  ConvertDrivers(argv[1], argv[2], drivers)
  races = dict()
  ConvertRaces(argv[1], argv[3], races)
  ConvertResults(argv[1], drivers, races, argv[4])


if __name__ == "__main__":
  main(sys.argv)
