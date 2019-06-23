#!/usr/bin/python

from __future__ import print_function

import csv
import sqlite3
import sys

# Figure out the set of races we care about
_RACE_LIST_QUERY = '''
SELECT id,_type
FROM races
WHERE date > '1949-01-01' AND (_type=3 OR _type=4)
ORDER BY date DESC
'''

# Get the dates and other info in the races we care about
_RACE_INFO_QUERY = '''
SELECT id,_type,date,race
FROM races
WHERE date > '1949-01-01' AND (_type=3 OR _type=4)
ORDER BY date
'''

# Get only the drivers who participated in the races we care about
_DRIVER_QUERY = '''
SELECT driver_entries._driver,drivers.driver
FROM races
LEFT JOIN entries ON races.id=entries._race
LEFT JOIN driver_entries ON entries.id=driver_entries._entry
LEFT JOIN drivers ON driver_entries._driver=drivers.id
WHERE date > '1949-01-01' AND (races._type=3 OR races._type=4)
GROUP BY driver_entries._driver
'''

# Get the driver placement for each race
_RESULT_QUERY = '''
SELECT races.id,races.race,races._type,driver_entries._driver,entries.result
FROM races
LEFT JOIN entries ON races.id=entries._race
LEFT JOIN driver_entries ON entries.id=driver_entries._entry
WHERE date > '1949-01-01' AND (races._type=3 OR races._type=4)
ORDER BY date,races.id,entries.result
'''


def MakeFullRaceID(year, event_id, race_type):
  tag = ''
  if race_type == 4:
    tag = 'R'
  elif race_type == 3:
    tag = 'Q'
  else:
    return None
  return 'S%dE%04dD%s' % (year, event_id, tag)


def ConvertDrivers(conn, outtsv):
  """Get the list of drivers and write to the specified TSV.
  """
  _FIELDS = ['DriverID', 'DriverName']
  with open(outtsv, 'w') as outfile:
    csvwriter = csv.DictWriter(outfile, delimiter='\t', fieldnames=_FIELDS)
    csvwriter.writeheader()
    for row in conn.execute(_DRIVER_QUERY):
      outdict = dict.fromkeys(_FIELDS)
      outdict['DriverID'] = row[0]
      outdict['DriverName'] = row[1]
      csvwriter.writerow(outdict)


def ConvertRaces(conn, outtsv, races):
  """Get the list of races, write to a TSV, and store the set of races.
  """
  last_race_type = 0
  for row in conn.execute(_RACE_LIST_QUERY):
    race_id = row[0]
    race_type = row[1]
    if race_type == last_race_type:
      continue
    races.add(race_id)
    last_race_type = race_type

  _FIELDS = ['RaceID', 'Season', 'Round', 'Date', 'RaceName']
  curr_round = 1
  curr_year = 0
  with open(outtsv, 'w') as outfile:
    csvwriter = csv.DictWriter(outfile, delimiter='\t', fieldnames=_FIELDS)
    csvwriter.writeheader()
    for row in conn.execute(_RACE_INFO_QUERY):
      race_id = row[0]
      if race_id not in races:
        continue
      race_type = row[1]
      # Use the race name for the year instead of the date of the event because
      # some qualifying sessions took place on New Year's Eve so the first race of
      # the season could take place on New Year's Day.
      year = int(row[3][:4])
      if year != curr_year:
        # A new season
        curr_year = year
        curr_round = 1
      outdict = dict.fromkeys(_FIELDS)
      outdict['RaceID'] = MakeFullRaceID(year, race_id, race_type)
      outdict['Season'] = year
      outdict['Round'] = curr_round
      outdict['Date'] = row[2]
      outdict['RaceName'] = row[3]
      csvwriter.writerow(outdict)
      if race_type == 4:
        curr_round += 1


def ConvertResults(conn, races, outtsv):
  """Convert the races from the SQL lite database to our TSV format.
  """
  _FIELDS = ['Year', 'RaceID', 'RaceType', 'DriverID', 'Result']
  # Input format is a sqlite3 connection
  with open(outtsv, 'w') as outfile:
    csvwriter = csv.DictWriter(outfile, delimiter='\t', fieldnames=_FIELDS)
    csvwriter.writeheader()
    for row in conn.execute(_RESULT_QUERY):
      if row[0] not in races:
        continue
      outdict = dict.fromkeys(_FIELDS)
      if row[2] == 3:
        outdict['RaceType'] = 'Q'
      elif row[2] == 4:
        outdict['RaceType'] = 'R'
      else:
        continue
      outdict['Year'] = row[1][:4]
      outdict['RaceID'] = row[0]
      outdict['DriverID'] = row[3]
      outdict['Result'] = row[4]
      csvwriter.writerow(outdict)


def main(argv):
  if len(argv) != 5:
    print('Usage: %s <sqlitedb> <drivertsv> <racetsv> <resulttsv>' % (argv[0]))
    sys.exit(1)
  conn = sqlite3.connect(argv[1])
  ConvertDrivers(conn, argv[2])
  races = set()
  ConvertRaces(conn, argv[3], races)
  ConvertResults(conn, races, argv[4])


if __name__ == "__main__":
  main(sys.argv)
