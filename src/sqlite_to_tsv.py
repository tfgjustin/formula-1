#!/usr/bin/python

from __future__ import print_function

import csv
import sqlite3
import sys

# Figure out the set of races we care about
_RACE_QUERY = '''
SELECT id,_type
FROM races
WHERE date > '1949-01-01' AND (_type=3 OR _type=4)
ORDER BY date DESC
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


def ConvertRaces(conn, outtsv):
  """Convert the races from the SQL lite database to our TSV format.
  """
  _FIELDS = ['Year', 'RaceID', 'RaceType', 'DriverID', 'Result']
  # Get list of valid races
  races = set()
  last_race_type = 0
  for row in conn.execute(_RACE_QUERY):
    race_id = row[0]
    race_type = row[1]
    if race_type == last_race_type:
      continue
    races.add(race_id)
    last_race_type = race_type
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
  if len(argv) < 3:
    print('Usage: %s <sqlitedb> <outtsv>' % (argv[0]))
    sys.exit(1)
  conn = sqlite3.connect(argv[1])
  ConvertRaces(conn, argv[2])


if __name__ == "__main__":
  main(sys.argv)
