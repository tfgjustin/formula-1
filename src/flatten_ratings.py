#!/usr/bin/python

from __future__ import print_function

import csv
import sys

class ResultFlattener(object):
  """Take lines which have before and after separately and merge them."""
  def __init__(self):
    self._races = set()
    self._drivers = set()
    self._places = dict()
    self._results = dict()
    self._SCORE_TYPES = [ 'Elo', 'ZScore' ]
    self._NUM_METRICS = len(self._SCORE_TYPES)


  def LoadData(self, infile):
    """Load data from a TSV input file.

    Lines will be structured as:
      [ScoreType]   RaceID  Year    DriverID    Score   KScore
    or
      Place RaceID  Year    DriverID    Placed  OutOf

    E.g.,
      Elo	S2019E2654DRZ	2019	1188	968.542	24
      Place	S2019E2654DR	2019	1181	9	18
    """
    with open(infile, 'r') as intsv:
      tsvreader = csv.reader(intsv, delimiter='\t')
      for row in tsvreader:
        if row[0].startswith('#') or row[0].startswith('Errors'):
          continue
        score_type  = row[0]
        event_id    = row[1]
        year        = row[2]
        driver_id   = row[3]
        score_val   = row[4]
        score_extra = row[5]
        dict_key = '%s:%s:%s' % (score_type, event_id, driver_id)
        if score_type in self._SCORE_TYPES:
          # This is one of the metrics we care about
          self._results[dict_key] = score_val
        elif score_type == 'Place':
          # This is letting us know where this driver placed
          self._places[dict_key] = [ score_val, score_extra ]
        else:
          print('ERROR: Unknown race type: %s' % (event_id))
          continue
        self._races.add(event_id[:12])
        self._drivers.add(driver_id)


  def PrintData(self, outfile):
    """Print the flattened data to a TSV file.

    Lines will be structured as:
      RaceID,DriverID,Placed,NumDrivers,EloPre,EloPost,ZScorePre,ZScorePost

    E.g.,
      S1989E1529DR,1031,1,10,2296.917,2304.593,0.992,0.993
    """
    with open(outfile, 'w') as outtsv:
      tsvwriter = csv.writer(outtsv, delimiter='\t')
      self._WriteHeader(tsvwriter)
      for race_id in sorted(self._races):
        for driver_id in sorted(self._drivers):
          # These store the before and after for the N metrics we care about
          v = [ 'none', None ] * self._NUM_METRICS
          i = 0
          for value_type in self._SCORE_TYPES:
            key_z = '%s:%s%s:%s' % (value_type, race_id, 'Z', driver_id)
            if key_z not in self._results:
              break
            v[(self._NUM_METRICS * i) + 1] = self._results[key_z]
            key_a = '%s:%s%s:%s' % (value_type, race_id, 'A', driver_id)
            if key_a not in self._results:
              break
            v[(self._NUM_METRICS * i) + 0] = self._results[key_a]
            i += 1
          if v[-1] is None:
            # They didn't take part in this race, so we can skip them
            continue
          out_of = [ 'dnf', 'dnf' ]
          place_key = 'Place:%s:%s' % (race_id, driver_id)
          if place_key in self._places:
            # If they placed, include that information
            out_of = self._places[place_key]
          tsvwriter.writerow([race_id, driver_id, out_of[0], out_of[1]] + v)


  def _WriteHeader(self, tsvwriter):
    """Print the header for the metrics and basic information.
    """
    rowdata = ['RaceID', 'DriverID', 'Placed', 'NumDrivers']
    for score in self._SCORE_TYPES:
      rowdata = rowdata + [ score + 'Pre', score + 'Post' ]
    tsvwriter.writerow(rowdata)


def main(argv):
  if len(argv) != 3:
    print('Usage: %s <in:ratings_tsv> <out:flat_ratings_tsv>' % (argv[0]))
    sys.exit(1)
  flatten = ResultFlattener()
  flatten.LoadData(argv[1])
  flatten.PrintData(argv[2])


if __name__ == "__main__":
  main(sys.argv)
