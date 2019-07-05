#!/usr/bin/python

from __future__ import print_function

import csv
import sys

from scipy.stats import norm

# Initial ranking and k-factor for a new driver.
INIT_RANKING = 1300
INIT_K_FACTOR = 32
# Revert this percentage back to the initial ranking each year.
YEAR_ADJ = 0.15
# Approximate number of drivers per race.
DRIVERS_PER_RACE = 25
# Minimum number of races before a driver is not considered "new".
MIN_RACES = 20
# What K-factor do "new" drivers use?
NEW_DRIVER_FACTOR = 24
# What's the minimum K-factor for a driver in a race?
MIN_RACE_FACTOR = 12
# Minimum K-factor for qualifying.
MIN_QUAL_FACTOR = 16
# How many races does a driver miss before we un-whitelist them?
# (Note: in the early years this usually happened because either their
#  team ran out of money, or the driver died in a crash.)
MISS_COUNT = 4
# What's the largest acceptable gap between two drivers to consider them
# to be competitors
MAX_GAP = 2000


class DriverRating(object):
  """Encapsulation class for Elo rating and K-factor."""
  rating = INIT_RANKING
  k_factor = INIT_K_FACTOR
  def __init__(self, init_rating, init_k_factor):
    rating = init_rating
    k_factor = init_k_factor


# Standard logistic calculator, per
# https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details
def WinProbability(r_a, r_b):
    """Standard logistic calculator, per
       https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details
    """
    q_a = 10 ** (r_a / 400)
    q_b = 10 ** (r_b / 400)
    return q_a / (q_a + q_b)


def InitDrivers(names, ratings, counts):
  """Initialize the full set of drivers with their initial rating, k-factor, and race count.

  names: Set of driver names
  ratings: Map of name to DriverRating object
  counts: Map of name to number of races in which the driver has participated
  """
  for name in names:
    ratings[name] = DriverRating(INIT_RANKING, INIT_K_FACTOR)
    counts[name] = 0


def SetDriversFromOrdering(ordering, curr_drivers):
  """Given a result ordering, ensure that all drivers are in the whitelist.
  """
  for a_place,a_name in ordering.items():
    if a_name not in curr_drivers:
      curr_drivers.add(a_name)


def OneRace(tag, ordering, ratings, counts, curr_drivers, missed_count, k_scale,
            tsvwriter):
  """Perform the Elo calculations for a single race.

  tag: unique string identifier for this race
  ordering: tuples of (place,driverId)
  ratings: map of driver ID to their DriverRating object
  counts: number of opponents this driver has faced
  curr_drivers: set of current active drivers
  missed_count: map of driver ID to the number of races they've missed
  k_scale: k-factor scaling for this race (1.0 for a race, 0.1 for qualifying)
  tsvwriter: TSV writer object where we log data
  """
  expected_wins = dict()
  actual_wins = dict()
  num_drivers = len(ordering)
  # For each driver go through and figure out how many opponents they should have
  # defeated versus how many they actually did defeat.
  for a_place,a_name in ordering.items():
    if a_name not in curr_drivers:
      curr_drivers.add(a_name)
    exp_wins = 0
    act_wins = 0
    for b_place,b_name in ordering.items():
      if a_name == b_name:
        # Driver cannot compete against themselves
        continue
      if abs(ratings[a_name].rating - ratings[b_name].rating) > MAX_GAP:
        # These drivers are too far apart to "compete" against each other
        continue
      exp_wins += WinProbability(ratings[a_name].rating, ratings[b_name].rating)
      if a_place < b_place:
        act_wins += 1
    expected_wins[a_name] = exp_wins
    actual_wins[a_name] = act_wins
    mse = (exp_wins - act_wins) ** 2
    tsvwriter.writerow(['Place', tag, tag[1:5], a_name, num_drivers - act_wins,
                        num_drivers])
    tsvwriter.writerow(['#ERR', tag, tag[1:5], a_name, '%.2f' % exp_wins,
                        '%.2f' % act_wins, '%.2f' % mse])
  # Mapping of driver ID to new ratings and k-factors
  new_ratings = dict()
  new_factors = dict()
  # Used for system-wide normalizing. This ensures that pointsIn==pointsOut.
  before_points = 0
  after_points = 0
  for name in curr_drivers:
    rating = ratings[name]
    before_points += rating.rating
    if name not in actual_wins:
      # Driver did not participate in/finish this race. In this case before==after points.
      after_points += rating.rating
      if name not in missed_count:
        # This driver hasn't missed a race before. First time for everything.
        missed_count[name] = 1
      else:
        # They've missed another. Could be worrying ....
        missed_count[name] += 1
    else:
      # Driver participated/finished (yay). Reset any missed count.
      missed_count[name] = 0
      # Rough heuristic to see if they're no longer a "new" driver.
      # If they're not new, then we use the USCF formula to determine k-factor:
      # https://en.wikipedia.org/wiki/Elo_rating_system#The_K-factor_used_by_the_USCF
      if counts[name] > (MIN_RACES * DRIVERS_PER_RACE):
        # k_scale is the multiplier (usually 1 for a race, 0.1 for quali)
        k_factor = k_scale * 800 / (counts[name] + len(ordering) - 1)
      else:
        k_factor = NEW_DRIVER_FACTOR
      # Adjust the k-factor upwards to make sure it doesn't go below any of the floors
      # we've established experimentally.
      if k_scale < 1:
        k_factor = max(k_factor, MIN_QUAL_FACTOR)
      else:
        k_factor = max(k_factor, MIN_RACE_FACTOR)
      # Update their rating based on their actual "wins" versus expected "wins".
      # https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details
      new_ratings[name] = rating.rating + k_factor * (actual_wins[name] - expected_wins[name])
      new_factors[name] = k_factor
      # Bump up the number of opponents by the number of finishers.
      counts[name] += (k_scale * len(ordering))
      # Now add to the total number of "after" points.
      after_points += new_ratings[name]
  # Drivers only get new ratings if they finished, so only update ratings of those drivers.
  # Also re-normalize so pointsIn==pointsOut
  ratio = before_points / after_points
  for name,new_rating in new_ratings.items():
    ratings[name].rating = new_rating * ratio
    ratings[name].k_factor = new_factors[name]


def IsValidRow(row):
  for header in ['Year', 'RaceID', 'RaceType', 'DriverID', 'Result']:
    if header not in row:
      return False
  return True


def ImportData(tsvfile, drivers, outdict):
  """Import races and drivers from a tab-separatef file.

  tsvfile: path to the tab-separated file
  drivers: set of driver IDs
  outdict: Output of results of the form {race: {place: driver}}
  """
  with open(tsvfile, 'r') as infile:
    # Year  Race    RaceType    DriverID    Result
    tsvreader = csv.DictReader(infile, delimiter='\t')
    for row in tsvreader:
      if not IsValidRow(row):
        continue
      year = int(row['Year'])
      race_id = int(row['RaceID'])
      # Race type Q: qualifying; R: race
      # Other types are non-championship races (e.g., the Indy 500 in the 50s)
      race_type = row['RaceType']
      if race_type != 'Q' and race_type != 'R':
        continue
      driver_id = row['DriverID']
      # Place can be numeric (e.g., "1") or not (e.g., "R" for "Retired").
      place = row['Result']
      try:
        place = int(place)
      except:
        # A non-numeric place; skip it
        continue
      # Race IDs are of the format
      # S${YEAR}E${EVENT_ID}D${TYPE}
      race_id = 'S%dE%04dD%s' % (year, race_id, race_type)
      drivers.add(driver_id)
      if race_id in outdict:
        outdict[race_id][place] = driver_id
      else:
        outdict[race_id] = dict({place: driver_id})


def SkipDriver(name, missed_count):
  return name in missed_count and missed_count[name] > MISS_COUNT


def PrintRatings(season, tag, ratings, counts, whitelist, missed_count,
                 tsvwriter):
  """Print the ratings and percentiles of all the active drivers.

  season: Year
  tag: Tag for this particular race
  ratings: map of driver ID to the DriverRating object
  counts: Number of opponents they've faced
  whitelist: whitelisted set of drivers (seeded at the start of the season)
  missed_count: number of consecutive events missed
  """
  # In order to get the percentiles on the normal distribution of a driver's rating,
  # we need to get both the mean and the stddev for driver ratings at this moment.
  sum = 0
  num = 0
  for name in whitelist:
    if SkipDriver(name, missed_count):
      continue
    sum += ratings[name].rating
    num += 1
  if num < 2:
    return
  avg = sum / num
  sos = 0
  for name in whitelist:
    if SkipDriver(name, missed_count):
      continue
    diff = (ratings[name].rating - avg)
    sos += (diff * diff)
  dev = (sos / (num - 1)) ** 0.5
  for name,rating in ratings.items():
    if name in whitelist:
      if SkipDriver(name, missed_count):
        continue
      ndevs = 0
      if dev != 0:
        ndevs = (rating.rating - avg) / dev
      tsvwriter.writerow(['Elo', tag, season, name, '%.3f' % rating.rating,
                          int(rating.k_factor)])
      tsvwriter.writerow(['ZScore', tag, season, name, '%.3f' % norm.cdf(ndevs),
                          int(rating.k_factor)])


def PrintHeader(tsvwriter):
  """Print the header row for the output.
  """
  tsvwriter.writerow(['Metric', 'EventID', 'Season', 'DriverID', 'Value',
                      'ValueExtra'])


def PartialRevertDriversToNew(ratings):
  """Revert each driver partially back to the rating of an new driver.
  """
  for name,rating in ratings.items():
    rating.rating = (rating.rating * (1 - YEAR_ADJ)) + INIT_RANKING * YEAR_ADJ


def RunAllRatings(drivers, races, outfile):
  """Run all the ratings and write the output and logging to a TSV.

  drivers: set of unique driver IDs
  races: dictionary where the key is the race ID and the value is a dict from
         place identifier to driver ID
  """
  # Initialize all the ratings and race participation counts
  driver_ratings = dict()
  driver_counts = dict()
  InitDrivers(drivers, driver_ratings, driver_counts)
  # Now actually run the ratings and dump data to an output file
  with open(outfile, 'w') as outf:
    csv.register_dialect('simple', delimiter='\t', lineterminator='\n')
    tsvwriter = csv.writer(outf, dialect='simple')
    last_season = 0
    sorted_races = sorted(races.keys())
    curr_drivers = set()
    missed_count = dict()
    race_count = 0
    # Go through the races in chronological order
    for race in sorted_races:
      # The season of the current race is not the same as the previous one.
      # Do a between-season reset.
      if last_season != int(race[1:5]):
        curr_drivers = set()
        SetDriversFromOrdering(races[race], curr_drivers)
        missed_count = dict()
        race_count = 0
        PartialRevertDriversToNew(driver_ratings)
      race_count += 1
      last_season = int(race[1:5])
      k_factor_scale = 1
      # The race ID will have a 'Q' at the end for qualifying, 'R' for race
      pos = race.rfind('Q')
      if pos > 0:
        k_factor_scale = 0.1
      # Print ratings before and after each session or race.
      # Append 'A' to the race ID to distinguish before-contest ratings, and 'Z'
      # to the race ID to signify after-contest rating.
      # This is done to ensure that sorted order will still work correctly.
      PrintRatings(last_season, race + 'A', driver_ratings, driver_counts, curr_drivers,
        missed_count, tsvwriter)
      OneRace(race, races[race], driver_ratings, driver_counts, curr_drivers,
        missed_count, k_factor_scale, tsvwriter)
      PrintRatings(last_season, race + 'Z', driver_ratings, driver_counts, curr_drivers,
        missed_count, tsvwriter)


def main(argv):
  if len(argv) != 3:
    print('Usage: %s <in:results_tsv> <out:ratings_tsv>' % (argv[0]))
    sys.exit(1)
  drivers = set()
  races = dict()
  ImportData(argv[1], drivers, races)
  RunAllRatings(drivers, races, argv[2])


if __name__ == "__main__":
  main(sys.argv)
