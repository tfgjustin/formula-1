#!/usr/bin/python3

from collections import defaultdict
from numpy import mean
from scipy.stats import hmean

import csv
import sys


_COMMON_NGRAMS = set(['Team', 'Racing', 'F1_Team', '38'])


def has_valid_headers(row, headers):
    for h in headers:
        if h not in row or not row[h]:
            return False
    return True


def is_aggregate_team(row):
    team_uuid = row['TeamUUID']
    if team_uuid == 'TeamNew' or team_uuid == 'TeamBase':
        return True
    return False


def import_teams(tsv_filename, teams, events):
    """Creates a mapping from (team UUID, race ID) to the team ID.
    """
    _HEADERS = ['RaceID', 'TeamUUID', 'TeamID']
    with open(tsv_filename, 'r') as infile:
        tsv_reader = csv.DictReader(infile, delimiter='\t')
        for row in tsv_reader:
            if not has_valid_headers(row, _HEADERS):
                continue
            if is_aggregate_team(row):
                continue
            race_id = row['RaceID']
            teams[row['TeamUUID']][race_id] = row['TeamID']
            parts = race_id.split('-')
            if len(parts) != 3:
                continue
            if parts[-1] == 'Q':
                continue
            events[parts[0]][parts[1]] = race_id


def import_races(tsv_filename, races):
    """Creates a mapping for season ~> round ~> race ID

    Input row:
        S1950E0043DR  1950  1 1950-05-13  1950 British Grand Prix
    Output value:
        [1950][1] = S1950E0043DR
    """
    with open(tsv_filename, 'r') as infile:
        tsv_reader = csv.DictReader(infile, delimiter='\t')
        for row in tsv_reader:
            races[row['season']][row['stage']] = row['event_id']


def load_flat_data(filename, metrics, data):
    """Loads flattened data.

    Format:
        event_id	TeamUUID    TeamID  NumTeams    [Metric0]Pre	[Metric0]Post

    Output dictionary:
        [team_uuid][year][event_id][metric] = value
    """
    _HEADERS = ['RaceID', 'TeamUUID']
    with open(filename, 'r') as infile:
        tsv_reader = csv.DictReader(infile, delimiter='\t')
        metrics.update([x[:-4] for x in tsv_reader.fieldnames if x.endswith('Post')])
        _HEADERS.extend([metric + 'Post' for metric in metrics])
        for row in tsv_reader:
            if not has_valid_headers(row, _HEADERS):
                continue
            if is_aggregate_team(row):
                continue
            team_uuid = row['TeamUUID']
            year = row['RaceID'][1:5]
            event_id = row['RaceID']
            for metric in metrics:
                data[team_uuid][year][event_id][metric] = float(row[metric + 'Post'])


def min_pct_races(year):
    """The minimum pct of events a driver has to participate in to get a rating.
    """
    y = int(year)
    if y < 1980:
        return 0.4
    elif y < 2000:
        return 0.6
    else:
        return 0.7


def longest_common_substring(s1, s2):
    m = [[0] * (1 + len(s2)) for _ in range(1 + len(s1))]
    longest, x_longest = 0, 0
    for x in range(1, 1 + len(s1)):
        for y in range(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0
    return s1[x_longest - longest: x_longest]


def common_team_name_inner(uuid, team_dict):
    names = list(team_dict.keys())
    common_name = names[0]
    for name in names[1:]:
        common_name = longest_common_substring(common_name, name)
        if not common_name or common_name == '_':
            break
    common_name = common_name.strip('_')
    if common_name and len(common_name) >= 4 and common_name not in _COMMON_NGRAMS:
        return common_name
    # print('Trying common name for [ %s ]' % ', '.join(names))
    often_used = max(team_dict.values())
    most_used_name = [name for name, count in team_dict.items() if count == often_used]
    if len(most_used_name) == 1 and most_used_name[0] not in _COMMON_NGRAMS:
        # print('Going with %s' % most_used_name[0])
        return most_used_name[0]
    elif len(most_used_name) != len(names):
        temp_dict = dict({name: team_dict[name] for name in most_used_name})
        return common_team_name(uuid, temp_dict)
    else:
        longest = max([len(name) for name in most_used_name])
        longest_names = [name for name in most_used_name if len(name) == longest]
        if len(longest_names) == 1:
            return longest_names[0]
        else:
            # print('ERROR: Team %s has %d usages of [ %s ]' % (uuid, often_used, ', '.join(most_used_name)))
            return longest_names[0]


def common_team_name(uuid, team_dict):
    return common_team_name_inner(uuid, team_dict).replace('_', ' ')


def print_one_year_peak(data, metrics, races, teams, tsv_writer):
    """Print the peak value a team had in a year.
    """
    # [team_uuid][year][event_id][metric] = value
    for team_uuid, year_dict in data.items():
        for year, race_dict in year_dict.items():
            if len(race_dict) < (len(races[year]) * min_pct_races(year)):
                continue
            if len(race_dict) <= 2:
                continue
            names = defaultdict(int)
            for event_id in race_dict.keys():
                team_id = teams[team_uuid][event_id]
                names[team_id] += 1
            team_name = common_team_name(team_uuid, names)
            for metric in metrics:
                max_val = max([x[metric] for x in race_dict.values()])
                tsv_writer.writerow(['Peak', metric, team_uuid, team_name, year, year, '%.4f' % max_val])


def print_one_year_overall(data, metrics, races, teams, tsv_writer):
    """Calculates the harmonic mean of the season peak, average, and end rating.
    """
    for team_uuid, year_dict in data.items():
        for year, race_dict in year_dict.items():
            if len(race_dict) < (len(races[year]) * min_pct_races(year)):
                continue
            sorted_races = sorted(race_dict.keys())
            names = defaultdict(int)
            for event_id in sorted_races:
                team_id = teams[team_uuid][event_id]
                names[team_id] += 1
            team_name = common_team_name(team_uuid, names)
            for metric in metrics:
                ratings = [x[metric] for x in race_dict.values()]
                min_val = min(ratings)
                if min_val < 0:
                    continue
                max_val = max(ratings)
                season_mean = mean(ratings)
                season_end = race_dict[sorted_races[-1]][metric]
                overall = hmean([max_val, season_mean, season_end])
                tsv_writer.writerow(['Overall', metric, team_uuid, team_name, year, year, '%.4f' % overall])


def print_n_years_average(data, metrics, races, teams, tag, num_years, tsv_writer):
    """Calculates the average rating over an N-year window.
    """
    for team_uuid, year_dict in data.items():
        for start_year in sorted(year_dict.keys()):
            num_completed = 0
            num_required = 0
            end_year = int(start_year) + num_years - 1
            if str(end_year) not in year_dict:
                continue
            values = []
            skip = False
            names = defaultdict(int)
            for y in range(int(start_year), end_year + 1):
                year = str(y)
                if year not in year_dict:
                    skip = True
                    break
                # Championship races are initially all the races in the year where the
                # driver received a rating.
                champ_races = {race_id: rating for race_id, rating in year_dict[year].items()}
                num_completed += len(champ_races)
                num_required += (len(races[year]) * min_pct_races(year))
                # print('%s\t%s\t%d\t%d\t%.1f' % (driver, year, len(champ_races), num_completed, num_required))
                if len(champ_races) <= 4:
                    skip = True
                    break
                for event_id in champ_races.keys():
                    team_id = teams[team_uuid][event_id]
                    names[team_id] += 1
                values.extend(champ_races.values())
            if not values or skip or num_completed < num_required:
                continue
            team_name = common_team_name(team_uuid, names)
            for metric in metrics:
                tsv_writer.writerow(
                    [tag, metric, team_uuid, team_name, start_year, end_year,
                     '%.4f' % mean([v[metric] for v in values])]
                )


def print_headers(tsv_writer):
    tsv_writer.writerow(['Aggregate', 'Metric', 'TeamUUID', 'CommonID', 'CommonName', 'StartYear', 'EndYear', 'Value'])


def main(argv):
    nested_dict = lambda: defaultdict(nested_dict)
    if len(argv) != 4:
        print('Usage: %s <in:races_tsv> <in:flat_ratings_tsv> <out:metrics_tsv>' % (argv[0]))
        sys.exit(1)
    teams = defaultdict(dict)
    events = defaultdict(dict)
    import_teams(argv[2], teams, events)
    races = nested_dict()
    import_races(argv[1], races)

    metrics = set()
    data = nested_dict()
    load_flat_data(argv[2], metrics, data)

    with open(argv[3], 'w') as outfile:
        tsv_writer = csv.writer(outfile, delimiter='\t')
        print_headers(tsv_writer)
        print_one_year_peak(data, metrics, races, teams, tsv_writer)
        print_one_year_overall(data, metrics, races, teams, tsv_writer)
        for num_years in [1, 3, 5, 7]:
            tag = 'Range%dy' % num_years
            print_n_years_average(data, metrics, races, teams, tag, num_years, tsv_writer)


if __name__ == "__main__":
    main(sys.argv)
