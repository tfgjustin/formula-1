import csv
import functools
import io

from collections import defaultdict
from copy import deepcopy
from driver import Driver
from entrant import Entrant
from event import Qualifying, Race, SprintQualifying, compare_events
from ratings import DriverReliability, EloRating, KFactor, Reliability
from result import Result
from season import Season
from team import TeamFactory


def _is_valid_row(row, headers):
    for header in headers:
        if header not in row or not row[header]:
            print('Missing header %s' % header)
            print(row)
            return False
    return True


class DataLoader(object):

    def __init__(self, args, base_output_filename):
        self._events = dict()
        self._future_events = dict()
        self._seasons = dict()
        self._future_seasons = dict()
        self._drivers = dict()
        self._results = list()
        self._grid_penalties = defaultdict(list)
        self._future_drivers = dict()
        self._future_teams = dict()
        self._args = args
        self._outfile = open(base_output_filename + '.loader', 'w') if base_output_filename is not None else None
        self._team_factory = TeamFactory(args)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._outfile is not None:
            self._outfile.close()

    def seasons(self):
        return self._seasons

    def future_seasons(self):
        return self._future_seasons

    def events(self):
        return self._events

    def drivers(self):
        return self._drivers

    def grid_penalties(self, event_id=None):
        if event_id is None:
            return self._grid_penalties
        return self._grid_penalties.get(event_id, [])

    def future_drivers(self):
        return self._future_drivers

    def results(self):
        return self._results

    def team_factory(self):
        return self._team_factory

    def future_teams(self):
        return self._future_teams

    def load_events(self, content):
        self._load_events_internal(content, 'Historical', self._events, self._seasons)

    def load_drivers(self, content):
        _HEADERS = ['driver_id', 'driver_name', 'birthday', 'birth_year']
        handle = io.StringIO(content)
        reader = csv.DictReader(handle, delimiter='\t')
        for row in reader:
            if not _is_valid_row(row, _HEADERS):
                continue
            self._drivers[row['driver_id']] = Driver(
                row['driver_id'], row['driver_name'],
                EloRating(
                    self._args.driver_elo_initial,
                    regress_rate=self._args.driver_elo_regress,
                    k_factor_regress_rate=self._args.driver_kfactor_regress
                ), row['birthday'], row['birth_year'])
        print('Loaded %d drivers' % (len(self._drivers)), file=self._outfile)

    def load_teams(self, content):
        _HEADERS = ['type', 'event_id', 'team_id', 'team_name']
        handle = io.StringIO(content)
        reader = csv.DictReader(handle, delimiter='\t')
        for row in reader:
            if not _is_valid_row(row, _HEADERS):
                continue
            self._team_factory.create_team(row['type'], row['event_id'], row['team_id'], row['team_name'],
                                           other_event_id=row.get('other_event_id', None),
                                           other_team_id=row.get('other_team_id', None))
        if not self._team_factory.finalize_create():
            print('ERROR: Could not load teams successfully.', file=self._outfile)
            return False
        else:
            print('Loaded %d teams' % (len(self._team_factory.teams())), file=self._outfile)
            return True

    def load_results(self, content):
        _HEADERS = ['event_id', 'driver_id', 'team_id', 'start_position', 'partial_position', 'end_position',
                    'laps', 'status', 'dnf_category', 'num_racers']
        all_rows = self._load_and_group_by_event(content, _HEADERS)
        self._team_factory.reset_current_teams()
        last_event_id = None
        # Iterate over all rows sorted by event ID
        for event_id in sorted(all_rows.keys(), key=functools.cmp_to_key(compare_events)):
            for row in all_rows[event_id]:
                event_id = row['event_id']
                if event_id.startswith('#'):
                    continue
                if last_event_id is None or event_id != last_event_id:
                    self._team_factory.update_for_event(event_id)
                    last_event_id = event_id
                self._process_one_result(row)
        self.identify_all_participants()
        print('Loaded %d results' % (len(self._results)), file=self._outfile)

    def update_drivers_from_ratings(self, content):
        """Load drivers from the log of a previous run.

        We need to have loaded drivers from elsewhere already in order to get things like birthdays.
        """
        _HEADERS = ['RaceID', 'DriverID', 'EloPost', 'KFEventsPost', 'KmSuccessPost', 'KmFailurePost',
                    'WearSuccessPost', 'WearFailurePost']
        if not self._drivers:
            print('ERROR: Have NOT already loaded drivers from previous source', file=self._outfile)
            return False
        all_rows = self._load_and_group_by_event(content, _HEADERS, event_id_tag='RaceID')
        seen_drivers = set()
        unseen_drivers = set(self._drivers.keys())
        set_reliability_metrics = False
        base_driver_reliability = DriverReliability()
        for event_id in sorted(all_rows.keys(), key=functools.cmp_to_key(compare_events), reverse=True):
            for row in all_rows[event_id]:
                event_id = row['RaceID'][1:]
                if event_id.startswith('#'):
                    continue
                driver_id = row['DriverID']
                if driver_id not in self._drivers:
                    print('ERROR: Driver %s is in ratings log but not database of drivers' % driver_id,
                          file=self._outfile)
                    continue
                if driver_id in seen_drivers:
                    # TODO: Add lookback data
                    self._drivers[driver_id].add_year_participation(event_id)
                    continue
                else:
                    unseen_drivers.remove(driver_id)
                seen_drivers.add(driver_id)
                if not set_reliability_metrics:
                    base_driver_reliability.update(float(row['KmSuccessPost']), float(row['KmFailurePost']),
                                                   float(row['WearSuccessPost']), float(row['WearFailurePost']))
                reliability = Reliability(km_success=float(row['KmSuccessPost']),
                                          km_failure=float(row['KmFailurePost']),
                                          wear_success=float(row['WearSuccessPost']),
                                          wear_failure=float(row['WearFailurePost']))
                k_factor = KFactor(num_events=float(row['KFEventsPost']))
                rating = EloRating(init_rating=float(row['EloPost']), reliability=reliability, k_factor=k_factor,
                                   last_event_id=event_id)
                self._drivers[driver_id].set_rating(rating)
            set_reliability_metrics = True
        if base_driver_reliability.km_success() > 0:
            for driver_id in unseen_drivers:
                self._drivers[driver_id].rating().set_reliability(deepcopy(base_driver_reliability))
        print('Loaded %d drivers from log' % (len(self._drivers)), file=self._outfile)
        return True

    def load_teams_from_ratings(self, content):
        """Load teams from the log of a previous run.
        """
        _HEADERS = ['RaceID', 'TeamUUID', 'TeamID', 'EloPost', 'KFEventsPost', 'KmSuccessPost', 'KmFailurePost',
                    'WearSuccessPost', 'WearFailurePost']
        all_rows = self._load_and_group_by_event(content, _HEADERS, event_id_tag='RaceID')
        seen_teams = set()
        for event_id in sorted(all_rows.keys(), key=functools.cmp_to_key(compare_events), reverse=True):
            for row in all_rows[event_id]:
                event_id = row['RaceID'][1:]
                if event_id.startswith('#'):
                    continue
                uuid = row['TeamUUID']
                if uuid in seen_teams:
                    # TODO: Add lookback data
                    continue
                seen_teams.add(uuid)
                canonical_team = self._team_factory.get_team_by_uuid(uuid)
                if canonical_team is None:
                    continue
                reliability = Reliability(km_success=float(row['KmSuccessPost']),
                                          km_failure=float(row['KmFailurePost']),
                                          wear_success=float(row['WearSuccessPost']),
                                          wear_failure=float(row['WearFailurePost']),
                                          wear_percent=self._args.wear_reliability_percent)
                k_factor = KFactor(num_events=float(row['KFEventsPost']))
                rating = EloRating(init_rating=float(row['EloPost']), reliability=reliability, k_factor=k_factor,
                                   last_event_id=event_id)
                canonical_team.set_rating(rating)
        print('Loaded %d teams from log' % (len(self._team_factory.teams())), file=self._outfile)
        return True

    def load_grid_penalties(self, content):
        _HEADERS = ['event_id', 'driver_id', 'num_places']
        handle = io.StringIO(content)
        reader = csv.DictReader(handle, delimiter='\t')
        is_valid_file = True
        for row in reader:
            if not _is_valid_row(row, _HEADERS):
                is_valid_file = False
                continue
            driver_id = row['driver_id']
            if driver_id not in self._drivers:
                print('ERROR: Driver %s is in grid penalties log but not database of drivers' % driver_id,
                      file=self._outfile)
                is_valid_file = False
                continue
            penalty = [driver_id, int(row['num_places'])]
            self._grid_penalties[row['event_id']].append(penalty)
        print('Loaded grid penalties for %d events' % len(self._grid_penalties), file=self._outfile)
        return is_valid_file

    def load_future_simulation_data(self):
        self.load_future_events(self._args.future_events_tsv)
        self.load_future_lineup(self._args.future_lineup_tsv)

    def load_future_events(self, content):
        if not content:
            return
        self._load_events_internal(content, 'Future', self._future_events, self._future_seasons)

    def load_future_lineup(self, filename):
        if filename:
            self._load_future_lineup_from_file(filename)
        else:
            self._get_future_lineup_from_last_event()

    def identify_all_participants(self):
        for season in self._seasons.values():
            season.identify_participants()

    @staticmethod
    def _load_and_group_by_event(content, headers, event_id_tag='event_id'):
        all_rows = defaultdict(list)
        handle = io.StringIO(content)
        reader = csv.DictReader(handle, delimiter='\t')
        for row in reader:
            if not _is_valid_row(row, headers):
                continue
            if row[event_id_tag].startswith('#'):
                continue
            all_rows[row[event_id_tag]].append(row)
        return all_rows

    def _process_one_result(self, row):
        start_position = None
        if 'start_position' in row and row['start_position']:
            start_position = row['start_position']
        team = self._team_factory.get_current_team(row['team_id'])
        if team is None:
            print('ERROR: Invalid team ID %s for event %s' % (row['team_id'], row['event_id']), file=self._outfile)
            return
        if row['event_id'] not in self._events:
            print('ERROR: Invalid event ID: %s' % row['event_id'], file=self._outfile)
            return
        event = self._events[row['event_id']]
        if row['driver_id'] not in self._drivers:
            print('ERROR: Invalid driver ID %s in event %s' % (row['driver_id'], row['event_id']),
                  file=self._outfile)
            return
        driver = self._drivers[row['driver_id']]
        entrant = Entrant(event, driver, team, start_position=start_position)
        result = Result(event, entrant, row['partial_position'], row['end_position'],
                        dnf_category=row['dnf_category'], laps_completed=row['laps'])
        entrant.set_result(result)
        self._results.append(result)
        self._events[event.id()].add_entrant(entrant)

    def _load_events_internal(self, contents, tag, event_log, season_log):
        _HEADERS = ['season', 'stage', 'date', 'name', 'type', 'event_id', 'laps', 'lap_distance', 'course_type',
                    'weather']
        handle = io.StringIO(contents)
        reader = csv.DictReader(handle, delimiter='\t')
        for row in reader:
            if not _is_valid_row(row, _HEADERS):
                continue
            if row['event_id'].startswith('#'):
                continue
            is_street_course = row['course_type'] == 'street'
            if row['type'] == 'Q':
                event = Qualifying(
                    row['event_id'], row['name'], row['season'], row['stage'], row['date'], row['laps'],
                    row['lap_distance'], is_street_course, row['weather'])
            elif row['type'] == 'S':
                event = SprintQualifying(
                    row['event_id'], row['name'], row['season'], row['stage'], row['date'], row['laps'],
                    row['lap_distance'], is_street_course, row['weather'])
            elif row['type'] == 'R':
                event = Race(
                    row['event_id'], row['name'], row['season'], row['stage'], row['date'], row['laps'],
                    row['lap_distance'], is_street_course, row['weather'])
            else:
                continue
            event_log[event.id()] = event
            if event.season() in season_log:
                season_log[event.season()].add_event(event)
            else:
                season = Season(event.season())
                season.add_event(event)
                season_log[event.season()] = season
        print('Loaded %d %s events' % (len(event_log), tag.lower()), file=self._outfile)
        print('Loaded %d %s seasons' % (len(season_log), tag.lower()), file=self._outfile)
        for year in sorted(season_log.keys()):
            season = season_log[year]
            num_events = len(season.events())
            print('%s Season: %4d #Events: %2d' % (tag, season.year(), num_events), file=self._outfile)

    def _load_future_lineup_from_file(self, filename):
        _HEADERS = ['driver_id', 'team_id']
        count = 0
        with open(filename, 'r') as infile:
            reader = csv.DictReader(infile, delimiter='\t')
            for row in reader:
                if not _is_valid_row(row, _HEADERS):
                    continue
                driver = self._add_driver_to_future_lineup_by_id(row['driver_id'])
                team = self._add_team_to_future_lineup_by_id(row['team_id'])
                if driver is None or team is None:
                    continue
                for event in self._future_events.values():
                    entrant = Entrant(event, driver, team)
                    event.add_entrant(entrant)
                count += 1
        print('Loaded %d future entrants from %s' % (count, filename), file=self._outfile)

    def _get_future_lineup_from_last_event(self):
        last_event_id = sorted(self._events.keys(),  key=functools.cmp_to_key(compare_events))[-1]
        last_event = self._events[last_event_id]
        count = 0
        for entrant in last_event.entrants():
            driver = self._add_driver_to_future_lineup_by_id(entrant.driver().id())
            team = self._add_team_to_future_lineup_by_id(entrant.team().id())
            if driver is None or team is None:
                continue
            for event in self._future_events.values():
                entrant = Entrant(event, driver, team)
                event.add_entrant(entrant)
            count += 1
        print('Reused %d future entrants from %s' % (count, last_event.id()), file=self._outfile)

    def _add_driver_to_future_lineup_by_id(self, driver_id):
        if driver_id not in self._drivers:
            print('ERROR: Invalid driver ID %s for future lineup' % (driver_id), file=self._outfile)
            return None
        driver = deepcopy(self._drivers[driver_id])
        self._disable_reliability_decay(driver.rating())
        self._future_drivers[driver_id] = driver
        return driver

    def _add_team_to_future_lineup_by_id(self, team_id):
        team = self._team_factory.get_current_team(team_id)
        if team is None:
            print('ERROR: Invalid team ID %s for future lineup' % (team_id), file=self._outfile)
            return None
        if team.id() in self._future_teams:
            return self._future_teams[team.id()]
        team = deepcopy(team)
        self._disable_reliability_decay(team.rating())
        self._future_teams[team.id()] = team
        return team

    @staticmethod
    def _disable_reliability_decay(rating):
        if rating.reliability() is None:
            return
        rating.reliability().set_decay_rate(1.0)
