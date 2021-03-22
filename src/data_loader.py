import csv
import io

from driver import Driver
from event import Qualifying
from event import Race
from ratings import EloRating
from result import Result
from season import Season
from team import TeamFactory


def _is_valid_row(row, headers):
    for header in headers:
        if header not in row or not row[header]:
            print('Missing header %s' % header)
            return False
    return True


class DataLoader(object):

    def __init__(self, args, base_filename):
        self._events = dict()
        self._seasons = dict()
        self._drivers = dict()
        self._results = list()
        self._args = args
        self._outfile = open(base_filename + '.loader', 'w')
        self._team_factory = TeamFactory(args)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._outfile.close()

    def seasons(self):
        return self._seasons

    def events(self):
        return self._events

    def drivers(self):
        return self._drivers

    def results(self):
        return self._results

    def team_factory(self):
        return self._team_factory

    def load_events(self, contents):
        _HEADERS = ['season', 'stage', 'date', 'name', 'type', 'event_id', 'laps', 'lap_distance']
        handle = io.StringIO(contents)
        reader = csv.DictReader(handle, delimiter='\t')
        for row in reader:
            if not _is_valid_row(row, _HEADERS):
                continue
            if row['type'] == 'Q':
                event = Qualifying(
                    row['event_id'], row['name'], row['season'], row['stage'], row['date'], row['laps'],
                    row['lap_distance'])
            elif row['type'] == 'R':
                event = Race(
                    row['event_id'], row['name'], row['season'], row['stage'], row['date'], row['laps'],
                    row['lap_distance'])
            else:
                continue
            self._events[event.id()] = event
            if event.season() in self._seasons:
                self._seasons[event.season()].add_event(event)
            else:
                season = Season(event.season())
                season.add_event(event)
                self._seasons[event.season()] = season
        print('Loaded %d events' % (len(self._events)), file=self._outfile)
        print('Loaded %d seasons' % (len(self._seasons)), file=self._outfile)
        for year in sorted(self._seasons.keys()):
            season = self._seasons[year]
            num_events = len(season.events())
            print('Season: %4d #Events: %2d' % (season.year(), num_events), file=self._outfile)

    def load_drivers(self, content):
        _HEADERS = ['driver_id', 'driver_name']
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
                ))
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
        _HEADERS = ['event_id', 'driver_id', 'end_position', 'num_racers', 'team_id', 'start_position', 'dnf_category',
                    'laps']
        all_rows = list()
        handle = io.StringIO(content)
        reader = csv.DictReader(handle, delimiter='\t')
        for row in reader:
            if not _is_valid_row(row, _HEADERS):
                continue
            all_rows.append(row)
        # Iterate over all rows sorted by event ID
        self._team_factory.reset_current_teams()
        last_event_id = None
        for row in sorted(all_rows, key=lambda r: r['event_id']):
            event_id = row['event_id']
            if last_event_id is None or event_id != last_event_id:
                self._team_factory.update_for_event(event_id)
                last_event_id = event_id
            start_position = None
            if 'start_position' in row and row['start_position']:
                start_position = row['start_position']
            team = self._team_factory.get_current_team(row['team_id'])
            if team is None:
                print('ERROR: Invalid team ID %s for event %s' % (row['team_id'], row['event_id']), file=self._outfile)
                continue
            if row['event_id'] not in self._events:
                print('ERROR: Invalid event ID: %s' % row['event_id'], file=self._outfile)
                continue
            event = self._events[row['event_id']]
            if row['driver_id'] not in self._drivers:
                print('ERROR: Invalid driver ID %s in event %s' % (row['driver_id'], row['event_id']),
                      file=self._outfile)
                continue
            driver = self._drivers[row['driver_id']]
            result = Result(event, driver, row['end_position'],
                            team=team, start_position=start_position,
                            num_racers=row['num_racers'], dnf_category=row['dnf_category'], laps=row['laps'])
            self._results.append(result)
            self._events[event.id()].add_result(result)
        self.identify_all_participants()
        print('Loaded %d results' % (len(self._results)), file=self._outfile)

    def identify_all_participants(self):
        for season in self._seasons.values():
            season.identify_participants()
