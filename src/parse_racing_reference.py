#!/usr/bin/python3

from html.parser import HTMLParser

import argparse
import csv
import re
import sys

_DRIVERS_HEADERS = ['driver_id', 'driver_name']
_EVENTS_HEADERS = ['event_id', 'season', 'stage', 'type', 'date', 'name', 'site', 'num_drivers', 'laps', 'lap_distance']
_RESULTS_HEADERS = [
    'event_id', 'driver_id', 'team_id', 'start_position', 'end_position', 'laps', 'status', 'dnf_category', 'num_racers'
]
_TEAMS_HEADERS = ['team_id', 'team_name']

_DRIVER_DNF_REASONS = (['crash', 'spin', 'damage', 'driver ill', 'spun off', 'crash damage', 'pit crash'])


def dnf_status(status):
    if status == 'running':
        return '-'
    elif status in _DRIVER_DNF_REASONS:
        return 'driver'
    if any(char.isdigit() for char in status):
        return '-'
    else:
        return 'car'


def create_argparser():
    argparser = argparse.ArgumentParser(description='Formula 1 rating parameters')
    argparser.add_argument('drivers_tsv', help='TSV file to write containing the list of drivers.',
                           type=argparse.FileType('w'))
    argparser.add_argument('events_tsv', help='TSV file to write containing the list of events.',
                           type=argparse.FileType('w'))
    argparser.add_argument('results_tsv', help='TSV file to write containing the list of results.',
                           type=argparse.FileType('w'))
    argparser.add_argument('teams_tsv', help='TSV file to write containing the list of teams.',
                           type=argparse.FileType('w'))
    return argparser


class RaceParser(HTMLParser):
    _TITLE_PATTERN = '^(\\d{2})/(\\d{2})/(\\d{4}) race: .*$'
    _DATE_LINK_PATTERN = '^.*/all.{0,1}dates/(\\d{2})(\\d{2})$'
    _YEAR_STAGE_PATTERN = '^.*/race[^/]*/(\\d{4})-0{0,1}(\\d{1,2})/.$'
    _RACE_ID_PATTERN = '^/race[^/]*/\\d{4}_(.*)/.$'
    _FINISHING_POS_TITLE = 'Fin'
    _STARTING_POS_TITLE = 'St'
    _DRIVER_TITLE = 'Driver'
    _TEAM_TITLE = 'Team'
    _PARTS_TITLE = 'Chass./Eng.'
    _LAPS_TITLE = 'Laps'
    _STATUS_TITLE = 'Status'

    def __init__(self, filename, events_tsv, results_tsv, drivers, teams):
        HTMLParser.__init__(self)
        self._filename = filename
        self._events_tsv = events_tsv
        self._results_tsv = results_tsv
        self._drivers = drivers
        self._teams = teams
        # Per-event placeholders
        self._year = None
        self._stage = None
        self._date = None
        self._race_id = None
        self._lap_distance = -1
        self._num_laps = 0
        self._site = 'TODO'
        self._in_title = False
        self._in_driver_table = False
        self._in_driver_header = False
        self._in_driver_row = False
        self._maybe_in_laps = False
        self._column_number = None
        self._current_idx = None
        self._start_idx = None
        self._finish_idx = None
        self._driver_idx = None
        self._team_idx = None
        self._parts_idx = None
        self._laps_idx = None
        self._status_idx = None
        # Per-driver placeholders
        self._start = None
        self._finish = None
        self._laps = None
        self._driver_name = None
        self._driver_id = None
        self._team_name = None
        self._team_id = None
        self._chassis = None
        self._status = None
        # Final rows which need augmenting
        self._data = list()
        self._driver_count = None

    def read_and_parse(self):
        with open(self._filename, 'r', encoding='ISO-8859-1') as infile:
            data = infile.read()
        self.feed(data)

    def handle_starttag(self, tag, attrs):
        if tag == 'title':
            self._in_title = True
        elif tag.endswith('like'):
            self.maybe_set_year(attrs)
        elif tag == 'table':
            self.maybe_in_driver_table(attrs)
        elif tag == 'tr':
            self.maybe_start_row(attrs)
        elif tag == 'td':
            self.maybe_get_driver_fact(attrs)
        elif tag == 'a':
            if self._date is None:
                self.get_date_from_link(attrs)
            if self._driver_count is None:
                self.maybe_get_race_id(attrs)
                self.maybe_get_track_id(attrs)
            elif self._in_driver_row:
                self.maybe_get_driver_id(attrs)
        elif tag == 'br':
            self._maybe_in_laps = True

    def handle_endtag(self, tag):
        if tag == 'title':
            self._in_title = False
        elif tag == 'table':
            if self._in_driver_table:
                self.print_data()
                self._in_driver_table = False
        elif tag == 'tr':
            if self._in_driver_row:
                self._in_driver_row = False
                self.add_row()
                self.reset_driver_data()
            if self._in_driver_header:
                self._in_driver_header = False
            self._column_number = 0
        elif tag == 'td':
            if self._in_driver_row:
                self._column_number += 1
        elif tag == 'br':
            self._maybe_in_laps = False

    def handle_data(self, data):
        if self._in_title:
            self.get_date_from_title(data)
        elif self._in_driver_header:
            self.maybe_get_column_header(data)
        elif self._in_driver_row:
            self.maybe_get_driver_fact(data)
        elif self._maybe_in_laps:
            self.maybe_get_lap_distance(data)

    def reset_driver_data(self):
        self._start = None
        self._finish = None
        self._laps = None
        self._driver_name = None
        self._driver_id = None
        self._team_name = None
        self._team_id = None
        self._chassis = None

    def get_date_from_title(self, data):
        # This is still working
        extract = re.search(self._TITLE_PATTERN, data)
        if not extract:
            return
        self._date = '%s-%s-%s' % (extract.group(3), extract.group(1), extract.group(2))

    def maybe_set_year(self, attrs):
        # This is still working
        for k, v in attrs:
            if k != 'href':
                continue
            extract = re.search(self._YEAR_STAGE_PATTERN, v)
            if extract:
                self._year = int(extract.group(1))
                self._stage = int(extract.group(2))

    def get_date_from_link(self, attrs):
        # This is now working
        if self._year is None:
            return
        for k, v in attrs:
            if k != 'href':
                continue
            extract = re.search(self._DATE_LINK_PATTERN, v)
            if extract:
                self._date = '%s-%s-%s' % (self._year, extract.group(1), extract.group(2))

    def maybe_in_driver_table(self, attrs):
        # This now works
        for tag, value in attrs:
            if tag != 'class':
                continue
            if value == 'tb' or value == 'tb race-results-tbl ':
                self._in_driver_table = True

    def maybe_start_row(self, attrs):
        # This still works
        if not self._in_driver_table:
            return
        for tag, value in attrs:
            if tag != 'class':
                continue
            if value == 'newhead':
                self._in_driver_header = True
            elif value == 'odd' or value == 'even':
                self._in_driver_row = True
        if self._in_driver_header or self._in_driver_row:
            self._column_number = 0
            if self._in_driver_header:
                self._driver_count = 0
        else:
            self._in_driver_table = False

    def maybe_get_column_header(self, data):
        # This still works
        if not self._in_driver_header:
            return
        if not len(str(data).strip('\n')):
            return
        if data == self._FINISHING_POS_TITLE:
            self._finish_idx = self._column_number
        elif data == self._STARTING_POS_TITLE:
            self._start_idx = self._column_number
        elif data == self._DRIVER_TITLE:
            self._driver_idx = self._column_number
        elif data == self._TEAM_TITLE:
            self._team_idx = self._column_number
        elif data == self._PARTS_TITLE:
            self._parts_idx = self._column_number
        elif data == self._LAPS_TITLE:
            self._laps_idx = self._column_number
        elif data == self._STATUS_TITLE:
            self._status_idx = self._column_number
        self._column_number += 1

    def maybe_get_driver_fact(self, data):
        # TODO: Test this
        if not self._in_driver_row or not isinstance(data, str):
            return
        if not data or re.match('^\\s*$', data):
            return
        if self._column_number == self._start_idx:
            self._start = data
#        elif self._column_number == self._finish_idx:
#            self._finish = data
        elif self._column_number == self._driver_idx:
            self._driver_name = data
        elif self._column_number == self._team_idx:
            self._team_name = data
            self._team_id = data.replace(' ', '_')
        elif self._column_number == self._parts_idx:
            self._chassis = data
        elif self._column_number == self._laps_idx:
            self._laps = data
            try:
                if int(data) > self._num_laps:
                    self._num_laps = int(data)
            finally:
                return
        elif self._column_number == self._status_idx:
            self._status = data

    def maybe_get_driver_id(self, attrs):
        # This now works
        if not self._in_driver_row or self._column_number != self._driver_idx:
            return
        for tag, value in attrs:
            if tag != 'href':
                continue
            extract = re.match('^.*/driver/([^/]*)/{0,1}.*$', value)
            if not extract:
                continue
            self._driver_id = extract.group(1)

    def maybe_get_race_id(self, attrs):
        # This now works
        race_id = None
        right_class = False
        for tag, value in attrs:
            if tag == 'class':
                if value != 'nodec':
                    return
                right_class = True
            if tag != 'href':
                continue
            extract = re.search(self._RACE_ID_PATTERN, value)
            if extract:
                race_id = extract.group(1)
        if race_id is None or not right_class:
            return
        self._race_id = race_id

    def maybe_get_track_id(self, attrs):
        # This is still good
        for tag, value in attrs:
            if tag != 'href':
                continue
            extract = re.match('^.*/tracks/(.*)$', value)
            if extract:
                self._site = extract.group(1).replace('_', ' ')

    def maybe_get_lap_distance(self, data):
        # This is still good
        extract = re.match('.* laps.*on a (.*) kilometer \\w+ (?:course|track).*', data)
        if not extract:
            return
        self._lap_distance = float(extract.group(1))

    def print_data(self):
        for row in self._data:
            row.append(self._driver_count)
            self._results_tsv.writerow(row)
        # Qualifying
        event_id = '%d-%02d-Q' % (self._year, self._stage)
        data = [event_id, self._year, self._stage, 'Q', self._date, self._race_id, self._site, self._driver_count,
                1, self._lap_distance
                ]
        self._events_tsv.writerow(data)
        # Race
        event_id = '%d-%02d-R' % (self._year, self._stage)
        data = [event_id, self._year, self._stage, 'R', self._date, self._race_id, self._site, self._driver_count,
                self._num_laps, self._lap_distance
                ]
        self._events_tsv.writerow(data)

    def add_row(self):
        # event_id, driver_id, team_id, start_position, end_position, laps, status, dnf_category
        if self._start is None:
            return
        self._driver_count += 1
        self._finish = self._driver_count
        self.add_driver(self._driver_id, self._driver_name)
        self.add_team(self._team_id, self._team_name)
        # Qualifying
        event_id = '%d-%02d-Q' % (self._year, self._stage)
        data = [event_id, self._driver_id, self._team_id, 0, self._start, 1, '-', '-']
        self._data.append(data)
        # Race
        event_id = '%d-%02d-R' % (self._year, self._stage)
        data = [event_id, self._driver_id, self._team_id, self._start, self._finish, self._laps, self._status,
                dnf_status(self._status)]
        self._data.append(data)

    def add_driver(self, driver_id, driver_name):
        if not driver_id or not driver_name:
            return
        existing_name = self._drivers.get(driver_id)
        if existing_name is None:
            self._drivers[driver_id] = driver_name
        elif existing_name != driver_name:
            print('ERROR: Mismatch for driver ID: "%s" vs "%s"' % (existing_name, driver_name))

    def add_team(self, team_id, team_name):
        if not team_id or not team_name:
            return
        existing_name = self._teams.get(team_id)
        if existing_name is None:
            self._teams[team_id] = team_name
        elif existing_name != team_name:
            print('ERROR: Mismatch for team ID: "%s" vs "%s"' % (existing_name, team_name))


def write_dict_to_file(headers, data_dict, handle):
    writer = csv.writer(handle, delimiter='\t')
    writer.writerow(headers)
    for key, value in data_dict.items():
        writer.writerow([key, value])


def main():
    parser = create_argparser()
    args = parser.parse_known_args()
    out_args = args[0]
    extra_args = args[1]
    if not extra_args:
        return 1
    all_drivers = dict()  # Mapping of driver ID to driver name
    all_teams = dict()    # Mapping of team ID to team name
    events_writer = csv.writer(out_args.events_tsv, delimiter='\t')
    events_writer.writerow(_EVENTS_HEADERS)
    results_writer = csv.writer(out_args.results_tsv, delimiter='\t')
    results_writer.writerow(_RESULTS_HEADERS)
    for f in sorted(extra_args):
        print(f)
        parser = RaceParser(f, events_writer, results_writer, all_drivers, all_teams)
        parser.read_and_parse()
    write_dict_to_file(_DRIVERS_HEADERS, all_drivers, out_args.drivers_tsv)
    write_dict_to_file(_TEAMS_HEADERS, all_teams, out_args.teams_tsv)
    return 0


if __name__ == '__main__':
    sys.exit(main())
