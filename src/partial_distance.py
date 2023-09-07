#!/usr/bin/python

import copy
import csv
import math
import sys
import unicodedata

from collections import defaultdict
from datetime import datetime

_OPENING_HEADER = 'opening_position'
_PARTIAL_HEADER = 'partial_position'


def parse_drivers(filename, normalized_filename, drivers):
    # [birthday] = (driver_id0, driver_id1, ..., driver_idN)
    normalized_drivers = defaultdict(set)
    with open(normalized_filename, 'r') as infile:
        reader = csv.DictReader(infile, delimiter='\t')
        for row in reader:
            dob = row.get('birthday')
            normalized_drivers[dob].add(row.get('driver_id'))
    with open(filename, 'r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            driver_id = row.get('driverId')
            if driver_id is None:
                continue
            surname = row.get('surname')
            driver_ref = row.get('driverRef')
            dob = row.get('dob')
            if surname is None or driver_ref is None or dob is None:
                continue
            if dob not in normalized_drivers:
                continue
            surname = unicodedata.normalize('NFKD', surname).encode('ascii', 'ignore').decode()
            surname = surname.replace(' ', '_').lower()
            for normalized_driver_id in normalized_drivers[dob]:
                if driver_ref in normalized_driver_id.lower() or surname in normalized_driver_id.lower():
                    drivers[driver_id] = normalized_driver_id
    print('Loaded info on %d drivers' % len(drivers))


def parse_races(filename, races):
    dt = datetime.today()
    today = dt.strftime('%Y-%m-%d')
    with open(filename, 'r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            # raceId,year,round
            race_id = row.get('raceId')
            season = row.get('year')
            stage = row.get('round')
            race_date = row.get('date')
            if race_id is None or season is None or stage is None or race_date is None:
                continue
            if int(season) < 2000 or race_date > today:
                continue
            races[race_id] = '%s-%02d-RA' % (season, int(stage))
    print('Loaded info on %d races' % len(races))


def parse_laps(filename, races, drivers, ordered_laps, max_laps):
    # Input:
    #   raceId,driverId,lap,position,time,milliseconds
    #   841,20,1,1,"1:38.109",98109
    # Temporary:
    #   [race_id][driver_id][lap_num] = lap_row
    tmp_laps = dict()
    with open(filename, 'r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            race_id = row.get('raceId')
            if race_id not in races:
                continue
            driver_id = row.get('driverId')
            if driver_id not in drivers:
                print('ERROR: Unknown driver %s in race %s' % (driver_id, race_id), file=sys.stderr)
                continue
            lap_num = row.get('lap')
            try:
                lap_num = int(lap_num)
            except ValueError:
                continue
            if lap_num > max_laps.get(race_id, 0):
                max_laps[race_id] = lap_num
            if 'time' not in row or 'milliseconds' not in row:
                continue
            if race_id not in tmp_laps:
                tmp_laps[race_id] = defaultdict(dict)
            tmp_laps[race_id][driver_id][lap_num] = row
    # Return:
    #   [race_id][cumulative_time] = lap_row
    for race_id, driver_data in tmp_laps.items():
        ordered_laps[race_id] = defaultdict(list)
        for driver_id, lap_rows in driver_data.items():
            total_msec = 0
            for _, lap_data in sorted(lap_rows.items()):
                try:
                    lap_msec = int(lap_data.get('milliseconds'))
                except ValueError:
                    print(lap_data)
                    continue
                total_msec += lap_msec
                ordered_laps[race_id][total_msec].append(lap_data)
    print('Loaded lap data on %d races' % len(ordered_laps))


def last_time_by_driver(race_laps, driver_to_last_time):
    for time_msec, lap_rows in sorted(race_laps.items(), reverse=True):
        for lap_row in lap_rows:
            driver_id = lap_row.get('driverId')
            if driver_id not in driver_to_last_time:
                driver_to_last_time[driver_id] = time_msec


def get_order_at_distance(races, drivers, lap_counts, ordered_laps, distance_metric, order_at_distance):
    for race_id, num_laps in lap_counts.items():
        tfg_race_id = races.get(race_id)
        if tfg_race_id is None:
            continue
        min_race_laps = distance_metric
        if min_race_laps < 1:
            min_race_laps = math.floor(distance_metric * num_laps)
        race_laps = ordered_laps.get(race_id)
        driver_to_last_time = dict()
        last_time_by_driver(race_laps, driver_to_last_time)
        distance_met_at = None
        # [lap][time_msec] = lap_row
        partial_ordering = defaultdict(dict)
        seen_drivers = set()
        for time_msec, lap_rows in sorted(race_laps.items()):
            # raceId,driverId,lap,position,time,milliseconds
            # 841,20,1,1,"1:38.109",98109
            if distance_met_at is None:
                for lap_row in lap_rows:
                    position = lap_row.get('position')
                    if int(position) != 1:
                        continue
                    lap_num = int(lap_row.get('lap'))
                    if lap_num != min_race_laps:
                        continue
                    distance_met_at = time_msec
                    driver_id = lap_row.get('driverId')
                    seen_drivers.add(driver_id)
                    partial_ordering[lap_num][time_msec] = lap_row
            else:
                for lap_row in lap_rows:
                    driver_id = lap_row.get('driverId')
                    if driver_id in seen_drivers:
                        continue
                    seen_drivers.add(driver_id)
                    lap_num = int(lap_row.get('lap'))
                    partial_ordering[lap_num][time_msec] = lap_row
        pos = 1
        for lap_num, lap_order in sorted(partial_ordering.items(), reverse=True):
            for time_msec, lap_row in sorted(lap_order.items()):
                driver_id = lap_row.get('driverId')
                driver_info = drivers.get(driver_id)
                order_at_distance[tfg_race_id][driver_info] = pos
                pos += 1


def insert_partial_field_name(tag, fieldnames):
    # If the column is already in the file, don't add it.
    if tag in fieldnames:
        return fieldnames
    f = copy.copy(fieldnames)
    end_idx = f.index('start_position')
    f.insert(end_idx + 1, tag)
    return f


def insert_partial_order(results_filename, order_at_distances, outfilename):
    with open(results_filename, 'r') as infile:
        reader = csv.DictReader(infile, delimiter='\t')
        fieldnames = reader.fieldnames
        for tag in sorted(order_at_distances, reverse=True):
            fieldnames = insert_partial_field_name(tag, fieldnames)
        with open(outfilename, 'w') as outfile:
            writer = csv.DictWriter(outfile, delimiter='\t', fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                for tag, distances in sorted(order_at_distances.items()):
                    row[tag] = '-'
                    event_id = row.get('event_id')
                    if event_id in distances:
                        driver_id = row.get('driver_id')
                        if driver_id in distances[event_id]:
                            row[tag] = distances[event_id][driver_id]
                writer.writerow(row)


def main(argv):
    if len(argv) != 7:
        print(
            (
                'Usage: %s <drivers_csv> <normalized_drivers_tsv> <races_csv> <laptimes_csv> <results_tsv> <outfile>'
            ) % argv[0]
        )
        return 1
    # [driver_id] = [lastName, birthday]
    drivers = dict()
    parse_drivers(argv[1], argv[2], drivers)
    races = dict()
    parse_races(argv[3], races)
    ordered_laps = dict()
    max_laps = dict()
    parse_laps(argv[4], races, drivers, ordered_laps, max_laps)
    # [race_id][driver_id] = position
    distances = {_PARTIAL_HEADER: 0.9, _OPENING_HEADER: 2}
    for tag, distance_metric in sorted(distances.items()):
        order_at_distance = defaultdict(dict)
        get_order_at_distance(races, drivers, max_laps, ordered_laps, distance_metric, order_at_distance)
        distances[tag] = order_at_distance
    insert_partial_order(argv[5], distances, argv[6])
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
