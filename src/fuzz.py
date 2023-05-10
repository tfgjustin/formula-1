import csv
import event as f1event
import functools
import io
import numpy as np

from collections import defaultdict
from numpy.random import normal


def avg_elo_age_experience(age, experience):
    """Formula to project out the average Elo rating of a driver based on age and experience."""
    age -= 20
    age_2 = age ** 2
    exp_2 = experience ** 2
    return -17.11 + (-2.84 * age) + (-0.200 * age_2) + (30.30 * experience) + (-1.14 * exp_2)


def is_valid_row(row, headers):
    for header in headers:
        if header not in row or not row[header]:
            print('Missing header %s' % header)
            print(row)
            return False
    return True


def target_elo_delta_left(target_elo_delta, current_stage_number, total_stages):
    per_stage_delta = target_elo_delta / total_stages
    return per_stage_delta * (total_stages - current_stage_number + 1)


class Fuzzer(object):

    def __init__(self, args, logfile):
        self._args = args
        self._logfile = logfile
        # All fuzz for the simulations (if any)
        # [entity_id][event_id][sim_id] = diff_amount
        self._all_fuzz = defaultdict(dict)

    def all_fuzz(self):
        return self._all_fuzz

    def generate_all_fuzz(self, year, events, drivers, teams):
        if self._logfile is not None:
            print('Generating fuzz: start (%d events %d drivers %d teams)' % (len(events), len(drivers), len(teams)),
                  file=self._logfile)
        event_ids = sorted(events.keys(),  key=functools.cmp_to_key(f1event.compare_events))
        current_stage_number = events[event_ids[0]].stage()
        season_total_stages = events[event_ids[-1]].stage()
        # Generate the base team fuzz first, then the per-event fuzz.
        self.generate_fuzz_team_estimate(events, current_stage_number, season_total_stages)
        self.generate_fuzz_team_event(events, teams)
        # Generate the base age fuzz first, and then the lookback fuzz.
        self.generate_fuzz_driver_age(year, events, current_stage_number, season_total_stages, drivers)
        self.generate_fuzz_driver_event(events, drivers)
        if self._logfile is not None:
            print('Generating fuzz: finish', file=self._logfile)

    def generate_fuzz_team_estimate(self, events, current_stage_number, season_total_stages):
        # team_id: name-ish identifier of the team
        # elo_adjust: adjustment of the team's rating, in raw Elo
        # mode: either 'current' or 'target'
        all_teams = set()
        current_fuzz = dict()
        target_fuzz = dict()
        _HEADERS = ['team_id', 'elo_adjust', 'mode']
        handle = io.StringIO(self._args.team_adjust_tsv)
        reader = csv.DictReader(handle, delimiter='\t')
        for row in reader:
            if not is_valid_row(row, _HEADERS):
                continue
            if row['mode'] != 'current' and row['mode'] != 'target':
                print('ERROR: Invalid team adjustment fuzz mode: %s' % row['mode'])
                continue
            all_teams.add(row['team_id'])
            if row['mode'] == 'current':
                current_fuzz[row['team_id']] = float(row['elo_adjust'])
            elif row['mode'] == 'target':
                target_fuzz[row['team_id']] = float(row['elo_adjust'])
        stages_left = season_total_stages - current_stage_number + 1
        for team_id in all_teams:
            current_elo_diff = current_fuzz.get(team_id, 0)
            target_elo_diff = target_fuzz.get(team_id, 0) - current_elo_diff
            elo_stddev = max([abs(current_elo_diff), abs(target_elo_diff)])
            elo_stddev = max([min([40, abs(elo_stddev)]), 20])
            team_fuzz = defaultdict(list)
            now_samples = normal(current_elo_diff, elo_stddev, self._args.num_iterations)
            target_samples = normal(target_elo_diff, elo_stddev, self._args.num_iterations)
            for idx in range(self._args.num_iterations):
                now_elo_diff = now_samples[idx]
                per_stage_elo_diff = target_samples[idx] / stages_left
                for event in events.values():
                    event_diff = now_elo_diff + ((event.stage() - current_stage_number) * per_stage_elo_diff)
                    team_fuzz[event.id()].append(event_diff)
            self._all_fuzz[team_id] = team_fuzz

    def generate_fuzz_team_event(self, events, teams):
        team_qualifying_fuzz = self.generate_fuzz_entity_event_type(teams, f1event.QUALIFYING)
        team_race_fuzz = self.generate_fuzz_entity_event_type(teams, f1event.RACE)
        # All fuzz for the simulations (if any)
        # [entity_id][event_id][sim_id] = diff_amount
        for team_id in teams.keys():
            if team_id not in self._all_fuzz:
                print('ERROR: Have not generated base team estimate fuzz for %s' % team_id)
                continue
            for event in events.values():
                if event.id() not in self._all_fuzz[team_id]:
                    print('ERROR: No base team estimate fuzz for %s in %s' % (team_id, event.id()))
                    continue
                if len(self._all_fuzz[team_id][event.id()]) != self._args.num_iterations:
                    print('ERROR: Mismatched amount of base fuzz for %s in %s' % (team_id, event.id()))
                    continue
                fuzz_dict = team_race_fuzz
                default_dist = [0, 30]
                if event.type() in [f1event.QUALIFYING, f1event.SPRINT_SHOOTOUT]:
                    fuzz_dict = team_qualifying_fuzz
                dist = fuzz_dict.get(team_id, default_dist)
                samples = normal(dist[0], dist[1], self._args.num_iterations)
                for idx in range(self._args.num_iterations):
                    self._all_fuzz[team_id][event.id()][idx] += samples[idx]

    def generate_fuzz_driver_age(self, year, events, current_stage_number, season_total_stages, drivers):
        stages_left = season_total_stages - current_stage_number + 1
        for driver in drivers.values():
            seasons = driver.seasons()
            birth_year = driver.birth_year()
            current_age = year - birth_year
            current_exp = len(seasons)
            # The index of the previous season in which they competed
            non_current_idx = 0
            if current_exp and seasons[0] == year:
                # If this driver does have experience, but part of that experience includes this year,
                # then remove that year (i.e., they haven't completed this year so it doesn't count)
                current_exp -= 1
                # Also bump back the index one slot because the head of the list is the current year.
                non_current_idx = 1
            # Start with the current delta.
            current_elo_diff = 0
            target_elo_diff = avg_elo_age_experience(current_age, current_exp)
            if current_exp:
                # This is not their rookie year. Subtract out their previous experience.
                previous_year = seasons[non_current_idx]
                target_elo_diff -= avg_elo_age_experience(previous_year - birth_year, current_exp - 1)
            else:
                # This is their rookie year. So instead of a "start where they are now and move to 2*EOY delta" slope,
                # we assume "start at EOY and have zero slope".
                current_elo_diff = target_elo_diff
                target_elo_diff = 0
            # The fuzz represents the change in the average rating from year to year.
            # We'll assume the change is linear; the delta at race=1 is current_elo_diff and race=N is 2*target_elo_diff
            target_elo_diff *= 2
            # Now we need to figure out how much is left of the target. E.g., if we expect the overall target to be +48
            # Elo but we've already finished 4 out of 24 rounds, then we'd expect there to only be +40 Elo delta left.
            target_elo_diff = target_elo_delta_left(target_elo_diff, current_stage_number, season_total_stages)
            if self._logfile is not None:
                print('DriverFuzz %d %d %d %s %6.2f %6.2f' % (year, birth_year, year - birth_year, driver.id(),
                                                              current_elo_diff, target_elo_diff),
                      file=self._logfile)
            elo_stddev = max([abs(current_elo_diff), abs(target_elo_diff)])
            elo_stddev = max([min([25, abs(elo_stddev)]), 10])
            driver_fuzz = defaultdict(list)
            now_samples = normal(current_elo_diff, elo_stddev, self._args.num_iterations)
            target_per_stage_samples = normal(target_elo_diff, elo_stddev, self._args.num_iterations)
            for idx in range(self._args.num_iterations):
                # First fuzz both the current and target diffs
                now_elo_diff = now_samples[idx]
                target_per_stage_elo_diff = target_per_stage_samples[idx] / stages_left
                for event in events.values():
                    # Note that because the age/experience curve starts at round=1 and ends at round=Final, we don't
                    # care about the where this particular event is in relation to the most recent event in real life,
                    # just where it is in relation to the start and the end of the year.
                    event_diff = now_elo_diff + ((event.stage() - current_stage_number) * target_per_stage_elo_diff)
                    driver_fuzz[event.id()].append(event_diff)
            self._all_fuzz[driver.id()] = driver_fuzz

    def generate_fuzz_driver_event(self, events, drivers):
        driver_qualifying_fuzz = self.generate_fuzz_entity_event_type(drivers, f1event.QUALIFYING)
        driver_race_fuzz = self.generate_fuzz_entity_event_type(drivers, f1event.RACE)
        # All fuzz for the simulations (if any)
        # [entity_id][event_id][sim_id] = diff_amount
        for driver_id in drivers.keys():
            if driver_id not in self._all_fuzz:
                print('ERROR: Have not generated base age/experience fuzz for %s' % driver_id)
                continue
            for event in events.values():
                if event.id() not in self._all_fuzz[driver_id]:
                    print('ERROR: No age/experience fuzz for %s in %s' % (driver_id, event.id()))
                    continue
                if len(self._all_fuzz[driver_id][event.id()]) != self._args.num_iterations:
                    print('ERROR: Mismatched amount of base fuzz for %s in %s' % (driver_id, event.id()))
                    continue
                fuzz_dict = driver_race_fuzz
                default_dist = [0, 10]
                if event.type() in [f1event.QUALIFYING, f1event.SPRINT_SHOOTOUT]:
                    fuzz_dict = driver_qualifying_fuzz
                dist = fuzz_dict.get(driver_id, default_dist)
                fuzz_samples = normal(dist[0], dist[1], self._args.num_iterations)
                for idx in range(self._args.num_iterations):
                    self._all_fuzz[driver_id][event.id()][idx] += fuzz_samples[idx]

    def generate_fuzz_entity_event_type(self, entities, event_type):
        recent_fuzz = dict()
        for entity in entities.values():
            lookback_deltas = entity.rating().lookback_deltas(event_type)
            if lookback_deltas is None or not lookback_deltas and self._logfile is not None:
                print('No lookback delta for %s' % entity.id(), file=self._logfile)
                continue
            deltas = [delta for delta in lookback_deltas if delta is not None]
            if len(deltas) < 2:
                if self._logfile is not None:
                    print('Insufficient lookback data for %s: %d' % (entity.id(), len(deltas)), file=self._logfile)
                continue
            delta_avg = np.mean(deltas)
            delta_dev = np.std(deltas)
            # print('%s\t%s\t%f\t%f' % (event_type, entity.id(), delta_avg[0], delta_dev[0]))
            recent_fuzz[entity.id()] = [delta_avg[0], delta_dev[0]]
        return recent_fuzz
