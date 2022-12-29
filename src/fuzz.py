import functools
import random

from collections import defaultdict
from event import compare_events


def avg_elo_age_experience(age, experience):
    """Formula to project out the average Elo rating of a driver based on age and experience."""
    age -= 20
    age_2 = age ** 2
    exp_2 = experience ** 2
    return -17.11 + (-2.84 * age) + (-0.200 * age_2) + (30.30 * experience) + (-1.14 * exp_2)


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
        if self._logfile is not None or True:
            print('Generating fuzz: start')  # , file=self._logfile)
        event_ids = sorted(events.keys(),  key=functools.cmp_to_key(compare_events))
        min_stage = events[event_ids[0]].stage()
        num_stages = events[event_ids[-1]].stage()
        self.generate_driver_age_fuzz(year, events, num_stages, drivers)
        self.generate_driver_event_fuzz(events, drivers)
        self.generate_team_estimate_fuzz(events, min_stage, num_stages)
        self.generate_team_event_fuzz(events, teams)
        if self._logfile is not None or True:
            print('Generating fuzz: finish')  # , file=self._logfile)

    def generate_driver_age_fuzz(self, year, events, num_stages, drivers):
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
            current_fuzz = avg_elo_age_experience(current_age, current_exp)
            if current_exp:
                # This is not their rookie year. Subtract out their previous experience.
                previous_year = seasons[non_current_idx]
                current_fuzz -= avg_elo_age_experience(previous_year - birth_year, current_exp - 1)
            if self._logfile is not None:
                print('%d %d %d %s %6.2f' % (year, birth_year, year - birth_year, driver.id(), current_fuzz), file=self._logfile)
            # The fuzz represents the change in the average rating from year to year.
            # We'll assume the change is linear, so the delta at race=1 is 0 and race=N is 2*fuzz
            # For each simulation we'll pick a number with mean=fuzz and stddev=12
            # Then we'll iterate through each event and apply the linear adjustment
            driver_fuzz = defaultdict(list)
            elo_stddev = max([12, abs(current_fuzz)])
            for _ in range(self._args.num_iterations):
                year_elo_diff = 2 * random.gauss(current_fuzz, elo_stddev)
                for event in events.values():
                    event_diff = (event.stage() * year_elo_diff) / num_stages
                    driver_fuzz[event.id()].append(event_diff)
            self._all_fuzz[driver.id()] = driver_fuzz

    def generate_driver_event_fuzz(self, events, drivers):
        driver_qualifying_fuzz = dict({})
        driver_race_fuzz = dict({})
        # All fuzz for the simulations (if any)
        # [entity_id][event_id][sim_id] = diff_amount
        for driver_id in drivers.keys():
            for event in events.values():
                fuzz_dict = driver_race_fuzz
                default_dist = [-3.1, 11.5]
                if event.type() == 'Q':
                    fuzz_dict = driver_qualifying_fuzz
                fuzz_list = list()
                for _ in range(self._args.num_iterations):
                    dist = fuzz_dict.get(driver_id, default_dist)
                    fuzz_list.append(random.gauss(dist[0], dist[1]))
                self._all_fuzz[driver_id][event.id()] = fuzz_list

    def generate_team_estimate_fuzz(self, events, min_stage, num_stages):
        team_fuzz = dict({})
        stages_left = num_stages - min_stage + 1
        for team_id, base_elo_diff in team_fuzz.items():
            team_fuzz = defaultdict(list)
            elo_stddev = max([min([30, abs(base_elo_diff)]), 15])
            for _ in range(self._args.num_iterations):
                year_elo_diff = random.gauss(base_elo_diff, elo_stddev)
                for event in events.values():
                    event_diff = ((event.stage() + 1 - min_stage) * year_elo_diff) / stages_left
                    team_fuzz[event.id()].append(event_diff)
            self._all_fuzz[team_id] = team_fuzz

    def generate_team_event_fuzz(self, events, teams):
        team_qualifying_fuzz = dict({})
        team_race_fuzz = dict({})
        # All fuzz for the simulations (if any)
        # [entity_id][event_id][sim_id] = diff_amount
        for team_id in teams.keys():
            for event in events.values():
                fuzz_dict = team_race_fuzz
                default_dist = [0, 35]
                if event.type() == 'Q':
                    fuzz_dict = team_qualifying_fuzz
                fuzz_list = list()
                for _ in range(self._args.num_iterations):
                    dist = fuzz_dict.get(team_id, default_dist)
                    fuzz_list.append(random.gauss(dist[0], dist[1]))
                self._all_fuzz[team_id][event.id()] = fuzz_list
