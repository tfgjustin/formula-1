from collections import defaultdict


def avg_elo_age_experience(age, experience):
    """Formula to project out the average Elo rating of a driver based on age and experience."""
    age -= 20
    age_2 = age ** 2
    exp_2 = experience ** 2
    return -17.11 + (-2.84 * age) + (-0.200 * age_2) + (30.30 * experience) + (-1.14 * exp_2)


class Fuzzer(object):

    BASE_TAG = '__BASE__'

    def __init__(self, args, logfile):
        self._args = args
        self._logfile = logfile
        # All fuzz for the simulations (if any)
        # [sim_idx][event_id][entity_id] = diff_amount
        self._all_fuzz = defaultdict(dict)

    def all_fuzz(self):
        return self._all_fuzz

    def generate_all_fuzz(self, year, events, drivers, teams):
        print('Generating fuzz: start', file=self._logfile)
        self.generate_driver_age_fuzz(year, drivers)
        self.generate_driver_event_fuzz(events, drivers)
        self.generate_team_estimate_fuzz()
        self.generate_team_event_fuzz(events, teams)
        print('Generating fuzz: finish', file=self._logfile)

    def generate_driver_age_fuzz(self, year, drivers):
        # TODO: For each driver in the dictionary estimate out the baseline delta between their last year and this year.
        return

    def generate_driver_event_fuzz(self, events, drivers):
        # TODO: Generate 'fuzz' for each event for each driver.
        return

    def generate_team_estimate_fuzz(self):
        # TODO: Estimate out/load hand estimates for how much up versus down a team is. Useful at the start of a season.
        return

    def generate_team_event_fuzz(self, events, teams):
        # TODO: Generate 'fuzz' for each event for each team.
        return
