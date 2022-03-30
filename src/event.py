def compare_events(e1, e2):
    left = e1.split('-')
    right = e2.split('-')
    if left[0] < right[0]:
        print(left[0], right[0])
        return -1
    elif left[0] > right[0]:
        return 1
    if left[1] < right[1]:
        return -1
    elif left[1] > right[1]:
        return 1
    if left[2] == 'Q':
        if right[2] == 'Q':
            return 0
        else:
            return -1
    elif left[2] == 'S':
        if right[2] == 'Q':
            return 1
        elif right[2] == 'R':
            return -1
        else:
            return 0
    else:
        if right[2] == 'R':
            return 0
        else:
            return 1


class Event(object):

    def __init__(self, event_id, name, season, stage, date, event_type, num_laps, lap_distance_km):
        self._id = event_id
        self._name = name
        self._season = int(season)
        self._stage = int(stage)
        self._date = date
        self._type = event_type
        self._num_laps = int(num_laps)
        self._lap_distance_km = float(lap_distance_km)
        self._entrants = list()
        self._results = list()
        self._drivers = set()
        self._teams = set()

    def add_entrant(self, entrant):
        if entrant.event().id() == self._id:
            self._entrants.append(entrant)
            self._drivers.add(entrant.driver())
            if entrant.team() is not None:
                self._teams.add(entrant.team())
            if entrant.has_result():
                self._results.append(entrant.result())

    def num_entrants(self):
        return len(self._results)

    def entrants(self):
        return self._entrants

    def has_results(self):
        return len(self._results) > 0

    def results(self):
        return self._results

    def drivers(self):
        return self._drivers

    def teams(self):
        return self._teams

    def id(self):
        return self._id

    def name(self):
        return self._name

    def season(self):
        return self._season

    def stage(self):
        return self._stage

    def date(self):
        return self._date

    def type(self):
        return self._type

    def num_laps(self):
        return self._num_laps

    def lap_distance_km(self):
        return self._lap_distance_km

    def total_distance_km(self):
        return self._num_laps * self._lap_distance_km


class Qualifying(Event):
    def __init__(self, event_id, name, season, stage, date, num_laps, lap_distance_km):
        super().__init__(event_id, name, season, stage, date, 'Q', num_laps, lap_distance_km)


class SprintQualifying(Event):
    def __init__(self, event_id, name, season, stage, date, num_laps, lap_distance_km):
        super().__init__(event_id, name, season, stage, date, 'S', num_laps, lap_distance_km)


class Race(Event):
    def __init__(self, event_id, name, season, stage, date, num_laps, lap_distance_km):
        super().__init__(event_id, name, season, stage, date, 'R', num_laps, lap_distance_km)
