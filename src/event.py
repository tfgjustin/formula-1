class Event(object):

    def __init__(self, event_id, name, season, stage, date, event_type):
        self._id = event_id
        self._name = name
        self._season = int(season)
        self._stage = int(stage)
        self._date = date
        self._type = event_type
        self._results = list()
        self._drivers = set()
        self._teams = set()

    def start_updates(self):
        for driver in sorted(self._drivers, key=lambda d: d.id()):
            driver.start_update(self._id)
        for team in sorted(self._teams, key=lambda t: t.id()):
            team.start_update(self._id)

    def add_result(self, result):
        if result.event().id() == self._id:
            self._results.append(result)
            self._drivers.add(result.driver())
            if result.team() is not None:
                self._teams.add(result.team())

    def commit_updates(self):
        for driver in self._drivers:
            driver.commit_update()
        for team in self._teams:
            team.commit_update()

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


class Qualifying(Event):
    def __init__(self, event_id, name, season, stage, date):
        super().__init__(event_id, name, season, stage, date, 'Q')


class Race(Event):
    def __init__(self, event_id, name, season, stage, date):
        super().__init__(event_id, name, season, stage, date, 'R')
