from ratings import EloRating


class Team(object):
    def __init__(self, uuid, team_id, team_name, rating=None, is_alias=False):
        self._uuid = uuid
        self._id = team_id
        self._name = team_name
        self._rating = rating
        self._is_alias = is_alias

    def uuid(self):
        return self._uuid

    def id(self):
        return self._id

    def name(self):
        return self._name

    def is_alias(self):
        return self._is_alias

    def start_update(self, event_id):
        self._rating.start_update(event_id, self._id, is_alias=self._is_alias)

    def commit_update(self):
        self._rating.commit_update()
        if self._rating.rating() is None:
            print('ERROR: Team %s has None rating' % self._id)

    def rating(self):
        return self._rating


class TeamFactory(object):

    def __init__(self, args):
        # Various parameters when setting up and modifying ratings
        self._args = args
        # Unique counter so we can track canonical teams through aliases etc
        self._team_id_counter = 0
        # Set of valid team IDs, used for validation when loading results
        self._all_team_ids = set()
        # (event_id, team_id) ~> Team object
        self._all_teams = dict()
        # Check to see if this ID is an alias
        self._is_alias = dict()
        # These are changes to a team
        self._is_change = dict()
        # Collect errors where multiple teams are changing from one canonical team
        self._changes = dict()
        # String identifier (e.g., Team_Lotus) to the current Team object
        self._current_teams = dict()
        # Mapping of events to teams we have to handle
        self._teams_by_event = dict()
        # Overall flag if this operation succeeded or not.
        self._success = True

    def teams(self):
        return self._all_teams

    def is_valid_team_id(self, team_id):
        return team_id in self._all_team_ids

    def get_current_team(self, team_id):
        return self._current_teams.get(team_id, None)

    def update_for_event(self, event_id):
        teams = self._teams_by_event.get(event_id, None)
        if teams is None:
            return
        for team_key in teams:
            if team_key not in self._all_teams:
                print('ERROR: Event %s references invalid team %s' % (event_id, team_key))
                continue
            team_obj = self._all_teams[team_key]
            self._current_teams[team_obj.id()] = team_obj

    def reset_current_teams(self):
        self._current_teams = dict()

    def create_team(self, team_type, event_id, team_id, team_name, other_event_id=None, other_team_id=None):
        this_key = '%s:%s' % (event_id, team_id)
        teams = self._teams_by_event.get(event_id, list())
        teams.append(this_key)
        self._teams_by_event[event_id] = teams
        other_key = None
        if other_event_id is not None and other_team_id is not None:
            other_key = '%s:%s' % (other_event_id, other_team_id)
        if team_type == 'new':
            uuid = 'Team%04d' % self._team_id_counter
            self._all_teams[this_key] = Team(
                uuid, team_id, team_name,
                rating=EloRating(
                    self._args.team_elo_initial,
                    regress_rate=self._args.team_elo_regress,
                    k_factor_regress_rate=self._args.team_kfactor_regress
                ))
            self._all_team_ids.add(team_id)
            self._team_id_counter += 1
        elif team_type == 'change':
            if other_key is not None:
                self._is_change[this_key] = other_key
                changelog = self._changes.get(other_key, list())
                changelog.append(this_key)
                self._changes[other_key] = changelog
                self._all_team_ids.add(team_id)
            else:
                print('ERROR: change for %s is missing a valid other team' % (this_key))
                self._success = False
        elif team_type == 'alias':
            if other_key is not None:
                self._is_alias[this_key] = other_key
                self._all_team_ids.add(team_id)
            else:
                print('ERROR: alias for %s is missing a valid other team' % (this_key))
                self._success = False
        else:
            print('ERROR: Invalid team_type: %s' % (team_type))
            self._success = False

    def finalize_create(self):
        # A few checks:
        # 1) Ensure that team 'A' doesn't change into 'B' and then 'A' back into 'C'
        for team_id, next_ids in self._changes.items():
            if len(next_ids) > 1:
                print('ERROR: Team "%s" becomes [%s]' % (team_id, ', '.join(next_ids)))
                self._success = False
        # 2) Check to make sure that an alias does not point to another alias
        for alias_id, canonical_id in self._is_alias.items():
            if canonical_id in self._is_alias:
                print('ERROR: Alias of "%s" points to "%s", which is itself an alias' % (
                    alias_id, canonical_id
                ))
                self._success = False
        # 3) Check to make sure that the other team in a change is canonical and not an alias
        for team_id, prev_id in self._is_change.items():
            if prev_id in self._is_alias:
                print('ERROR: Team "%s" has a previous non-canonical ID of %s' % (
                    team_id, prev_id
                ))
                self._success = False
        # If all the data looks good, then finish setting up the links
        if not self._success:
            return self._success
        # Go through and set up the changes.
        # This should iterate the team IDs in order
        for team_key, prev_key in self._is_change.items():
            prev_team_obj = self._all_teams.get(prev_key, None)
            if prev_team_obj is None:
                print('ERROR: Team %s comes from invalid previous team %s' % (
                    team_key, prev_key
                ))
                self._success = False
                continue
            event_id, team_id = team_key.split(':', 1)
            self._all_teams[team_key] = Team(prev_team_obj.uuid(), team_id, None, rating=prev_team_obj.rating())
        # Now all the aliases
        for alias_key, canonical_key in self._is_alias.items():
            canonical_obj = self._all_teams.get(canonical_key, None)
            if canonical_obj is None:
                print('ERROR: Team %s is aliased to a non-canonical team %s' % (
                    alias_key, canonical_key
                ))
                self._success = False
                continue
            event_id, alias_id = alias_key.split(':', 1)
            self._all_teams[alias_key] = Team(canonical_obj.uuid(), alias_id, None,
                                              rating=canonical_obj.rating(), is_alias=True)
        return self._success
