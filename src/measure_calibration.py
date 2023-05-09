# Metrics of importance
# Overall H2H correct
# - Race only
# - Qualifying only
# - Race and qualifying
# Calibration
# - Race only
# - Qualifying only
# - Race and qualifying
# Correct winners
# Correct podiums

import sys

_TAG_H2H_IDX = 0
_SEASON_H2H_IDX = _TAG_H2H_IDX + 1
_EVENT_H2H_IDX = _SEASON_H2H_IDX + 1
_CONFIDENCE_H2H_IDX = _EVENT_H2H_IDX + 1
_CORRECT_H2H_IDX = _CONFIDENCE_H2H_IDX + 1
_NUM_H2H_PARTS = _CORRECT_H2H_IDX + 1

_TAG_WIN_IDX = 0
_EVENT_WIN_IDX = _TAG_WIN_IDX + 1
_DRIVER_WIN_IDX = _EVENT_WIN_IDX + 1
_ELO_WIN_IDX = _DRIVER_WIN_IDX + 1
_ODDS_WIN_IDX = _ELO_WIN_IDX + 1
_CORRECT_WIN_IDX = _ODDS_WIN_IDX + 1
_NUM_WIN_PARTS = _CORRECT_WIN_IDX + 1

_TAG_FIN_IDX = 0
_FIN_EVENT_IDX = _TAG_FIN_IDX + 1
_FIN_MODE_IDX = _FIN_EVENT_IDX + 1
_FIN_ODDS_IDX = _FIN_MODE_IDX + 1
_FIN_CORRECT_IDX = _FIN_ODDS_IDX + 1
_FIN_NUM_PARTS = _FIN_CORRECT_IDX + 1


def get_h2h_predictions(filename, h2h_predictions):
    with open(filename, 'r') as infile:
        for line in infile:
            if not line.startswith('Predict\t'):
                continue
            parts = line.split('\t')
            if len(parts) != _NUM_H2H_PARTS:
                continue
            h2h_predictions.append(parts)


def head_to_head(h2h_predictions):
    race_num = race_correct = race_exp = race_sse = 0
    qual_num = qual_correct = qual_exp = qual_sse = 0
    for parts in h2h_predictions:
        confidence = float(parts[_CONFIDENCE_H2H_IDX])
        if confidence <= 0.5:
            continue
        correct = int(parts[_CORRECT_H2H_IDX])
        diff = correct - confidence
        if parts[_EVENT_H2H_IDX].endswith('QU'):
            qual_num += 1
            qual_correct += correct
            qual_exp += confidence
            qual_sse += (diff * diff)
        elif parts[_EVENT_H2H_IDX].endswith('RA'):
            race_num += 1
            race_correct += correct
            race_exp += confidence
            race_sse += (diff * diff)
    total_num = race_num + qual_num
    total_correct = race_correct + qual_correct
    total_exp = race_exp + qual_exp
    total_sse = race_sse + qual_sse
    if race_num:
        print('Races\t%7d\t%7d\t%.4f\t%.4f\t%.4f' % (race_num, race_correct,
                                                     float(race_correct) / race_num, race_exp / race_num,
                                                     race_sse / race_num))
    if qual_num:
        print('Qual\t%7d\t%7d\t%.4f\t%.4f\t%.4f' % (qual_num, qual_correct,
                                                    float(qual_correct) / qual_num, qual_exp / qual_num,
                                                    qual_sse / qual_num))
    if total_num:
        print('Total\t%7d\t%7d\t%.4f\t%.4f\t%.4f' % (total_num, total_correct,
                                                     float(total_correct) / total_num, total_exp / total_num,
                                                     total_sse / total_num))


def get_win_predictions(filename, win_predictions, event_type):
    with open(filename, 'r') as infile:
        for line in infile:
            if not line.startswith('Win'):
                continue
            parts = line.split('\t')
            if len(parts) != _NUM_WIN_PARTS:
                continue
            if not parts[_EVENT_WIN_IDX].endswith(event_type):
                continue
            if parts[_TAG_WIN_IDX] in win_predictions:
                win_predictions[parts[_TAG_WIN_IDX]].append(parts)
            else:
                win_predictions[parts[_TAG_WIN_IDX]] = list()
                win_predictions[parts[_TAG_WIN_IDX]].append(parts)


def one_win_calibration(tag, predictions, event_type):
    _TARGET = 750
    print(len(predictions))
    n = 0
    odds_sum = 0
    correct_sum = 0
    bucket_num = 0
    for row in sorted(predictions, key=lambda r: r[_ODDS_WIN_IDX]):
        odds = float(row[_ODDS_WIN_IDX])
        correct = int(row[_CORRECT_WIN_IDX])
        odds_sum += odds
        correct_sum += correct
        n += 1
        if n == _TARGET:
            expected_pct = odds_sum / n
            actual_pct = correct_sum / n
            error_pct = expected_pct - actual_pct
            print('%s-%s\t%3d\t%4d\t%4d\t%4d\t%.4f\t%.4f\t%7.4f' % (
                tag, event_type, bucket_num, n, odds_sum, correct_sum, expected_pct, actual_pct, error_pct)
                  )
            n = 0
            bucket_num += 1
            odds_sum = 0
            correct_sum = 0
    if n:
        expected_pct = odds_sum / n
        actual_pct = correct_sum / n
        error_pct = expected_pct - actual_pct
        print('%s-%s\t%3d\t%4d\t%4d\t%4d\t%.4f\t%.4f\t%7.4f' % (
            tag, event_type, bucket_num, n, odds_sum, correct_sum, expected_pct, actual_pct, error_pct)
              )


def win_calibration(win_predictions, event_type):
    for tag, predictions in win_predictions.items():
        one_win_calibration(tag, predictions, event_type)


def get_finish_predictions(filename, mode, finish_predictions):
    with open(filename, 'r') as infile:
        for line in infile:
            if not line.startswith('Finish\t'):
                continue
            parts = line.split('\t')
            if len(parts) != _FIN_NUM_PARTS:
                continue
            if parts[_FIN_MODE_IDX] != mode:
                continue
            decade = int(parts[_FIN_EVENT_IDX][:3]) * 10
            if decade not in finish_predictions:
                finish_predictions[decade] = list()
            finish_predictions[decade].append(parts)


def print_finish_calibration(mode, all_predictions):
    _TARGET = 500
    for decade in (sorted(all_predictions.keys())):
        predictions = all_predictions[decade]
        print(len(predictions))
        n = 0
        odds_sum = 0
        bucket_num = 0
        correct_sum = 0
        for row in sorted(predictions, key=lambda r: float(r[_FIN_ODDS_IDX])):
            odds = float(row[_FIN_ODDS_IDX])
            correct = int(row[_FIN_CORRECT_IDX])
            odds_sum += odds
            correct_sum += correct
            n += 1
            if n % _TARGET == 0:
                expected_pct = odds_sum / n
                actual_pct = correct_sum / n
                error_pct = expected_pct - actual_pct
                print('%s\t%d\t%4d\t%4d\t%4d\t%.4f\t%.4f\t%7.4f' % (
                    mode, decade, n, odds_sum, correct_sum, expected_pct, actual_pct, error_pct)
                      )
                odds_sum = 0
                correct_sum = 0
                n = 0
        if n % _TARGET != 0:
            expected_pct = odds_sum / n
            actual_pct = correct_sum / n
            error_pct = expected_pct - actual_pct
            print('%s\t%d\t%4d\t%4d\t%4d\t%.4f\t%.4f\t%7.4f' % (
                mode, decade, n, odds_sum, correct_sum, expected_pct, actual_pct, error_pct)
                  )


def main(argv):
    if len(argv) != 2:
        print("Usage: {0} <in:logfile_csv>".format(argv[0]))
        sys.exit(1)
    # h2h_predictions = list()
    # get_h2h_predictions(argv[1], h2h_predictions)
    # head_to_head(h2h_predictions)
    # for event_type in ['QU', 'RA']:
    #     win_predictions = dict()
    #     get_win_predictions(argv[1], win_predictions, event_type)
    #     win_calibration(win_predictions, event_type)
    for mode in ['All', 'Car', 'Driver']:
        finish_predictions = dict()
        get_finish_predictions(argv[1], mode, finish_predictions)
        print_finish_calibration(mode, finish_predictions)
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv)
