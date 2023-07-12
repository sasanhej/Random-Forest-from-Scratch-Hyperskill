from hstest import StageTest, TestCase, CheckResult
from hstest.stage_test import List
import ast

# The source data I will test on
true_data = [0.755, 0.818, 0.783, 0.839, 0.79, 0.825, 0.79, 0.811, 0.818,
             0.783, 0.825, 0.832, 0.804, 0.825, 0.825, 0.825, 0.839, 0.762,
             0.839, 0.825]


def get_list(s):
    index_from = s.find('[')
    index_to = s.find(']')
    data_str = s[index_from: index_to + 1]
    data_list = ast.literal_eval(data_str)
    if index_to + 2 > len(s):
        return data_list, None
    else:
        return data_list, s[index_to + 2:]


def full_check(result: list, true_result: list, name: str, tolerance=0.05, error_str=None):
    # Checking if the student's reply is a list
    if not isinstance(result, list):
        return CheckResult.wrong(f'Output for {name} is not a list.')

    # Checking size of the student's reply
    if len(result) != len(true_result):
        return CheckResult.wrong(f'Output for {name} should contain {len(true_result)} values,\
        found {len(result)}.')

    # Checking values of the student's reply
    for value, true_value in zip(result, true_result):
        if tolerance:
            if not (abs((value - true_value) / true_value) < tolerance):
                if error_str is not None:
                    return CheckResult.wrong(error_str)
                else:
                    return CheckResult.wrong(f'Incorrect {name} values. Check your {name} function.')
        else:
            if value != true_value:
                if error_str is not None:
                    return CheckResult.wrong(error_str)
                else:
                    return CheckResult.wrong(f'Incorrect {name} values. Check your {name} function.')

    return None


class Tests6(StageTest):

    def generate(self) -> List[TestCase]:
        return [TestCase(time_limit=1000000)]

    def check(self, reply: str, attach):
        reply = reply.strip().lower()

        if len(reply) == 0:
            return CheckResult.wrong("No output was printed!")

        if reply.count('[') != 1 or reply.count(']') != 1:
            return CheckResult.wrong('No expected list was found in output!')

        # Getting the student's results from the reply
        try:
            student, _ = get_list(reply)
        except Exception:
            return CheckResult.wrong('Seems that data output is in wrong format!')

        error = 'Incorrect predictions.'
        check_result = full_check(
            student,
            true_data,
            '',
            tolerance=0.05,
            error_str=error
        )
        if check_result:
            return check_result

        return CheckResult.correct()
