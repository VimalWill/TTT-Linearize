"""CPU-only checks that the diagnostic preserves targets and excludes near needles."""
import unittest

from diagnose_retrieval import select_rows, short_control


def example(task='single_number', distance=1000):
    answer = '1234567' if task == 'single_number' else '12345678-1234-1234-1234-123456789abc'
    return dict(id=task, task=task, length_bucket=4096, needle_distances=[distance],
                answer=answer, prompt='Read the following document.\n\nfiller\n'
                f'Register record_abcdef has value {answer}.\nmore filler\n\n'
                'Question: Return the values for registers record_abcdef, in that order. '
                'Output only the values separated by commas.\nAnswer: ')


class RetrievalDiagnosticTests(unittest.TestCase):
    def test_short_controls_preserve_fact_and_question_for_both_tasks(self):
        for task in ('single_number', 'single_uuid'):
            row = example(task)
            short = short_control(row)
            self.assertEqual(short.count(row['answer']), 1)
            self.assertNotIn('filler', short)
            self.assertEqual(short.split('\n\nQuestion: ')[1], row['prompt'].split('\n\nQuestion: ')[1])

    def test_rejects_wrong_or_duplicate_facts(self):
        row = example()
        row['answer'] = '7654321'
        with self.assertRaises(ValueError):
            short_control(row)
        row = example()
        row['prompt'] = row['prompt'].replace('more filler', 'Register record_aaaa has value 7654321.')
        with self.assertRaises(ValueError):
            short_control(row)

    def test_selects_distant_rows_in_both_cells(self):
        near = example(distance=200)
        far = example()
        uuid = example('single_uuid')
        self.assertEqual(select_rows([near, far, uuid], [4096], 1, 768), [far, uuid])

    def test_missing_cell_is_error_not_an_empty_average(self):
        with self.assertRaisesRegex(ValueError, 'single_uuid/4096'):
            select_rows([example()], [4096], 1, 768)


if __name__ == '__main__':
    unittest.main()
