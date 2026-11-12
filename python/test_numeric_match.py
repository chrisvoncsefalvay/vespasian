"""Final numeric grading for worked maths answers."""
from fractions import Fraction
import unittest

from worker import final_number, numeric_match


class NumericMatchTests(unittest.TestCase):
    def test_different_worked_solutions_keep_the_correct_final_answer(self):
        reference = "48/2 = <<48/2=24>>24. Then 24 + 6 = <<24+6=30>>30.\n#### 30"
        for answer in ("30", "The answer is 30.", "After the two calculations, we get 30.",
                       "24 + 6 = 30", r"The result is \boxed{30}", "#### 30.0"):
            with self.subTest(answer=answer):
                self.assertEqual(numeric_match(answer, reference), 1.)
        self.assertEqual(numeric_match("30 was the intermediate value. The answer is 31.", reference), 0.)

    def test_explicit_final_answers_take_precedence_over_other_numbers(self):
        self.assertEqual(final_number("#### 42\nThis took 3 steps."), 42)
        self.assertEqual(final_number(r"The result is \boxed{42}, after 3 steps."), 42)
        self.assertEqual(final_number("The final answer is 42.\nThere are 3 steps."), 42)
        self.assertIsNone(final_number("42 appeared in the work. #### unknown"))

    def test_declared_answer_is_not_replaced_by_same_line_explanation(self):
        for answer in ("The answer is 42. This uses 3 groups.",
                       "The final answer is 42, after 3 steps.",
                       "#### 42. This uses 3 groups.",
                       r"\boxed{42} after 3 steps."):
            with self.subTest(answer=answer):
                self.assertEqual(final_number(answer), 42)
                self.assertEqual(numeric_match(answer, "#### 42"), 1.)
                self.assertEqual(numeric_match(answer, "#### 3"), 0.)
        self.assertEqual(final_number("The answer is **$1,200.00**, in 3 instalments."), 1200)
        self.assertEqual(final_number(r"The answer is \(\frac{1}{2}\), after 3 steps."), Fraction(1, 2))
        self.assertEqual(final_number("We combine 2 and 3 to get 5."), 5)

    def test_invalid_declared_answer_never_falls_back_to_another_number(self):
        for answer in ("Earlier we got 42. ####", "The answer is unknown after 42 steps.",
                       "#### unknown, though we tried 42 times.", "The final answer is:"):
            with self.subTest(answer=answer):
                self.assertIsNone(final_number(answer))

    def test_exact_numeric_normalisation(self):
        for answer, expected in (("$1,200.00", "1200"), ("-0.5", "-1/2"),
                                 (".25", "1/4"), ("1e3", "1000"), ("−12", "-12")):
            with self.subTest(answer=answer):
                self.assertEqual(numeric_match(answer, expected), 1.)
        self.assertEqual(final_number("1/3"), Fraction(1, 3))
        self.assertEqual(final_number(r"\boxed{\frac{1}{3}}"), Fraction(1, 3))
        self.assertEqual(numeric_match(r"\boxed{\frac{1}{2}}", "2"), 0.)
        self.assertEqual(numeric_match("0.333", "1/3"), 0.)

    def test_unfinished_or_missing_answers_do_not_score(self):
        self.assertEqual(numeric_match("42", "#### 42", finish_reason="length"), 0.)
        for answer in ("", "No answer", "NaN", "infinity", "1/0", "1e9999999",
                       "<think>42</think>", "<think>42"):
            with self.subTest(answer=answer):
                self.assertEqual(numeric_match(answer, "42"), 0.)
        self.assertEqual(numeric_match("<think>42</think>The answer is 43.", "42"), 0.)
        self.assertEqual(numeric_match("No answer", "No reference"), 0.)


if __name__ == "__main__":
    unittest.main()
