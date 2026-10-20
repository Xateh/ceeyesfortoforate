import copy
from decimal import Decimal, localcontext
import math
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from reranker import Candidate as C, Question as Q, FeatureConfig as F, Reranker as R
from reranker import questions_from_records


def fixture():
    questions = [
        Q("q1", [C("a", "wrong", {"score": 0., "type_match": 0.}),
                  C("b", "right", {"score": 2., "type_match": 1.})]),
        Q("q2", [C("a", "low", {"score": -1., "type_match": 0.}),
                  C("b", "middle", {"score": 1., "type_match": 0.5}),
                  C("c", "best", {"score": 3., "type_match": 1.})]),
        Q("q3", [C("only", "single", {"score": 4., "type_match": 0.})]),
    ]
    return questions, {"q1": {"a": 0., "b": 1.},
                       "q2": {"a": 0., "b": 0.5, "c": 1.},
                       "q3": {"only": 0.}}


def records(questions):
    return [{"question_id": q.question_id, "candidates": [
        {"candidate_id": c.candidate_id, "answer": c.answer, "features": dict(c.features)}
        for c in q.candidates]} for q in questions]


class RerankerTests(unittest.TestCase):
    def setUp(self):
        self.questions, self.targets = fixture()
        self.basic = F(("score",), ("type_match",))
        self.full = F(("score",), ("type_match",), "full")

    def test_variants_train_select_and_reload(self):
        for config in (self.basic, self.full):
            with self.subTest(group=config.group), tempfile.TemporaryDirectory() as tmp:
                model = R(config).fit(self.questions, self.targets)
                self.assertEqual(model.predict(self.questions),
                                 {"q1": "right", "q2": "best", "q3": "single"})
                for question in self.questions:
                    self.assertIn(model.select([question])[question.question_id], question.candidates)
                path = Path(tmp) / "model.json"
                model.save(path)
                other = R.load(path)
                self.assertEqual(model.scores(self.questions), other.scores(self.questions))
                self.assertEqual(other.features, config)
                other.save_predictions(self.questions, Path(tmp) / "predictions.json")
                self.assertEqual(json.loads((Path(tmp) / "predictions.json").read_text()),
                                 model.predict(self.questions))

    def test_basic_excludes_compatibility_even_when_missing_or_invalid(self):
        model = R(self.basic).fit(self.questions, self.targets)
        changed = copy.deepcopy(self.questions)
        for q in changed:
            for c in q.candidates:
                c.features.pop("type_match")
                c.features["irrelevant"] = "not numeric"
        self.assertEqual(model.scores(changed), model.scores(self.questions))
        changed[0].candidates[0].features["type_match"] = float("nan")
        self.assertEqual(model.scores(changed), model.scores(self.questions))
        retrained = R(self.basic).fit(changed, self.targets)
        self.assertEqual(retrained.scores(changed), model.scores(self.questions))
        with self.assertRaises(ValueError):
            R(self.full).fit(changed, self.targets)

    def test_full_uses_compatibility(self):
        questions = [Q("x", [C("a", "no", {"score": 1., "type_match": 0.}),
                             C("b", "yes", {"score": 1., "type_match": 1.})])]
        targets = {"x": {"a": 0., "b": 1.}}
        self.assertEqual(R(self.basic).fit(questions, targets).predict(questions), {"x": "no"})
        self.assertEqual(R(self.full).fit(questions, targets).predict(questions), {"x": "yes"})

    def test_train_only_preprocessing_and_no_inference_mutation(self):
        model = R(self.basic).fit(self.questions, self.targets)
        train_values = torch.tensor([0., 2., -1., 1., 3., 4.], dtype=torch.float64)
        self.assertEqual(model._state["mean"].item(), train_values.mean().item())
        self.assertEqual(model._state["scale"].item(), train_values.std(correction=0).item())
        before = copy.deepcopy(model._state)
        model.predict([Q("validation", [C("v", "v", {"score": 1000000.})])])
        for key in ("mean", "scale", "coef"):
            self.assertTrue(torch.equal(before[key], model._state[key]))

    def test_deterministic_ties_follow_supplied_order(self):
        model = R(self.basic).fit(self.questions, self.targets)
        tied = Q("tie", [C("z", "first", {"score": 1.}), C("a", "second", {"score": 1.})])
        for _ in range(3):
            self.assertEqual(model.select([tied])["tie"].candidate_id, "z")
        reversed_q = Q("tie", list(reversed(tied.candidates)))
        self.assertEqual(model.predict([reversed_q]), {"tie": "second"})

    def test_all_tied_and_zero_targets_require_explicit_policy(self):
        for value in (0., 0.7, 1.):
            targets = {q.question_id: {c.candidate_id: value for c in q.candidates}
                       for q in self.questions}
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "no within-question"):
                    R(self.basic).fit(self.questions, targets)
                model = R(self.basic, no_signal="first").fit(self.questions, targets)
                self.assertEqual(model.predict(self.questions),
                                 {"q1": "wrong", "q2": "low", "q3": "single"})
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "zero.json"
                    model.save(path)
                    self.assertEqual(R.load(path).predict(self.questions), model.predict(self.questions))
        model = R(self.basic).fit(self.questions, self.targets)
        self.assertEqual(model._state["informative_questions"], 2)

    def test_no_gold_leakage(self):
        model = R(self.basic).fit(self.questions, self.targets)
        changed = copy.deepcopy(self.questions)
        for q in changed:
            for c in q.candidates:
                c.features.update({"gold": "secret", "target": float("nan"), "f1": 1.})
        self.assertEqual(model.scores(changed), model.scores(self.questions))
        self.assertEqual(R(self.basic).fit(changed, self.targets).scores(changed),
                         model.scores(self.questions))
        # Altering separate labels cannot alter an already-fitted model.
        self.targets["q1"]["a"] = 1.
        self.assertEqual(model.predict(changed)["q1"], "right")
        for name in ("gold", "gold_overlap", "candidate_f1", "label", "target"):
            with self.assertRaises(ValueError):
                F((name,))
        bad = records(self.questions)
        bad[0]["candidates"][0]["target"] = 1.
        with self.assertRaises(ValueError):
            questions_from_records(bad)

    def test_bad_questions_and_features(self):
        bad_sets = [
            [], [Q("empty", [])], [self.questions[0], self.questions[0]],
            [Q("", [C("a", "a", {"score": 1.})])],
            [Q("x", [C("a", "a", {"score": 1.}), C("a", "b", {"score": 2.})])],
            [Q("x", [C("", "a", {"score": 1.})])],
            [Q("x", [C("a", None, {"score": 1.})])],
            [Q("x", [C("a", "a", {})])],
            [Q("x", [C("a", "a", None)])],
            [None], "questions",
        ]
        bad_sets += [[Q("x", [C("a", "a", {"score": value})])]
                     for value in (None, True, "1", float("inf"), float("nan"))]
        model = R(self.basic).fit(self.questions, self.targets)
        for questions in bad_sets:
            with self.subTest(questions=questions), self.assertRaises(ValueError):
                model.predict(questions)

    def test_bad_targets_and_failed_refit_preserves_model(self):
        model = R(self.basic).fit(self.questions, self.targets)
        before = model.predict(self.questions)
        cases = [None, {}, {**self.targets, "extra": {}}]
        for value in (None, True, "1", float("nan"), -0.1, 1.1):
            labels = copy.deepcopy(self.targets)
            labels["q1"]["a"] = value
            cases.append(labels)
        labels = copy.deepcopy(self.targets)
        labels["q1"] = {"a": 0.}
        cases.append(labels)
        for labels in cases:
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                model.fit(self.questions, labels)
            self.assertEqual(model.predict(self.questions), before)

    def test_bad_config_and_unfitted(self):
        for args in [((),), (("score", "score"),), (("score",), ("score",)),
                     (("score",), (), "full"), (("score",), (), "other"),
                     ("score",)]:
            with self.assertRaises(ValueError):
                F(*args)
        for alpha in (0., -1., float("nan"), True):
            with self.assertRaises(ValueError):
                R(self.basic, alpha=alpha)
        with self.assertRaises(ValueError):
            R(self.basic, no_signal="silent")
        with self.assertRaises(ValueError):
            R(self.basic).predict(self.questions)
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            R(self.basic).save(Path(tmp) / "unfit.json")

    def test_corrupt_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            R(self.basic).fit(self.questions, self.targets).save(path)
            good = json.loads(path.read_text())
            cases = []
            for key, value in (("mean", []), ("coef", [float("nan")]), ("scale", [0.]),
                               ("training_questions", -1), ("informative_questions", 999)):
                bad = copy.deepcopy(good)
                bad["state"][key] = value
                cases.append(bad)
            cases += [{**good, "version": 99}, {**good, "objective": "unknown"}, {}]
            for bad in cases:
                path.write_text(json.dumps(bad))
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    R.load(path)

    def test_configuration_is_read_only(self):
        model = R(self.basic).fit(self.questions, self.targets)
        for attr, value in (("features", F(("other",))), ("alpha", 2.),
                            ("no_signal", "first")):
            with self.subTest(attr=attr), self.assertRaises(AttributeError):
                setattr(model, attr, value)
        self.assertEqual(model.predict(self.questions)["q1"], "right")

    def test_oversized_integers_fail_cleanly(self):
        huge = 10 ** 1000
        with self.assertRaises(ValueError):
            R(self.basic, alpha=huge)
        model = R(self.basic).fit(self.questions, self.targets)
        with self.assertRaises(ValueError):
            model.predict([Q("x", [C("a", "a", {"score": huge})])])
        labels = copy.deepcopy(self.targets)
        labels["q1"]["a"] = huge
        with self.assertRaises(ValueError):
            model.fit(self.questions, labels)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            model.save(path)
            data = json.loads(path.read_text())
            data["state"]["coef"] = [huge]
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                R.load(path)

    def test_equal_question_weight_with_duplicate_candidate_rows(self):
        # Repeat every candidate of one informative question with new IDs.
        # With identical training statistics, loss weight must remain unchanged.
        original = [
            Q("a", [C("1", "a", {"score": 0.}), C("2", "b", {"score": 2.})]),
            Q("b", [C("1", "c", {"score": 0.}), C("2", "d", {"score": 2.})]),
        ]
        labels = {"a": {"1": 0., "2": 1.}, "b": {"1": 1., "2": 0.}}
        repeated = [original[0], Q("b", [
            C(str(i), "unused", {"score": float(i % 2) * 2}) for i in range(10)])]
        repeated_labels = {"a": labels["a"], "b": {str(i): float(1 - i % 2) for i in range(10)}}
        first = R(self.basic).fit(original, labels)
        second = R(self.basic).fit(repeated, repeated_labels)
        torch.testing.assert_close(first._state["coef"], second._state["coef"], atol=1e-15, rtol=0)


    def test_collinear_features_with_tiny_penalty(self):
        questions = [Q("q", [C("a", "first", {"x": 0., "y": 0.}),
                              C("b", "second", {"x": 1., "y": 1.})])]
        labels = {"q": {"a": 0., "b": 1.}}
        for alpha in (1., 1e-20, 1e-300):
            with self.subTest(alpha=alpha), tempfile.TemporaryDirectory() as tmp:
                model = R(F(("x", "y")), alpha=alpha).fit(questions, labels)
                # Standardized columns are identical [-1, 1]; the analytic
                # symmetric ridge solution has both weights 0.5 / (2 + alpha).
                expected = torch.full((2,), 0.5 / (2. + alpha), dtype=torch.float64)
                torch.testing.assert_close(model._state["coef"], expected,
                                           atol=1e-14, rtol=0.)
                self.assertEqual(model.predict(questions), {"q": "second"})
                path = Path(tmp) / "model.json"
                model.save(path)
                self.assertEqual(R.load(path).scores(questions), model.scores(questions))

    def test_subnormal_penalty_and_small_singular_value(self):
        alpha, delta = 5e-324, 1e-162
        questions = [Q("signal", [C("a", "a", {"x": -delta}),
                                   C("b", "b", {"x": delta})]),
                     Q("scaling", [C("a", "a", {"x": -1.}),
                                    C("b", "b", {"x": 1.})])]
        labels = {"signal": {"a": 0., "b": 1.}, "scaling": {"a": 0., "b": 0.}}
        model = R(F(("x",)), alpha=alpha).fit(questions, labels)
        # Only the first question is informative. Its standardized inputs are
        # +/- amplitude, so the scalar optimum is (amplitude/2)/(amplitude^2+alpha).
        # Decimal supplies an independent oracle where the square is below the
        # float64 representable range; scaling still uses all training rows.
        with localcontext() as context:
            context.prec = 100
            amplitude = Decimal.from_float(delta) / Decimal.from_float(math.sqrt(.5))
            expected = float((amplitude / 2) /
                             (amplitude * amplitude + Decimal.from_float(alpha)))
        self.assertAlmostEqual(model._state["coef"].item() / expected, 1., delta=1e-14)
        self.assertEqual(model.predict([questions[0]]), {"signal": "b"})

    def test_numeric_overflow_refit_preserves_model(self):
        model = R(self.basic).fit(self.questions, self.targets)
        before = model.scores(self.questions)
        questions = [Q("q", [C("a", "first", {"score": 1e308}),
                              C("b", "second", {"score": 1e308})])]
        with self.assertRaisesRegex(ValueError, "numeric overflow"):
            model.fit(questions, {"q": {"a": 0., "b": 1.}})
        self.assertEqual(model.scores(self.questions), before)

    def test_adapter_and_cli_end_to_end(self):
        self.assertEqual(questions_from_records(records(self.questions)), self.questions)
        for invalid in (None, {}, [{"question_id": "q"}],
                        [{"question_id": "q", "candidates": {}}]):
            with self.assertRaises(ValueError):
                questions_from_records(invalid)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, data in (("questions", records(self.questions)), ("targets", self.targets)):
                (root / (name + ".json")).write_text(json.dumps(data))
            for config in (self.basic, self.full):
                (root / "features.json").write_text(json.dumps(config.to_dict()))
                cmd = [sys.executable, "-m", "reranker"]
                train = subprocess.run(cmd + ["train", "--questions", str(root / "questions.json"),
                    "--targets", str(root / "targets.json"), "--features", str(root / "features.json"),
                    "--model", str(root / "model.json")], capture_output=True, text=True)
                self.assertEqual(train.returncode, 0, train.stderr)
                predict = subprocess.run(cmd + ["predict", "--questions", str(root / "questions.json"),
                    "--model", str(root / "model.json"), "--output", str(root / "predictions.json")],
                    capture_output=True, text=True)
                self.assertEqual(predict.returncode, 0, predict.stderr)
                self.assertEqual(json.loads((root / "predictions.json").read_text()),
                                 {"q1": "right", "q2": "best", "q3": "single"})


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
