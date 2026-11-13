from collections import defaultdict
from pathlib import Path

from submission_system.similarity.corpus import (
    iter_corpus_pairs,
    iter_corpus_submissions,
    load_assignments,
    summarize_corpus,
)

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus"


def test_corpus_has_required_size_and_provenance():
    summary = summarize_corpus(CORPUS / "pairs.jsonl", CORPUS / "submissions.jsonl")

    assert summary.pairs == 40
    assert summary.submissions == 60
    assert summary.families == 20
    assert summary.template_pairs >= 5
    assert summary.tuning_pairs + summary.evaluation_pairs == 40
    assert summary.relations["exact_copy"] >= 1
    assert summary.relations["formatting"] >= 1
    assert summary.relations["renaming"] >= 1
    assert summary.relations["template_only"] >= 5
    assert summary.relations["independent"] >= 1


def test_families_do_not_cross_tuning_evaluation_boundary():
    splits: dict[str, set[str]] = defaultdict(set)
    for pair in iter_corpus_pairs(CORPUS / "pairs.jsonl"):
        splits[pair.family_id].add(pair.split)

    assert all(len(values) == 1 for values in splits.values())
    tuning_families = {
        pair.family_id for pair in iter_corpus_pairs(CORPUS / "pairs.jsonl") if pair.split == "tuning"
    }
    evaluation_families = {
        pair.family_id
        for pair in iter_corpus_pairs(CORPUS / "pairs.jsonl")
        if pair.split == "evaluation"
    }
    assert tuning_families.isdisjoint(evaluation_families)


def test_every_pair_has_different_students_and_matching_assignment():
    for pair in iter_corpus_pairs(CORPUS / "pairs.jsonl"):
        assert pair.left.student_id != pair.right.student_id
        assert pair.left.assignment_id == pair.assignment_id
        assert pair.right.assignment_id == pair.assignment_id
        assert pair.left.family_id == pair.family_id
        assert pair.right.family_id == pair.family_id


def test_template_pairs_have_allowed_template():
    assignments = load_assignments(CORPUS / "assignments.jsonl")
    assert assignments["hw_template"].allowed_template

    for pair in iter_corpus_pairs(CORPUS / "pairs.jsonl"):
        if pair.template_only:
            template = assignments[pair.assignment_id].allowed_template
            assert template
            assert template.strip().splitlines()[0] in pair.left.code
            assert template.strip().splitlines()[0] in pair.right.code


def test_submission_ids_are_unique():
    submissions = list(iter_corpus_submissions(CORPUS / "submissions.jsonl"))
    assert len({item.submission_id for item in submissions}) == len(submissions)
    assert len({item.student_id for item in submissions}) > 20
