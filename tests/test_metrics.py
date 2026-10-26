from eeg_agent_rag.evaluation.metrics import ndcg_at_k, recall_at_k, reciprocal_rank_at_k


def test_metrics():
    retrieved = ["x", "b", "a"]
    relevant = {"a", "b"}
    assert recall_at_k(retrieved, relevant, 2) == 0.5
    assert reciprocal_rank_at_k(retrieved, relevant, 3) == 0.5
    assert 0 < ndcg_at_k(retrieved, relevant, 3) < 1
    assert ndcg_at_k(["a", "b"], relevant, 2) == 1.0
