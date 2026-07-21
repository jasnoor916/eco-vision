"""Shared label-vote debouncing for the EcoVision pipelines (main.py and
main_no_grip.py)."""


def weighted_vote(votes, min_count, dominance=0.6):
    """Decide a stable label from recent (label, conf) detections.

    Counting votes alone treats a 0.90 Plastic and a 0.36 Paper as equal
    evidence — which is exactly how a flickering transparent bottle ends up
    announced as Paper. So instead:

      1. Each detection votes with its confidence as weight.
      2. The winning label must appear at least `min_count` times (rules out
         a single lucky high-confidence misread).
      3. The winner must hold at least `dominance` (default 60%) of the total
         confidence mass. While two classes are still fighting over the same
         object, nobody wins and we simply wait for more frames.

    Returns (label, best_conf_seen_for_label) when confident, else (None, 0.0).
    """
    if not votes:
        return None, 0.0

    mass = {}
    count = {}
    for lbl, conf in votes:
        mass[lbl] = mass.get(lbl, 0.0) + conf
        count[lbl] = count.get(lbl, 0) + 1

    winner = max(mass, key=mass.get)
    if count[winner] < min_count:
        return None, 0.0
    if mass[winner] / sum(mass.values()) < dominance:
        return None, 0.0

    best_conf = max(conf for lbl, conf in votes if lbl == winner)
    return winner, best_conf
