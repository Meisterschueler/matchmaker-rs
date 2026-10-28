"""Convert a .match file to a ground-truth TSV (perf_sec, score_beat) with the reference itself.

For match files that matchmaker-rs does not read natively (version < 1.0, or no note ids
shared with the score). Run in the tools/golden environment:

    python -m uv run match_to_tsv.py SCORE PERFORMANCE MATCH OUT.tsv
"""

import argparse

import numpy as np
from matchmaker import Matchmaker
from matchmaker.utils.eval import resolve_gt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("score")
    parser.add_argument("performance", help="MIDI file the follower receives")
    parser.add_argument("match")
    parser.add_argument("out")
    args = parser.parse_args()

    # the score as the follower sees it (unfolded, same note ids)
    mm = Matchmaker(score_file=args.score, performance_file=args.performance, input_type="midi", method="pthmm")
    perf_sec, score_beat = resolve_gt(args.match, mm.score_part.note_array(), performance=args.performance)
    np.savetxt(
        args.out,
        np.column_stack([perf_sec, score_beat]),
        delimiter="\t",
        fmt="%.17g",
        header="perf_sec\tscore_beat",
        comments="",
    )


if __name__ == "__main__":
    main()
