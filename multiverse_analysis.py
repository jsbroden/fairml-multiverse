#!/usr/bin/env python

import argparse
from pathlib import Path

from fairness_multiverse.multiverse import MultiverseAnalysis

parser = argparse.ArgumentParser("multiverse_analysis")
parser.add_argument(
    "--mode",
    help=(
        "How to run the multiverse analysis. "
        "(continue: continue from previous run, "
        "full: run all universes, "
        "test: run only a small subset of universes)"
    ),
    choices=["full", "continue", "test"],
    default="full",
)
def verify_dir(string):
    if Path(string).is_dir():
        return string
    else:
        raise NotADirectoryError(string)
parser.add_argument(
    "--output-dir",
    help=(
        "Relative path to output directory for the results."
    ),
    default="./output",
    type=verify_dir,
)
parser.add_argument(
    "--seed",
    help=(
        "The seed to use for the analysis."
    ),
    default="2023",
    type=int,
)
args = parser.parse_args()

multiverse_analysis = MultiverseAnalysis(
    dimensions={
        "model": ["logreg", "rf", "gbm"],
        "exclude_features": [
            "none",
            "frau1",
            "maxdeutsch1",
            "maxdeutsch.Missing.",
        ],
        # Post-Deployment
        "cutoff": [["raw_0.5", "quantile_0.1", "quantile_0.25"]],
    },
    output_dir=Path(args.output_dir),
    new_run=(args.mode != "continue"),
    seed=args.seed,
)

multiverse_grid = multiverse_analysis.generate_grid(save=True)
print(f"Generated N = {len(multiverse_grid)} universes")


print(f"~ Starting Run No. {multiverse_analysis.run_no} (Seed: {multiverse_analysis.seed})~")

# Run the analysis for the first universe
if args.mode == "test":
    print("Small-Scale-Test Run")
    multiverse_analysis.visit_universe(multiverse_grid[0])
    multiverse_analysis.visit_universe(multiverse_grid[1])
elif args.mode == "continue":
    print("Continuing Previous Run")
    missing_universes = multiverse_analysis.check_missing_universes()[
        "missing_universes"
    ]

    # Run analysis only for missing universes
    multiverse_analysis.examine_multiverse(multiverse_grid=missing_universes)
else:
    print("Full Run")
    # Run analysis for all universes
    multiverse_analysis.examine_multiverse(multiverse_grid=multiverse_grid)

multiverse_analysis.aggregate_data(save=True)

multiverse_analysis.check_missing_universes()
