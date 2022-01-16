# Learning Reward Functions from a Combination of Demonstration and Evaluative Feedback

This is a code snapshot as of December 3rd 2021 prior to the HRI LBR 2022 submission deadline.

Data presented in the paper was trained using 50 seeds per set of hyperparameters on 5x10 World 1.

For example to reproduce the 50/50 mixed policy-independent action-based
and myopic ranked path-cost, run the following 50 times to generate base
data for the red line in Figure 1C.

python multimodal_feedback.py --mode train --episodes 100 --feedback_policy mixed --mixed_strat action_path_cost r1_evaluative --mixed_percent 0.5

Each set of 50 should be saved to separate directory indicating the experimental parameters used.

Then, to generate the deterministic and deterministic policy violation plots, follow the
directions in parallel_plots.sh

