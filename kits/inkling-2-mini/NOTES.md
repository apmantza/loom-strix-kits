# inkling-2-mini: model knowledge for optimisation work

What is specific to this model. The method is in skills/loom-strix-optimise/; device facts in its devices/ files.

## Model and serving shape
TODO: layers and their kinds, heads, formats, speculative decoding; the launch shapes serving actually runs (read them from
a served profile: common/events_ab.sh, then the workgroup counts per kernel).

## Lanes and the kernels behind them
TODO: adapt lanes.json's regexes to this kit's kernel names; one row per lane with its bound (bytes / issue / latency).

## Where the time is
TODO: common/events_summary.py kits/inkling-2-mini/lanes.json RUN after the first served profile.
