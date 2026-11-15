"""Relay rank-zero metrics to the launcher that owns the online SwanLab run."""

import json

METRIC_PREFIX = "__TRD_SWANLAB_METRIC__="


class SwanLabRelay:
    def log(self, data):
        print(METRIC_PREFIX + json.dumps(data, default=float), flush=True)

    def finish(self, **kwargs):
        # The parent owns success/failure, including rollout and HF export errors.
        pass
