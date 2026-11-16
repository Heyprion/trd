"""Run-local evaluation scope changes, checked after persisted benchmark scores."""
import json
import os
from pathlib import Path
import signal
import time


def read_control(output_json_path):
    if not output_json_path:
        return None
    path = Path(output_json_path).resolve().with_name("evaluation-control.json")
    if not path.is_file():
        # Requests made during training live at the run root, so the old
        # controller can still create the eval directory with exist_ok=False.
        path = path.parent.parent / "evaluation-control.json"
        if not path.is_file():
            return None
    control = json.loads(path.read_text())
    names = control.get("datasets")
    if not isinstance(names, list) or not names or not all(isinstance(n, str) and n for n in names) or len(set(names)) != len(names):
        raise ValueError(f"Invalid evaluation scope in {path}")
    return control


def stop_legacy_suite_if_requested(output_json_path, generated_path, model_name, cleanup):
    """An already-running old suite cannot reload its loop; stop at this exact boundary.

    Called by its newly launched scoring child, after scores have been written.
    Only a matching, explicitly armed PID may be stopped. A separate finalizer
    records the requested smaller scope after the legacy controller exits.
    """
    control = read_control(output_json_path)
    if not control or not control.get("legacy_suite_pid"):
        return False
    dataset = Path(generated_path).name.split("_pass", 1)[0]
    if dataset != control.get("stop_after"):
        return False
    if os.getppid() != control["legacy_suite_pid"]:
        return False
    import psutil

    parent = psutil.Process(control["legacy_suite_pid"])
    if abs(parent.create_time() - control["legacy_suite_created_at"]) > .01:
        raise RuntimeError("Evaluation controller PID was reused")
    command = parent.cmdline()
    if not any(Path(arg).name == "run_eval_suite.py" for arg in command):
        raise RuntimeError("Scope control does not refer to a math evaluation suite")
    result_index = command.index("--results_file")
    if Path(command[result_index + 1]).resolve() != Path(output_json_path).resolve():
        raise RuntimeError("Scope control points to another evaluation run")
    results = json.loads(Path(output_json_path).read_text())[model_name]
    for name in control["datasets"]:
        if f"{name}_pass16_generation_pass_16" not in results:
            raise RuntimeError(f"Cannot close scope before {name} has been scored")
    cleanup()
    marker = Path(output_json_path).resolve().with_name("scope-boundary.json")
    temp = marker.with_suffix(".tmp")
    temp.write_text(json.dumps({"completed_datasets": control["datasets"], "stopped_after": dataset,
                                "suite_pid": parent.pid, "time": time.time()}, indent=2) + "\n")
    temp.replace(marker)
    print(f"Requested evaluation scope completed: {control['datasets']}; ending legacy suite before next dataset.",
          flush=True)
    # Ray's SIGTERM handler raises SystemExit, so the suite runs its finally
    # block and releases generation servers. No SIGSTOP or model interruption.
    os.kill(parent.pid, signal.SIGTERM)
    return True
