from __future__ import annotations

from typing import Callable

import validation_prediction
import vali_predict_graph
import vali_predict_graph_r2
import vali_predict_graph_r3


def run_step(name: str, func: Callable[[], None]) -> None:
    print(f"\n=== Running: {name} ===")
    func()
    print(f"=== Finished: {name} ===")


def main() -> None:
    """Run prediction generation first, then create all validation figures."""

    run_step("validation_prediction", validation_prediction.main)
    run_step("vali_predict_graph", vali_predict_graph.main)
    run_step("vali_predict_graph_r2", vali_predict_graph_r2.main)
    run_step("vali_predict_graph_r3", vali_predict_graph_r3.main)

    print("\nAll steps completed.")


if __name__ == "__main__":
    main()