import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.backend.services.job_service import update_job_status
from src.backend.workers.module_runners import (
    run_module1,
    run_module2,
    run_module3,
    run_module4,
)

# Demo-only toggle: restrict which modules actually run, e.g. for a
# module-by-module demo where running all three (esp. Module 1/3's
# frame-heavy processing) would take too long. Comma-separated, e.g.
# "module2" or "module1,module3". Defaults to all three - unset/leave out
# of .env entirely for the normal full pipeline.
ENABLED_MODULES = {
    m.strip() for m in os.getenv("ENABLED_MODULES", "module1,module2,module3").split(",") if m.strip()
}
_RUNNERS = {"module1": run_module1, "module2": run_module2, "module3": run_module3}


def process_pipeline(job_id: str, video_path: str, output_dir: str) -> None:
    try:
        initial_status = {
            name: ("running" if name in ENABLED_MODULES else "skipped")
            for name in ("module1", "module2", "module3")
        }
        update_job_status(
            job_id,
            status="processing",
            module4="waiting",
            **initial_status,
        )

        results = {}
        failures = {}

        with ThreadPoolExecutor(max_workers=max(1, len(ENABLED_MODULES))) as executor:
            futures = {
                executor.submit(_RUNNERS[name], video_path, output_dir): name
                for name in ("module1", "module2", "module3")
                if name in ENABLED_MODULES
            }

            for future in as_completed(futures):
                module_name = futures[future]

                try:
                    output_json = future.result()
                    results[module_name] = output_json

                    update_job_status(
                        job_id,
                        **{module_name: "completed"}
                    )

                except Exception as e:
                    failures[module_name] = str(e)

                    update_job_status(
                        job_id,
                        **{module_name: "failed"}
                    )

        # Every enabled module has now reported its own real, final status above -
        # only decide the overall job outcome once they're actually done.
        if failures:
            error_summary = "; ".join(f"{m}: {err}" for m, err in failures.items())
            update_job_status(
                job_id,
                status="failed",
                error=f"Module(s) failed: {error_summary}"
            )
            return

        # Module 4 needs all three module outputs to fuse - if this run only
        # enabled a subset (e.g. ENABLED_MODULES=module2 for a Module 2-only
        # demo), stop here rather than attempting fusion with missing inputs.
        if ENABLED_MODULES != {"module1", "module2", "module3"}:
            update_job_status(
                job_id,
                status="completed",
                module4="skipped",
            )
            return

        update_job_status(
            job_id,
            status="module4_processing",
            module4="running"
        )

        final_video, final_json = run_module4(
            video_path=video_path,
            module1_json=results["module1"],
            module2_json=results["module2"],
            module3_json=results["module3"],
            output_dir=output_dir
        )

        update_job_status(
            job_id,
            status="completed",
            module4="completed",
            final_video=final_video,
            final_json=final_json
        )

    except Exception as e:
        update_job_status(
            job_id,
            status="failed",
            error=str(e)
        )