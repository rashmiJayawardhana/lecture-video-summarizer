from concurrent.futures import ThreadPoolExecutor, as_completed

from src.backend.services.job_service import update_job_status
from src.backend.workers.module_runners import (
    run_module1,
    run_module2,
    run_module3,
    run_module4,
)


def process_pipeline(job_id: str, video_path: str, output_dir: str) -> None:
    try:
        update_job_status(
            job_id,
            status="processing",
            module1="running",
            module2="running",
            module3="running",
            module4="waiting"
        )

        results = {}
        failures = {}

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(run_module1, video_path, output_dir): "module1",
                executor.submit(run_module2, video_path, output_dir): "module2",
                executor.submit(run_module3, video_path, output_dir): "module3",
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

        # Every module has now reported its own real, final status above -
        # only decide the overall job outcome once all three are actually done.
        if failures:
            error_summary = "; ".join(f"{m}: {err}" for m, err in failures.items())
            update_job_status(
                job_id,
                status="failed",
                error=f"Module(s) failed: {error_summary}"
            )
            return

        update_job_status(
            job_id,
            status="module4_processing",
            module4="running"
        )

        final_video, final_json = run_module4(
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