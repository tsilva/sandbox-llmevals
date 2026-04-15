from __future__ import annotations

import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.metrics import apply_metric
from lighteval.models.model_output import ModelResponse
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.tasks.requests import Doc, SamplingMethod

from llmevals.runtime import (
    api_key_from_env,
    build_litellm_model_config,
    ensure_lm_studio,
    pretty_prompt,
    start_server,
)

from .adapters import adapt_sample_result
from .schemas import RunDetail, RunEvent, RunStartRequest, SampleResult
from .store import RunStore, utc_now


class LightevalWebRunner:
    execution_mode = "process"

    def _build_pipeline(self, run_id: str, settings: dict[str, Any], store: RunStore) -> Pipeline:
        benchmark_cfg = settings["benchmark"]
        model_cfg = settings["model"]
        lighteval_dir = store.lighteval_dir(run_id)
        store.write_log(run_id, "stdout", f"Starting LM Studio server for model {model_cfg['id']}")

        start_server(
            Path(settings["paths"]["root_dir"]),
            model_id=model_cfg["id"],
            identifier=model_cfg["identifier"],
            port=int(model_cfg["lm_studio"]["port"]),
            bind_address=model_cfg["lm_studio"]["bind_address"],
            parallel=int(model_cfg["lm_studio"]["parallel"]),
            context_length=int(model_cfg["lm_studio"]["context_length"]),
            gpu=model_cfg["lm_studio"]["gpu"],
        )

        api_key = api_key_from_env()
        ensure_lm_studio(model_cfg["lm_studio"]["base_url"], api_key)
        store.write_log(run_id, "stdout", f"LM Studio ready at {model_cfg['lm_studio']['base_url']}")

        evaluation_tracker = EvaluationTracker(
            output_dir=str(lighteval_dir),
            save_details=True,
        )
        pipeline_params = PipelineParameters(
            launcher_type=ParallelismManager.NONE,
            job_id=run_id,
            dataset_loading_processes=1,
            custom_tasks_directory=None,
            num_fewshot_seeds=1,
            max_samples=benchmark_cfg.get("limit"),
            remove_reasoning_tags=True,
            reasoning_tags="[('<think>', '</think>')]",
            load_tasks_multilingual=False,
        )
        pipeline = Pipeline(
            tasks=benchmark_cfg["task_name"],
            pipeline_parameters=pipeline_params,
            evaluation_tracker=evaluation_tracker,
            model_config=build_litellm_model_config(settings, api_key),
        )
        store.write_log(run_id, "stdout", f"Initialized pipeline for task {benchmark_cfg['task_name']}")
        pipeline.evaluation_tracker.general_config_logger.log_args_info(
            num_fewshot_seeds=pipeline.pipeline_parameters.num_fewshot_seeds,
            max_samples=pipeline.pipeline_parameters.max_samples,
            job_id=str(run_id),
        )
        return pipeline

    def _run_doc(self, pipeline: Pipeline, doc: Doc) -> tuple[ModelResponse | None, dict[str, Any]]:
        task = pipeline.tasks_dict[doc.task_name]
        metric_outputs: dict[str, Any] = {}
        representative_response: ModelResponse | None = None

        sampling_method_responses: dict[SamplingMethod, list[ModelResponse]] = {}
        for sampling_method in doc.sampling_methods:
            match sampling_method:
                case SamplingMethod.GENERATIVE:
                    responses = pipeline.model.greedy_until([doc])
                case SamplingMethod.LOGPROBS:
                    responses = pipeline.model.loglikelihood([doc])
                case SamplingMethod.PERPLEXITY:
                    responses = pipeline.model.loglikelihood_rolling([doc])
                case _:
                    continue
            sampling_method_responses[sampling_method] = responses

        pipeline._post_process_outputs(sampling_method_responses)

        for sampling_method, responses in sampling_method_responses.items():
            metrics = [metric for metric in task.metrics if metric.category == sampling_method]
            outputs = apply_metric(docs=[doc], responses=responses, metrics=metrics)
            if not outputs:
                continue
            output = outputs[0]
            response = responses[0] if responses else None
            metric_outputs.update(output)
            pipeline.evaluation_tracker.metrics_logger.log(doc.task_name, output)
            if response is not None:
                pipeline.evaluation_tracker.details_logger.log(doc.task_name, doc, response, output)
                if representative_response is None:
                    representative_response = response

        return representative_response, metric_outputs

    def _primary_metric_name(self, pipeline: Pipeline) -> str | None:
        first_task = next(iter(pipeline.tasks_dict.values()), None)
        if first_task is None or not first_task.metrics:
            return None
        return first_task.metrics[0].metric_name

    def _current_score(self, pipeline: Pipeline, primary_metric: str | None) -> float | None:
        if primary_metric is None:
            return None
        pipeline.evaluation_tracker.metrics_logger.metric_aggregated.clear()
        pipeline.evaluation_tracker.metrics_logger.aggregate(pipeline.tasks_dict, bootstrap_iters=0)
        return pipeline.evaluation_tracker.metrics_logger.metric_aggregated.get("all", {}).get(primary_metric)

    def _official_results_path(self, run_id: str, store: RunStore) -> str | None:
        lighteval_dir = store.lighteval_dir(run_id)
        results = sorted(lighteval_dir.rglob("results_*.json"))
        return str(results[-1]) if results else None

    def run(
        self,
        run_id: str,
        request: RunStartRequest,
        settings: dict[str, Any],
        store: RunStore,
        publish: Callable[[RunEvent], None],
    ) -> None:
        pipeline = self._build_pipeline(run_id, settings, store)
        primary_metric = self._primary_metric_name(pipeline)
        docs = [doc for task_docs in pipeline.documents_dict.values() for doc in task_docs]
        store.write_log(run_id, "stdout", f"Loaded {len(docs)} samples for this run")
        detail = store.update_run(run_id, total_samples=len(docs))
        publish(RunEvent(type="progress", run_id=run_id, data=detail.model_dump(mode="json")))

        try:
            for sample_index, doc in enumerate(docs):
                response, metrics = self._run_doc(pipeline, doc)
                raw_prompt = pretty_prompt(response.input if response else None) or doc.query
                raw_response = ""
                if response is not None:
                    raw_response = "\n\n".join(response.final_text or response.text)

                parsed_response, expected_response, success = adapt_sample_result(
                    benchmark_name=settings["benchmark"]["name"],
                    doc=doc,
                    raw_response=raw_response,
                    metrics=metrics,
                )
                sample = SampleResult(
                    sample_index=sample_index,
                    sample_id=doc.id,
                    task_name=doc.task_name,
                    raw_prompt=raw_prompt,
                    raw_response=raw_response,
                    parsed_response=parsed_response,
                    expected_response=expected_response,
                    success=success,
                    metrics=metrics,
                    completed_at=utc_now(),
                )
                store.append_sample(run_id, sample)
                publish(RunEvent(type="sample_completed", run_id=run_id, data=sample.model_dump(mode="json")))
                store.write_log(run_id, "stdout", f"Completed sample {sample_index + 1}/{len(docs)}")

                current_score = self._current_score(pipeline, primary_metric)
                detail = store.update_run(
                    run_id,
                    completed_samples=sample_index + 1,
                    score=current_score,
                    score_metric=primary_metric,
                )
                publish(RunEvent(type="progress", run_id=run_id, data=detail.model_dump(mode="json")))

            pipeline.evaluation_tracker.general_config_logger.log_end_time()
            pipeline.evaluation_tracker.metrics_logger.metric_aggregated.clear()
            pipeline.evaluation_tracker.metrics_logger.aggregate(
                task_dict=pipeline.tasks_dict,
                bootstrap_iters=pipeline.pipeline_parameters.bootstrap_iters,
            )
            pipeline.evaluation_tracker.details_logger.aggregate()
            final_results = pipeline.evaluation_tracker.generate_final_dict()
            pipeline.save_and_push_results()
            store.write_log(run_id, "stdout", "Saved final lighteval results")

            results_path = store.write_results(run_id, final_results)
            detail = store.update_run(
                run_id,
                status="finished",
                completed_at=utc_now(),
                score=final_results.get("results", {}).get("all", {}).get(primary_metric) if primary_metric else None,
                score_metric=primary_metric,
                results_path=results_path,
                official_results_path=self._official_results_path(run_id, store),
            )
            publish(RunEvent(type="run_finished", run_id=run_id, data=detail.model_dump(mode="json")))
        except Exception as exc:
            store.write_log(run_id, "stderr", traceback.format_exc())
            detail = store.update_run(
                run_id,
                status="failed",
                completed_at=utc_now(),
                error_message=str(exc),
            )
            publish(RunEvent(type="run_failed", run_id=run_id, data=detail.model_dump(mode="json")))
            raise
        finally:
            try:
                pipeline.model.cleanup()
            except Exception:
                pass
