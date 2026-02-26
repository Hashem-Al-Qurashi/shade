"""Experiment runner: baseline -> abliterate -> steer -> benchmark -> Pareto.

Usage:
    python -m experiments.run_comparison experiments/configs/gpt2.yaml
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import optuna
import torch
import torch.nn.functional as F

from shade.benchmark import BenchmarkResult, run_benchmark
from shade.results import save_result

logger = logging.getLogger("shade.experiment")


def run_baseline(
    model: object,
    evaluator: object,
    model_name: str,
    run_gsm8k: bool = False,
    gsm8k_limit: int = 50,
    run_perplexity: bool = False,
    perplexity_limit: int = 50,
) -> BenchmarkResult:
    """Step 2: Run baseline benchmark on the unmodified model."""
    logger.info("Running baseline benchmark...")
    start = time.time()

    result = run_benchmark(
        model=model,
        evaluator=evaluator,
        model_name=f"{model_name} (baseline)",
        run_gsm8k=run_gsm8k,
        gsm8k_limit=gsm8k_limit,
        run_perplexity=run_perplexity,
        perplexity_limit=perplexity_limit,
    )

    elapsed = time.time() - start
    logger.info(
        "Baseline: refusals=%d/%d, KL=%.4f, perplexity=%s (%.1fs)",
        result.refusals,
        result.total_prompts,
        result.kl_divergence,
        f"{result.perplexity:.2f}" if result.perplexity else "N/A",
        elapsed,
    )
    return result


def run_abliteration(
    model: object,
    evaluator: object,
    model_name: str,
    n_trials: int = 5,
    n_startup_trials: int = 3,
    run_gsm8k: bool = False,
    gsm8k_limit: int = 50,
    run_perplexity: bool = False,
    perplexity_limit: int = 50,
) -> BenchmarkResult:
    """Step 3-4: Run Optuna abliteration search, then benchmark the best trial."""
    from shade.model import AbliterationParameters
    from shade.utils import load_prompts

    logger.info("Running abliteration optimization (%d trials)...", n_trials)

    # Compute refusal directions
    settings = model.settings
    good_prompts = load_prompts(settings, settings.good_prompts)
    bad_prompts = load_prompts(settings, settings.bad_prompts)

    good_residuals = model.get_residuals_batched(good_prompts)
    bad_residuals = model.get_residuals_batched(bad_prompts)

    refusal_directions = F.normalize(
        bad_residuals.mean(dim=0) - good_residuals.mean(dim=0), p=2, dim=1
    )

    del good_residuals, bad_residuals

    last_layer_index = len(model.get_layers()) - 1
    best_trial_data: dict = {
        "refusals": float("inf"),
        "kl": float("inf"),
        "params": None,
        "direction_index": None,
    }

    def objective(trial):
        direction_scope = trial.suggest_categorical(
            "direction_scope", ["global", "per layer"]
        )
        direction_index = trial.suggest_float(
            "direction_index", 0.4 * last_layer_index, 0.9 * last_layer_index
        )
        if direction_scope == "per layer":
            direction_index = None

        parameters = {}
        for component in model.get_abliterable_components():
            max_weight = trial.suggest_float(f"{component}.max_weight", 0.8, 1.5)
            max_weight_position = trial.suggest_float(
                f"{component}.max_weight_position",
                0.6 * last_layer_index,
                1.0 * last_layer_index,
            )
            min_weight_frac = trial.suggest_float(f"{component}.min_weight", 0.0, 1.0)
            min_weight_distance = trial.suggest_float(
                f"{component}.min_weight_distance", 1.0, 0.6 * last_layer_index
            )
            parameters[component] = AbliterationParameters(
                max_weight=max_weight,
                max_weight_position=max_weight_position,
                min_weight=(min_weight_frac * max_weight),
                min_weight_distance=min_weight_distance,
            )

        model.reset_model()
        model.abliterate(refusal_directions, direction_index, parameters)
        score, kl_divergence, refusals = evaluator.get_score()

        trial.set_user_attr("kl_divergence", kl_divergence)
        trial.set_user_attr("refusals", refusals)

        logger.info(
            "  Trial %d: refusals=%d, KL=%.4f",
            trial.number + 1,
            refusals,
            kl_divergence,
        )

        # Track best by lowest refusals, then lowest KL
        if (refusals, kl_divergence) < (
            best_trial_data["refusals"],
            best_trial_data["kl"],
        ):
            from dataclasses import asdict

            best_trial_data["refusals"] = refusals
            best_trial_data["kl"] = kl_divergence
            best_trial_data["params"] = {k: asdict(v) for k, v in parameters.items()}
            best_trial_data["direction_index"] = direction_index

        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=n_startup_trials,
            multivariate=True,
        ),
        directions=["minimize", "minimize"],
    )
    study.optimize(objective, n_trials=n_trials)

    # Restore best trial and benchmark it
    logger.info(
        "Restoring best trial (refusals=%d, KL=%.4f)...",
        best_trial_data["refusals"],
        best_trial_data["kl"],
    )
    model.reset_model()
    if best_trial_data["params"] is not None:
        params = {
            k: AbliterationParameters(**v) for k, v in best_trial_data["params"].items()
        }
        model.abliterate(refusal_directions, best_trial_data["direction_index"], params)

    result = run_benchmark(
        model=model,
        evaluator=evaluator,
        model_name=f"{model_name} (abliterated)",
        run_gsm8k=run_gsm8k,
        gsm8k_limit=gsm8k_limit,
        run_perplexity=run_perplexity,
        perplexity_limit=perplexity_limit,
    )

    logger.info(
        "Abliterated: refusals=%d/%d, KL=%.4f, perplexity=%s",
        result.refusals,
        result.total_prompts,
        result.kl_divergence,
        f"{result.perplexity:.2f}" if result.perplexity else "N/A",
    )
    return result


def run_steering_sweep(
    model: object,
    evaluator: object,
    model_name: str,
    layers: list[int],
    multipliers: list[float],
    batch_size: int = 4,
    run_gsm8k: bool = False,
    gsm8k_limit: int = 50,
    run_perplexity: bool = False,
    perplexity_limit: int = 50,
) -> list[BenchmarkResult]:
    """Step 5-6: Reset model, train steering vector, sweep multipliers."""
    from shade.steering import train_steering_vector_from_prompts
    from shade.utils import load_prompts

    logger.info("Training steering vector (layers=%s)...", layers)
    model.reset_model()

    settings = model.settings
    harmless = load_prompts(settings, settings.good_evaluation_prompts)
    harmful = load_prompts(settings, settings.bad_evaluation_prompts)

    base_model = model.model
    if hasattr(base_model, "base_model"):
        base_model = base_model.base_model

    sv = train_steering_vector_from_prompts(
        model=base_model,
        tokenizer=model.tokenizer,
        harmless_prompts=harmless,
        harmful_prompts=harmful,
        layers=layers,
        batch_size=batch_size,
        show_progress=False,
    )

    results = []
    for mult in multipliers:
        logger.info("Benchmarking steering multiplier x%.1f...", mult)
        with sv.apply(base_model, multiplier=mult):
            # Reinitialize evaluator inside the steering context
            from shade.evaluator import Evaluator

            steered_eval = Evaluator(settings, model)
            result = run_benchmark(
                model=model,
                evaluator=steered_eval,
                model_name=f"{model_name} (steered x{mult})",
                run_gsm8k=run_gsm8k,
                gsm8k_limit=gsm8k_limit,
                run_perplexity=run_perplexity,
                perplexity_limit=perplexity_limit,
            )
        results.append(result)
        logger.info(
            "  Steered x%.1f: refusals=%d/%d, KL=%.4f",
            mult,
            result.refusals,
            result.total_prompts,
            result.kl_divergence,
        )

    return results


def run_experiment(config_path: str) -> None:
    """Run the full experiment pipeline from a YAML config."""
    from experiments.config import load_config, seed_everything
    from shade.compare import format_comparison_table
    from shade.config import Settings
    from shade.evaluator import Evaluator
    from shade.model import Model

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config(config_path)
    logger.info("Loaded config for model: %s", config.model)

    # Create results directory
    model_slug = config.model.replace("/", "--")
    results_dir = Path("results") / "experiments" / model_slug
    results_dir.mkdir(parents=True, exist_ok=True)

    # Seed for reproducibility
    seed_everything(config.seed)
    logger.info("Seeded with %d", config.seed)

    # Step 1: Load model
    logger.info("Step 1/9: Loading model %s...", config.model)
    settings_kwargs: dict = {"model": config.model, "_cli_parse_args": False}
    if config.quantization == "bnb_4bit":
        settings_kwargs["quantization"] = "bnb_4bit"
    settings = Settings(**settings_kwargs)
    model = Model(settings)
    torch.set_grad_enabled(False)

    # Step 2: Baseline benchmark
    baseline_path = results_dir / "baseline.json"
    if baseline_path.exists():
        logger.info("Step 2/9: Baseline already exists, skipping.")
        from shade.results import load_result

        baseline = load_result(baseline_path)
    else:
        logger.info("Step 2/9: Running baseline benchmark...")
        evaluator = Evaluator(settings, model)
        baseline = run_baseline(
            model=model,
            evaluator=evaluator,
            model_name=config.model,
            run_gsm8k=config.benchmark.gsm8k,
            gsm8k_limit=config.benchmark.gsm8k_limit,
            run_perplexity=config.benchmark.perplexity,
            perplexity_limit=config.benchmark.perplexity_limit,
        )
        save_result(baseline, baseline_path)
        logger.info("Saved baseline to %s", baseline_path)

    # Step 3-4: Abliterate + benchmark
    abliterated_path = results_dir / "abliterated.json"
    if abliterated_path.exists():
        logger.info("Steps 3-4/9: Abliterated result already exists, skipping.")
        from shade.results import load_result

        abliterated = load_result(abliterated_path)
    else:
        logger.info("Steps 3-4/9: Running abliteration optimization...")
        evaluator = Evaluator(settings, model)
        abliterated = run_abliteration(
            model=model,
            evaluator=evaluator,
            model_name=config.model,
            n_trials=config.abliteration.n_trials,
            n_startup_trials=config.abliteration.n_startup_trials,
            run_gsm8k=config.benchmark.gsm8k,
            gsm8k_limit=config.benchmark.gsm8k_limit,
            run_perplexity=config.benchmark.perplexity,
            perplexity_limit=config.benchmark.perplexity_limit,
        )
        save_result(abliterated, abliterated_path)
        logger.info("Saved abliterated result to %s", abliterated_path)

    # Step 5-6: Steering sweep
    steered_results = []
    all_steered_exist = all(
        (results_dir / f"steered_{mult}.json").exists()
        for mult in config.steering.multipliers
    )
    if all_steered_exist:
        logger.info("Steps 5-6/9: Steered results already exist, skipping.")
        from shade.results import load_result

        for mult in config.steering.multipliers:
            steered_results.append(load_result(results_dir / f"steered_{mult}.json"))
    else:
        logger.info("Steps 5-6/9: Running steering sweep...")
        evaluator = Evaluator(settings, model)
        steered_results = run_steering_sweep(
            model=model,
            evaluator=evaluator,
            model_name=config.model,
            layers=config.steering.layers,
            multipliers=config.steering.multipliers,
            batch_size=config.steering.batch_size,
            run_gsm8k=config.benchmark.gsm8k,
            gsm8k_limit=config.benchmark.gsm8k_limit,
            run_perplexity=config.benchmark.perplexity,
            perplexity_limit=config.benchmark.perplexity_limit,
        )
        for result, mult in zip(steered_results, config.steering.multipliers):
            path = results_dir / f"steered_{mult}.json"
            save_result(result, path)
            logger.info("Saved steered x%.1f result to %s", mult, path)

    # Step 7: Save comparison JSON
    logger.info("Step 7/9: Saving comparison...")
    all_results = [baseline, abliterated] + steered_results
    from shade.compare import format_comparison_json

    comparison_path = results_dir / "comparison.json"
    comparison_path.write_text(format_comparison_json(all_results))

    # Step 8: Generate Pareto frontier plot
    logger.info("Step 8/9: Generating Pareto frontier plot...")
    try:
        from shade.pareto import plot_pareto_frontier

        # Build data dicts that plot_pareto_frontier expects
        abliteration_data = {
            "refusals": [abliterated.refusals],
            "kl": [abliterated.kl_divergence],
        }
        steering_data = {
            "refusals": [r.refusals for r in steered_results],
            "kl": [r.kl_divergence for r in steered_results],
            "multipliers": config.steering.multipliers,
        }
        plot_pareto_frontier(
            abliteration_data=abliteration_data,
            steering_data=steering_data,
            total_prompts=baseline.total_prompts,
            model_name=config.model,
            save_path=str(results_dir / "pareto.png"),
        )
        logger.info("Saved Pareto plot to %s", results_dir / "pareto.png")
    except ImportError:
        logger.warning("matplotlib not available, skipping Pareto plot")

    # Step 9: Print comparison table
    logger.info("Step 9/9: Results summary")
    print("\n" + format_comparison_table(all_results) + "\n")
    logger.info("All results saved to %s", results_dir)


def main():
    """CLI entry point."""
    if len(sys.argv) != 2:
        print("Usage: python -m experiments.run_comparison <config.yaml>")
        sys.exit(1)
    run_experiment(sys.argv[1])


if __name__ == "__main__":
    main()
