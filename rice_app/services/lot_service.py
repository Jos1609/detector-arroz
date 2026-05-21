from __future__ import annotations

from typing import Iterable

MINIMUM_SAMPLES = 3
SIMILARITY_TOLERANCE_PCT = 60.0


def calculate_sample_percentages(counts: dict[str, int]) -> dict[str, float]:
    total = max(0, int(sum(counts.values())))
    if total == 0:
        return {"healthy_pct": 0.0, "chalky_pct": 0.0, "broken_pct": 0.0}
    return {
        "healthy_pct": (counts.get("sano", 0) / total) * 100,
        "chalky_pct": (counts.get("panza_blanca", 0) / total) * 100,
        "broken_pct": (counts.get("quebrado", 0) / total) * 100,
    }


def _spread(values: list[float]) -> float:
    return max(values) - min(values) if values else 0.0


def _sample_value(sample, key: str, default=0):
    try:
        value = sample[key]
    except (KeyError, IndexError, TypeError):
        return default
    return default if value is None else value


def summarize_lot(total_bags: float, samples: Iterable[dict]) -> dict[str, float]:
    sample_list = list(samples)
    total_samples = len(sample_list)
    healthy_values = [float(sample["healthy_pct"]) for sample in sample_list]
    chalky_values = [float(sample["chalky_pct"]) for sample in sample_list]
    broken_values = [float(sample["broken_pct"]) for sample in sample_list]

    total_healthy = sum(int(_sample_value(sample, "healthy_count", 0)) for sample in sample_list)
    total_chalky = sum(int(_sample_value(sample, "chalky_count", 0)) for sample in sample_list)
    total_broken = sum(int(_sample_value(sample, "broken_count", 0)) for sample in sample_list)
    total_grains = total_healthy + total_chalky + total_broken

    if total_grains > 0:
        healthy_avg = (total_healthy / total_grains) * 100.0
        chalky_avg = (total_chalky / total_grains) * 100.0
        broken_avg = (total_broken / total_grains) * 100.0
    else:
        healthy_avg = 0.0
        chalky_avg = 0.0
        broken_avg = 0.0

    healthy_spread = _spread(healthy_values)
    chalky_spread = _spread(chalky_values)
    broken_spread = _spread(broken_values)
    has_exact_samples = total_samples == MINIMUM_SAMPLES
    
    current_tolerance = SIMILARITY_TOLERANCE_PCT
    if total_grains < 45:
        current_tolerance *= 2.0
        
    max_spread = max(healthy_spread, chalky_spread, broken_spread) if total_samples > 0 else 0.0
    percentages_are_similar = has_exact_samples and max_spread <= current_tolerance
    analysis_ready = has_exact_samples and percentages_are_similar

    if total_samples < MINIMUM_SAMPLES:
        analysis_message = f"Faltan {MINIMUM_SAMPLES - total_samples} pruebas para completar el analisis."
    elif total_samples > MINIMUM_SAMPLES:
        analysis_message = "Este lote supera las 3 pruebas esperadas. Revisa el historial antes de usar el analisis final."
    elif percentages_are_similar:
        analysis_message = "Las 3 pruebas mantienen porcentajes similares. El analisis final del lote ya es confiable."
    else:
        analysis_message = (
            "Las 3 pruebas no son lo bastante similares. Conviene revisar la toma de muestras antes de confiar en el analisis."
        )

    return {
        "sample_count": total_samples,
        "required_samples": MINIMUM_SAMPLES,
        "remaining_samples": max(0, MINIMUM_SAMPLES - total_samples),
        "can_add_sample": total_samples < MINIMUM_SAMPLES,
        "meets_minimum_samples": total_samples >= MINIMUM_SAMPLES,
        "has_exact_samples": has_exact_samples,
        "percentages_are_similar": percentages_are_similar,
        "analysis_ready": analysis_ready,
        "analysis_message": analysis_message,
        "similarity_tolerance_pct": SIMILARITY_TOLERANCE_PCT,
        "healthy_avg_pct": healthy_avg,
        "chalky_avg_pct": chalky_avg,
        "broken_avg_pct": broken_avg,
        "healthy_spread_pct": healthy_spread,
        "chalky_spread_pct": chalky_spread,
        "broken_spread_pct": broken_spread,
        "healthy_estimated_bags": total_bags * healthy_avg / 100,
        "chalky_estimated_bags": total_bags * chalky_avg / 100,
        "broken_estimated_bags": total_bags * broken_avg / 100,
    }
