# ============================================================
# AI INFRASTRUCTURE SCANNER
# technical_score.py
#
# PARTE 1 DE 3
#
# Technical Entry Score para swing trade de até 6 meses.
#
# Componentes:
# - Score de desconto
# - Score de momentum
# - Score de tendência
# - Score de volume e fluxo
# - Score de risco
# - Penalidades técnicas
# - Classificação e diagnóstico
# ============================================================

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import (
    INDICATOR_FILE,
    MAX_SCORE,
    SHOW_PROGRESS,
    WEIGHT_DISCOUNT,
    WEIGHT_MOMENTUM,
    WEIGHT_TREND,
    WEIGHT_VOLUME,
    WEIGHT_RISK,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURAÇÕES
# ============================================================

TECHNICAL_OUTPUT_FILE = (
    Path(INDICATOR_FILE).parent
    / "technical_score.csv"
)

MINIMUM_REQUIRED_COLUMNS = {
    "ticker",
    "date",
    "close",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "macd_hist_variacao",
    "adx_14",
    "plus_di_14",
    "minus_di_14",
    "mfi_14",
    "atr_percentual",
    "sma_20",
    "sma_50",
    "sma_200",
    "distancia_sma_20",
    "distancia_sma_50",
    "distancia_sma_200",
    "distancia_maxima_52s",
    "posicao_intervalo_52s",
    "volume_relativo_20d",
    "retorno_1m",
    "retorno_3m",
    "retorno_6m",
    "cmf_20",
    "obv",
    "obv_sma_20",
    "vwap_20",
}


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def is_valid_number(
    value: Any,
) -> bool:
    """
    Verifica se um valor é numérico e finito.
    """

    try:
        return bool(
            np.isfinite(
                float(value)
            )
        )

    except (
        TypeError,
        ValueError,
    ):
        return False


def clip_score(
    value: float,
) -> float:
    """
    Limita o score entre 0 e MAX_SCORE.
    """

    if not is_valid_number(value):
        return 0.0

    return float(
        np.clip(
            float(value),
            0,
            MAX_SCORE,
        )
    )


def weighted_average_available(
    values: list[float],
    weights: list[float],
    fallback: float = 40.0,
) -> float:
    """
    Calcula média ponderada ignorando valores inválidos.
    """

    valid_values: list[float] = []
    valid_weights: list[float] = []

    for value, weight in zip(
        values,
        weights,
    ):

        if is_valid_number(value):

            valid_values.append(
                float(value)
            )

            valid_weights.append(
                float(weight)
            )

    if not valid_values:
        return fallback

    return float(
        np.average(
            valid_values,
            weights=valid_weights,
        )
    )


def normalize_columns(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Padroniza os nomes das colunas.
    """

    df = data.copy()

    df.columns = [
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        for column in df.columns
    ]

    aliases = {
        "atr_percent": "atr_percentual",
        "distance_sma_20": "distancia_sma_20",
        "distance_sma_50": "distancia_sma_50",
        "distance_sma_200": "distancia_sma_200",
        "distance_from_52w_high":
            "distancia_maxima_52s",
        "position_in_52w_range":
            "posicao_intervalo_52s",
        "relative_volume_20d":
            "volume_relativo_20d",
        "return_1m": "retorno_1m",
        "return_3m": "retorno_3m",
        "return_6m": "retorno_6m",
        "macd_hist_change":
            "macd_hist_variacao",
    }

    return df.rename(
        columns={
            column: aliases.get(
                column,
                column,
            )
            for column in df.columns
        }
    )


def validate_input(
    indicators: pd.DataFrame,
) -> pd.DataFrame:
    """
    Valida o DataFrame produzido pelo módulo
    technical_indicators.py.
    """

    if not isinstance(
        indicators,
        pd.DataFrame,
    ):
        raise TypeError(
            "Os indicadores devem ser fornecidos "
            "em um pandas DataFrame."
        )

    if indicators.empty:
        raise ValueError(
            "O DataFrame de indicadores está vazio."
        )

    df = normalize_columns(
        indicators
    )

    missing_columns = (
        MINIMUM_REQUIRED_COLUMNS
        .difference(
            df.columns
        )
    )

    if missing_columns:
        raise KeyError(
            "Colunas obrigatórias ausentes: "
            f"{sorted(missing_columns)}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df["ticker"] = (
        df["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    numeric_columns = [
        "close",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "macd_hist_variacao",
        "adx_14",
        "plus_di_14",
        "minus_di_14",
        "mfi_14",
        "atr_percentual",
        "sma_20",
        "sma_50",
        "sma_200",
        "distancia_sma_20",
        "distancia_sma_50",
        "distancia_sma_200",
        "distancia_maxima_52s",
        "posicao_intervalo_52s",
        "volume_relativo_20d",
        "retorno_1m",
        "retorno_3m",
        "retorno_6m",
        "cmf_20",
        "obv",
        "obv_sma_20",
        "vwap_20",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df.dropna(
            subset=[
                "ticker",
                "date",
                "close",
            ]
        )
        .sort_values(
            [
                "ticker",
                "date",
            ]
        )
        .drop_duplicates(
            subset=[
                "ticker",
                "date",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError(
            "Nenhuma linha válida permaneceu "
            "após a validação."
        )

    return df


# ============================================================
# SCORE DE DESCONTO
# ============================================================

def score_distance_from_high(
    value: float,
) -> float:
    """
    Avalia a distância da máxima de 52 semanas.

    O melhor cenário não é obrigatoriamente a ação mais
    distante da máxima, mas um pullback relevante dentro
    de uma estrutura ainda recuperável.
    """

    if not is_valid_number(value):
        return 35.0

    value = float(value)

    if -30 <= value <= -18:
        return 100.0

    if -18 < value <= -10:
        return 90.0

    if -40 <= value < -30:
        return 82.0

    if -10 < value <= -5:
        return 72.0

    if -50 <= value < -40:
        return 58.0

    if -5 < value <= 0:
        return 45.0

    if -60 <= value < -50:
        return 35.0

    if value < -60:
        return 15.0

    return 25.0


def score_distance_sma_20(
    value: float,
) -> float:
    """
    Avalia a distância para a média móvel de 20 dias.
    """

    if not is_valid_number(value):
        return 40.0

    value = float(value)

    if -3 <= value <= 3:
        return 100.0

    if -7 <= value < -3:
        return 85.0

    if 3 < value <= 7:
        return 78.0

    if -12 <= value < -7:
        return 65.0

    if 7 < value <= 12:
        return 55.0

    if -20 <= value < -12:
        return 42.0

    if value > 12:
        return 25.0

    return 20.0


def score_distance_sma_50(
    value: float,
) -> float:
    """
    Avalia a distância para a média móvel de 50 dias.
    """

    if not is_valid_number(value):
        return 40.0

    value = float(value)

    if -5 <= value <= 5:
        return 100.0

    if -10 <= value < -5:
        return 88.0

    if 5 < value <= 10:
        return 82.0

    if -18 <= value < -10:
        return 68.0

    if 10 < value <= 18:
        return 58.0

    if -30 <= value < -18:
        return 45.0

    if value > 18:
        return 30.0

    return 20.0


def score_position_52w_range(
    value: float,
) -> float:
    """
    Avalia a posição do preço dentro do intervalo anual.
    """

    if not is_valid_number(value):
        return 40.0

    value = float(value)

    if 35 <= value <= 65:
        return 100.0

    if 20 <= value < 35:
        return 85.0

    if 65 < value <= 80:
        return 80.0

    if 10 <= value < 20:
        return 65.0

    if 80 < value <= 90:
        return 60.0

    if value < 10:
        return 35.0

    return 40.0


def calculate_discount_score(
    row: pd.Series,
) -> float:
    """
    Calcula o score de desconto técnico.
    """

    score = weighted_average_available(
        values=[
            score_distance_from_high(
                row.get(
                    "distancia_maxima_52s"
                )
            ),
            score_distance_sma_20(
                row.get(
                    "distancia_sma_20"
                )
            ),
            score_distance_sma_50(
                row.get(
                    "distancia_sma_50"
                )
            ),
            score_position_52w_range(
                row.get(
                    "posicao_intervalo_52s"
                )
            ),
        ],
        weights=[
            0.45,
            0.15,
            0.25,
            0.15,
        ],
        fallback=40.0,
    )

    return clip_score(
        score
    )


# ============================================================
# SCORE DE MOMENTUM
# ============================================================

def score_rsi(
    value: float,
) -> float:
    """
    Avalia o RSI para entrada de swing.

    A faixa ideal prioriza momentum positivo sem
    sobrecompra excessiva.
    """

    if not is_valid_number(value):
        return 35.0

    value = float(value)

    if 45 <= value <= 58:
        return 100.0

    if 40 <= value < 45:
        return 88.0

    if 58 < value <= 65:
        return 82.0

    if 35 <= value < 40:
        return 70.0

    if 65 < value <= 70:
        return 58.0

    if 30 <= value < 35:
        return 52.0

    if 70 < value <= 75:
        return 35.0

    if value < 30:
        return 42.0

    return 15.0


def score_macd_histogram(
    histogram: float,
    variation: float,
) -> float:
    """
    Avalia o histograma do MACD e sua aceleração.
    """

    histogram_valid = (
        is_valid_number(histogram)
    )

    variation_valid = (
        is_valid_number(variation)
    )

    if not histogram_valid:
        return 35.0

    histogram = float(histogram)

    variation = (
        float(variation)
        if variation_valid
        else 0.0
    )

    if (
        histogram > 0
        and variation > 0
    ):
        return 100.0

    if (
        histogram > 0
        and variation <= 0
    ):
        return 72.0

    if (
        histogram <= 0
        and variation > 0
    ):
        return 68.0

    if (
        histogram < 0
        and variation == 0
    ):
        return 35.0

    return 15.0


def score_macd_position(
    macd: float,
    signal: float,
) -> float:
    """
    Avalia a posição do MACD em relação à linha de sinal.
    """

    if not (
        is_valid_number(macd)
        and is_valid_number(signal)
    ):
        return 35.0

    macd = float(macd)
    signal = float(signal)

    if macd > signal:
        return 100.0

    distance = abs(
        macd - signal
    )

    base = max(
        abs(signal),
        1e-9,
    )

    relative_distance = (
        distance / base
    )

    if relative_distance <= 0.10:
        return 70.0

    if relative_distance <= 0.25:
        return 50.0

    return 20.0


def score_short_term_return(
    value: float,
) -> float:
    """
    Avalia o retorno de um mês.
    """

    if not is_valid_number(value):
        return 40.0

    value = float(value)

    if 2 <= value <= 12:
        return 100.0

    if -2 <= value < 2:
        return 82.0

    if 12 < value <= 20:
        return 72.0

    if -6 <= value < -2:
        return 65.0

    if 20 < value <= 30:
        return 50.0

    if -12 <= value < -6:
        return 42.0

    if value > 30:
        return 25.0

    return 15.0


def score_medium_term_return(
    value: float,
) -> float:
    """
    Avalia o retorno de três meses.
    """

    if not is_valid_number(value):
        return 40.0

    value = float(value)

    if 5 <= value <= 30:
        return 100.0

    if 0 <= value < 5:
        return 82.0

    if 30 < value <= 55:
        return 78.0

    if -10 <= value < 0:
        return 60.0

    if 55 < value <= 90:
        return 52.0

    if -20 <= value < -10:
        return 35.0

    if value > 90:
        return 30.0

    return 15.0


def calculate_momentum_score(
    row: pd.Series,
) -> float:
    """
    Calcula o score de momentum técnico.
    """

    score = weighted_average_available(
        values=[
            score_rsi(
                row.get(
                    "rsi_14"
                )
            ),
            score_macd_histogram(
                row.get(
                    "macd_hist"
                ),
                row.get(
                    "macd_hist_variacao"
                ),
            ),
            score_macd_position(
                row.get(
                    "macd"
                ),
                row.get(
                    "macd_signal"
                ),
            ),
            score_short_term_return(
                row.get(
                    "retorno_1m"
                )
            ),
            score_medium_term_return(
                row.get(
                    "retorno_3m"
                )
            ),
        ],
        weights=[
            0.25,
            0.30,
            0.15,
            0.12,
            0.18,
        ],
        fallback=40.0,
    )

    return clip_score(
        score
    )
