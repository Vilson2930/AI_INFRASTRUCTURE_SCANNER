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
    # ============================================================
# SCORE DE TENDÊNCIA
# ============================================================

def score_price_vs_sma_20(
    close: float,
    sma_20: float,
) -> float:
    """
    Avalia a posição do preço em relação à SMA20.
    """

    if not (
        is_valid_number(close)
        and is_valid_number(sma_20)
    ):
        return 35.0

    close = float(close)
    sma_20 = float(sma_20)

    distance = (
        close / sma_20 - 1
    ) * 100

    if 0 <= distance <= 5:
        return 100.0

    if -3 <= distance < 0:
        return 88.0

    if 5 < distance <= 10:
        return 78.0

    if -7 <= distance < -3:
        return 68.0

    if 10 < distance <= 18:
        return 55.0

    if -12 <= distance < -7:
        return 42.0

    if distance > 18:
        return 30.0

    return 20.0


def score_price_vs_sma_50(
    close: float,
    sma_50: float,
) -> float:
    """
    Avalia a posição do preço em relação à SMA50.
    """

    if not (
        is_valid_number(close)
        and is_valid_number(sma_50)
    ):
        return 35.0

    close = float(close)
    sma_50 = float(sma_50)

    distance = (
        close / sma_50 - 1
    ) * 100

    if 0 <= distance <= 8:
        return 100.0

    if -5 <= distance < 0:
        return 88.0

    if 8 < distance <= 15:
        return 80.0

    if -10 <= distance < -5:
        return 65.0

    if 15 < distance <= 25:
        return 55.0

    if -18 <= distance < -10:
        return 42.0

    if distance > 25:
        return 30.0

    return 18.0


def score_price_vs_sma_200(
    close: float,
    sma_200: float,
) -> float:
    """
    Avalia a tendência estrutural pela SMA200.
    """

    if not (
        is_valid_number(close)
        and is_valid_number(sma_200)
    ):
        return 35.0

    close = float(close)
    sma_200 = float(sma_200)

    distance = (
        close / sma_200 - 1
    ) * 100

    if 0 <= distance <= 25:
        return 100.0

    if -5 <= distance < 0:
        return 80.0

    if 25 < distance <= 50:
        return 72.0

    if -12 <= distance < -5:
        return 52.0

    if distance > 50:
        return 55.0

    if -20 <= distance < -12:
        return 32.0

    return 12.0


def score_moving_average_structure(
    sma_20: float,
    sma_50: float,
    sma_200: float,
) -> float:
    """
    Avalia o alinhamento das médias móveis.
    """

    if not all(
        is_valid_number(value)
        for value in [
            sma_20,
            sma_50,
            sma_200,
        ]
    ):
        return 35.0

    sma_20 = float(sma_20)
    sma_50 = float(sma_50)
    sma_200 = float(sma_200)

    if (
        sma_20 > sma_50
        and sma_50 > sma_200
    ):
        return 100.0

    if (
        sma_20 > sma_50
        and sma_50 <= sma_200
    ):
        return 72.0

    if (
        sma_20 <= sma_50
        and sma_50 > sma_200
    ):
        return 65.0

    if (
        sma_20 < sma_50
        and sma_50 < sma_200
    ):
        return 15.0

    return 40.0


def score_adx_direction(
    adx: float,
    plus_di: float,
    minus_di: float,
) -> float:
    """
    Avalia força e direção da tendência.
    """

    if not is_valid_number(adx):
        return 35.0

    adx = float(adx)

    directional_positive = (
        is_valid_number(plus_di)
        and is_valid_number(minus_di)
        and float(plus_di) > float(minus_di)
    )

    if adx >= 35:
        return (
            100.0
            if directional_positive
            else 45.0
        )

    if adx >= 25:
        return (
            92.0
            if directional_positive
            else 50.0
        )

    if adx >= 20:
        return (
            80.0
            if directional_positive
            else 48.0
        )

    if adx >= 15:
        return (
            62.0
            if directional_positive
            else 40.0
        )

    return (
        45.0
        if directional_positive
        else 25.0
    )


def calculate_trend_score(
    row: pd.Series,
) -> float:
    """
    Calcula o score de tendência.
    """

    score = weighted_average_available(
        values=[
            score_price_vs_sma_20(
                row.get("close"),
                row.get("sma_20"),
            ),
            score_price_vs_sma_50(
                row.get("close"),
                row.get("sma_50"),
            ),
            score_price_vs_sma_200(
                row.get("close"),
                row.get("sma_200"),
            ),
            score_moving_average_structure(
                row.get("sma_20"),
                row.get("sma_50"),
                row.get("sma_200"),
            ),
            score_adx_direction(
                row.get("adx_14"),
                row.get("plus_di_14"),
                row.get("minus_di_14"),
            ),
        ],
        weights=[
            0.15,
            0.20,
            0.25,
            0.20,
            0.20,
        ],
        fallback=40.0,
    )

    return clip_score(
        score
    )


# ============================================================
# SCORE DE VOLUME E FLUXO
# ============================================================

def score_relative_volume(
    value: float,
) -> float:
    """
    Avalia volume relativo de 20 dias.
    """

    if not is_valid_number(value):
        return 35.0

    value = float(value)

    if value >= 2.0:
        return 100.0

    if value >= 1.5:
        return 92.0

    if value >= 1.2:
        return 82.0

    if value >= 1.0:
        return 70.0

    if value >= 0.8:
        return 52.0

    if value >= 0.6:
        return 30.0

    return 10.0


def score_mfi_flow(
    value: float,
) -> float:
    """
    Avalia o Money Flow Index para timing.
    """

    if not is_valid_number(value):
        return 35.0

    value = float(value)

    if 45 <= value <= 65:
        return 100.0

    if 35 <= value < 45:
        return 82.0

    if 65 < value <= 75:
        return 78.0

    if 25 <= value < 35:
        return 60.0

    if 75 < value <= 85:
        return 52.0

    if value < 25:
        return 28.0

    return 18.0


def score_cmf_flow(
    value: float,
) -> float:
    """
    Avalia o Chaikin Money Flow.
    """

    if not is_valid_number(value):
        return 35.0

    value = float(value)

    if value >= 0.20:
        return 100.0

    if value >= 0.10:
        return 90.0

    if value >= 0.03:
        return 78.0

    if value >= 0:
        return 65.0

    if value >= -0.05:
        return 45.0

    if value >= -0.15:
        return 25.0

    return 8.0


def score_obv_position(
    obv: float,
    obv_average: float,
) -> float:
    """
    Avalia se o OBV está acima da média.
    """

    if not (
        is_valid_number(obv)
        and is_valid_number(obv_average)
    ):
        return 35.0

    obv = float(obv)
    obv_average = float(obv_average)

    if obv > obv_average:
        return 90.0

    if obv == obv_average:
        return 55.0

    return 25.0


def score_vwap_position(
    close: float,
    vwap: float,
) -> float:
    """
    Avalia se o preço está acima do VWAP móvel.
    """

    if not (
        is_valid_number(close)
        and is_valid_number(vwap)
    ):
        return 35.0

    close = float(close)
    vwap = float(vwap)

    distance = (
        close / vwap - 1
    ) * 100

    if 0 <= distance <= 5:
        return 100.0

    if -2 <= distance < 0:
        return 80.0

    if 5 < distance <= 10:
        return 72.0

    if -5 <= distance < -2:
        return 55.0

    if distance > 10:
        return 45.0

    return 25.0


def calculate_volume_flow_score(
    row: pd.Series,
) -> float:
    """
    Calcula o score de volume e fluxo.
    """

    score = weighted_average_available(
        values=[
            score_relative_volume(
                row.get(
                    "volume_relativo_20d"
                )
            ),
            score_mfi_flow(
                row.get(
                    "mfi_14"
                )
            ),
            score_cmf_flow(
                row.get(
                    "cmf_20"
                )
            ),
            score_obv_position(
                row.get(
                    "obv"
                ),
                row.get(
                    "obv_sma_20"
                ),
            ),
            score_vwap_position(
                row.get(
                    "close"
                ),
                row.get(
                    "vwap_20"
                ),
            ),
        ],
        weights=[
            0.25,
            0.20,
            0.20,
            0.20,
            0.15,
        ],
        fallback=40.0,
    )

    return clip_score(
        score
    )


# ============================================================
# SCORE DE RISCO
# ============================================================

def score_atr_percent(
    value: float,
) -> float:
    """
    Avalia a volatilidade pelo ATR percentual.
    """

    if not is_valid_number(value):
        return 35.0

    value = float(value)

    if 2.0 <= value <= 5.0:
        return 100.0

    if 1.0 <= value < 2.0:
        return 80.0

    if 5.0 < value <= 7.0:
        return 78.0

    if 0.5 <= value < 1.0:
        return 58.0

    if 7.0 < value <= 9.0:
        return 55.0

    if 9.0 < value <= 12.0:
        return 30.0

    if value > 12.0:
        return 10.0

    return 35.0


def score_risk_by_rsi(
    value: float,
) -> float:
    """
    Penaliza extremos do RSI.
    """

    if not is_valid_number(value):
        return 35.0

    value = float(value)

    if 40 <= value <= 65:
        return 100.0

    if 35 <= value < 40:
        return 78.0

    if 65 < value <= 70:
        return 70.0

    if 30 <= value < 35:
        return 55.0

    if 70 < value <= 75:
        return 42.0

    if value < 30:
        return 45.0

    return 15.0


def score_risk_by_distance_sma_200(
    value: float,
) -> float:
    """
    Avalia risco de extensão ou fraqueza estrutural.
    """

    if not is_valid_number(value):
        return 35.0

    value = float(value)

    if 0 <= value <= 25:
        return 100.0

    if -5 <= value < 0:
        return 82.0

    if 25 < value <= 45:
        return 72.0

    if -12 <= value < -5:
        return 55.0

    if 45 < value <= 70:
        return 48.0

    if -20 <= value < -12:
        return 30.0

    if value > 70:
        return 25.0

    return 12.0


def score_risk_by_drawdown(
    value: float,
) -> float:
    """
    Avalia o risco pela distância da máxima anual.
    """

    if not is_valid_number(value):
        return 35.0

    value = float(value)

    if -30 <= value <= -8:
        return 100.0

    if -8 < value <= -3:
        return 78.0

    if -40 <= value < -30:
        return 72.0

    if -3 < value <= 0:
        return 55.0

    if -50 <= value < -40:
        return 45.0

    if value < -50:
        return 20.0

    return 35.0


def calculate_risk_score(
    row: pd.Series,
) -> float:
    """
    Calcula o score de risco.
    """

    score = weighted_average_available(
        values=[
            score_atr_percent(
                row.get(
                    "atr_percentual"
                )
            ),
            score_risk_by_rsi(
                row.get(
                    "rsi_14"
                )
            ),
            score_risk_by_distance_sma_200(
                row.get(
                    "distancia_sma_200"
                )
            ),
            score_risk_by_drawdown(
                row.get(
                    "distancia_maxima_52s"
                )
            ),
        ],
        weights=[
            0.40,
            0.20,
            0.20,
            0.20,
        ],
        fallback=40.0,
    )

    return clip_score(
        score
    )


# ============================================================
# PENALIDADES TÉCNICAS
# ============================================================

def calculate_technical_penalties(
    row: pd.Series,
) -> tuple[float, str]:
    """
    Aplica penalidades independentes do score bruto.
    """

    penalty = 0.0
    reasons: list[str] = []

    rsi = row.get(
        "rsi_14"
    )

    atr_percent = row.get(
        "atr_percentual"
    )

    distance_sma_200 = row.get(
        "distancia_sma_200"
    )

    distance_high = row.get(
        "distancia_maxima_52s"
    )

    macd_hist = row.get(
        "macd_hist"
    )

    macd_hist_change = row.get(
        "macd_hist_variacao"
    )

    adx = row.get(
        "adx_14"
    )

    plus_di = row.get(
        "plus_di_14"
    )

    minus_di = row.get(
        "minus_di_14"
    )

    volume_relative = row.get(
        "volume_relativo_20d"
    )

    cmf = row.get(
        "cmf_20"
    )

    return_1m = row.get(
        "retorno_1m"
    )

    if (
        is_valid_number(rsi)
        and float(rsi) >= 75
    ):
        penalty += 10
        reasons.append(
            "RSI excessivamente elevado"
        )

    if (
        is_valid_number(atr_percent)
        and float(atr_percent) > 10
    ):
        penalty += 8
        reasons.append(
            "ATR excessivamente elevado"
        )

    if (
        is_valid_number(distance_sma_200)
        and float(distance_sma_200) < -20
    ):
        penalty += 12
        reasons.append(
            "tendência estrutural fraca"
        )

    if (
        is_valid_number(distance_high)
        and float(distance_high) < -55
    ):
        penalty += 8
        reasons.append(
            "drawdown muito profundo"
        )

    if (
        is_valid_number(macd_hist)
        and is_valid_number(
            macd_hist_change
        )
        and float(macd_hist) < 0
        and float(macd_hist_change) < 0
    ):
        penalty += 8
        reasons.append(
            "MACD negativo e deteriorando"
        )

    if (
        is_valid_number(adx)
        and is_valid_number(plus_di)
        and is_valid_number(minus_di)
        and float(adx) >= 25
        and float(minus_di) > float(plus_di)
    ):
        penalty += 10
        reasons.append(
            "tendência de baixa com ADX forte"
        )

    if (
        is_valid_number(volume_relative)
        and float(volume_relative) < 0.60
    ):
        penalty += 5
        reasons.append(
            "volume relativo muito fraco"
        )

    if (
        is_valid_number(cmf)
        and float(cmf) < -0.15
    ):
        penalty += 8
        reasons.append(
            "fluxo de capital fortemente negativo"
        )

    if (
        is_valid_number(return_1m)
        and float(return_1m) > 30
    ):
        penalty += 6
        reasons.append(
            "movimento de curto prazo esticado"
        )

    return (
        float(penalty),
        "; ".join(reasons),
    )
    # ============================================================
# CLASSIFICAÇÃO TÉCNICA
# ============================================================

def classify_technical_score(
    score: float,
) -> str:
    """
    Classifica o Technical Entry Score.
    """

    if score >= 85:
        return "EXCELENTE"

    if score >= 75:
        return "BOA"

    if score >= 65:
        return "MODERADA"

    if score >= 55:
        return "NEUTRA"

    if score >= 45:
        return "FRACA"

    return "MUITO FRACA"


# ============================================================
# CONFIRMAÇÕES TÉCNICAS
# ============================================================

def bullish_macd_confirmation(
    row: pd.Series,
) -> bool:
    """
    Confirma melhora do MACD.

    Aceita:
    - MACD acima da linha de sinal; ou
    - histograma negativo, porém em recuperação.
    """

    macd = row.get(
        "macd"
    )

    macd_signal = row.get(
        "macd_signal"
    )

    macd_histogram = row.get(
        "macd_hist"
    )

    macd_variation = row.get(
        "macd_hist_variacao"
    )

    macd_above_signal = (
        is_valid_number(macd)
        and is_valid_number(macd_signal)
        and float(macd) > float(macd_signal)
    )

    histogram_recovering = (
        is_valid_number(macd_histogram)
        and is_valid_number(macd_variation)
        and float(macd_histogram) < 0
        and float(macd_variation) > 0
    )

    histogram_positive = (
        is_valid_number(macd_histogram)
        and float(macd_histogram) > 0
    )

    return bool(
        macd_above_signal
        or histogram_recovering
        or histogram_positive
    )


def trend_confirmation(
    row: pd.Series,
) -> bool:
    """
    Confirma tendência minimamente favorável.
    """

    close = row.get(
        "close"
    )

    sma_50 = row.get(
        "sma_50"
    )

    sma_200 = row.get(
        "sma_200"
    )

    plus_di = row.get(
        "plus_di_14"
    )

    minus_di = row.get(
        "minus_di_14"
    )

    price_above_sma_200 = (
        is_valid_number(close)
        and is_valid_number(sma_200)
        and float(close) > float(sma_200)
    )

    price_near_sma_50 = (
        is_valid_number(close)
        and is_valid_number(sma_50)
        and float(close)
        >= float(sma_50) * 0.95
    )

    directional_positive = (
        is_valid_number(plus_di)
        and is_valid_number(minus_di)
        and float(plus_di) >= float(minus_di)
    )

    return bool(
        price_above_sma_200
        and price_near_sma_50
        and directional_positive
    )


def volume_confirmation(
    row: pd.Series,
) -> bool:
    """
    Confirma fluxo de capital suficiente.
    """

    relative_volume = row.get(
        "volume_relativo_20d"
    )

    cmf = row.get(
        "cmf_20"
    )

    obv = row.get(
        "obv"
    )

    obv_average = row.get(
        "obv_sma_20"
    )

    close = row.get(
        "close"
    )

    vwap = row.get(
        "vwap_20"
    )

    relative_volume_ok = (
        is_valid_number(relative_volume)
        and float(relative_volume) >= 0.80
    )

    cmf_ok = (
        is_valid_number(cmf)
        and float(cmf) >= -0.05
    )

    obv_ok = (
        is_valid_number(obv)
        and is_valid_number(obv_average)
        and float(obv) > float(obv_average)
    )

    vwap_ok = (
        is_valid_number(close)
        and is_valid_number(vwap)
        and float(close) > float(vwap)
    )

    confirmations = sum(
        [
            relative_volume_ok,
            cmf_ok,
            obv_ok,
            vwap_ok,
        ]
    )

    return confirmations >= 2


def risk_confirmation(
    row: pd.Series,
) -> bool:
    """
    Confirma que a volatilidade não está excessiva.
    """

    atr_percent = row.get(
        "atr_percentual"
    )

    rsi = row.get(
        "rsi_14"
    )

    atr_ok = (
        is_valid_number(atr_percent)
        and float(atr_percent) <= 10
    )

    rsi_ok = (
        is_valid_number(rsi)
        and 28 <= float(rsi) <= 75
    )

    return bool(
        atr_ok
        and rsi_ok
    )


# ============================================================
# DIAGNÓSTICO DA ENTRADA
# ============================================================

def diagnose_technical_entry(
    row: pd.Series,
) -> str:
    """
    Gera diagnóstico prático para swing trade.
    """

    final_score = row.get(
        "technical_entry_score"
    )

    discount_score = row.get(
        "score_discount"
    )

    momentum_score = row.get(
        "score_momentum"
    )

    trend_score = row.get(
        "score_trend"
    )

    volume_score = row.get(
        "score_volume_flow"
    )

    risk_score = row.get(
        "score_risk"
    )

    macd_confirmed = bool(
        row.get(
            "macd_confirmation",
            False,
        )
    )

    trend_confirmed = bool(
        row.get(
            "trend_confirmation",
            False,
        )
    )

    volume_confirmed = bool(
        row.get(
            "volume_confirmation",
            False,
        )
    )

    technical_entry_approved = bool(
        row.get(
            "technical_entry_approved",
            False,
        )
    )

    if technical_entry_approved:
        return "ENTRADA CONFIRMADA"

    if (
        is_valid_number(final_score)
        and float(final_score) >= 70
        and macd_confirmed
        and volume_confirmed
        and not trend_confirmed
    ):
        return "REVERSÃO EM FORMAÇÃO"

    if (
        is_valid_number(discount_score)
        and float(discount_score) >= 80
        and is_valid_number(momentum_score)
        and float(momentum_score) < 60
    ):
        return "DESCONTADA — AGUARDAR GATILHO"

    if (
        is_valid_number(momentum_score)
        and float(momentum_score) >= 75
        and is_valid_number(trend_score)
        and float(trend_score) >= 70
        and not volume_confirmed
    ):
        return "MOMENTUM FORTE — VOLUME NÃO CONFIRMADO"

    if (
        is_valid_number(volume_score)
        and float(volume_score) >= 75
        and is_valid_number(momentum_score)
        and float(momentum_score) < 60
    ):
        return "ACUMULAÇÃO — AGUARDAR MOMENTUM"

    if (
        is_valid_number(trend_score)
        and float(trend_score) < 45
    ):
        return "TENDÊNCIA DESFAVORÁVEL"

    if (
        is_valid_number(risk_score)
        and float(risk_score) < 40
    ):
        return "RISCO ELEVADO"

    if (
        is_valid_number(final_score)
        and float(final_score) >= 60
    ):
        return "NEUTRA / OBSERVAÇÃO"

    return "SEM ENTRADA TÉCNICA"


# ============================================================
# MOTIVOS DE APROVAÇÃO E REPROVAÇÃO
# ============================================================

def list_positive_factors(
    row: pd.Series,
) -> str:
    """
    Lista os principais fatores positivos da ação.
    """

    factors: list[str] = []

    if (
        is_valid_number(
            row.get("score_discount")
        )
        and float(
            row["score_discount"]
        ) >= 75
    ):
        factors.append(
            "desconto técnico favorável"
        )

    if (
        is_valid_number(
            row.get("score_momentum")
        )
        and float(
            row["score_momentum"]
        ) >= 70
    ):
        factors.append(
            "momentum positivo"
        )

    if (
        is_valid_number(
            row.get("score_trend")
        )
        and float(
            row["score_trend"]
        ) >= 70
    ):
        factors.append(
            "tendência favorável"
        )

    if (
        is_valid_number(
            row.get("score_volume_flow")
        )
        and float(
            row["score_volume_flow"]
        ) >= 70
    ):
        factors.append(
            "entrada de capital"
        )

    if bool(
        row.get(
            "macd_confirmation",
            False,
        )
    ):
        factors.append(
            "MACD favorável"
        )

    if bool(
        row.get(
            "trend_confirmation",
            False,
        )
    ):
        factors.append(
            "tendência confirmada"
        )

    if bool(
        row.get(
            "volume_confirmation",
            False,
        )
    ):
        factors.append(
            "volume confirmado"
        )

    return "; ".join(
        factors
    )


def list_pending_conditions(
    row: pd.Series,
) -> str:
    """
    Lista o que ainda impede uma entrada confirmada.
    """

    pending: list[str] = []

    if (
        not is_valid_number(
            row.get(
                "technical_entry_score"
            )
        )
        or float(
            row.get(
                "technical_entry_score",
                0,
            )
        ) < 70
    ):
        pending.append(
            "Technical Score abaixo de 70"
        )

    if not bool(
        row.get(
            "macd_confirmation",
            False,
        )
    ):
        pending.append(
            "MACD não confirmado"
        )

    if not bool(
        row.get(
            "trend_confirmation",
            False,
        )
    ):
        pending.append(
            "tendência não confirmada"
        )

    if not bool(
        row.get(
            "volume_confirmation",
            False,
        )
    ):
        pending.append(
            "volume/fluxo insuficiente"
        )

    if not bool(
        row.get(
            "risk_confirmation",
            False,
        )
    ):
        pending.append(
            "risco técnico elevado"
        )

    penalty_reasons = row.get(
        "technical_penalty_reasons",
        ""
    )

    if (
        isinstance(
            penalty_reasons,
            str,
        )
        and penalty_reasons.strip()
    ):
        pending.append(
            penalty_reasons
        )

    return "; ".join(
        pending
    )


# ============================================================
# FUNÇÃO CENTRAL DO SCORE
# ============================================================

def calculate_final_technical_score(
    row: pd.Series,
) -> float:
    """
    Calcula o Technical Entry Score bruto.
    """

    total_weight = (
        WEIGHT_DISCOUNT
        +
        WEIGHT_MOMENTUM
        +
        WEIGHT_TREND
        +
        WEIGHT_VOLUME
        +
        WEIGHT_RISK
    )

    if total_weight <= 0:
        raise ValueError(
            "A soma dos pesos técnicos "
            "deve ser maior que zero."
        )

    raw_score = (
        row["score_discount"]
        * WEIGHT_DISCOUNT
        +
        row["score_momentum"]
        * WEIGHT_MOMENTUM
        +
        row["score_trend"]
        * WEIGHT_TREND
        +
        row["score_volume_flow"]
        * WEIGHT_VOLUME
        +
        row["score_risk"]
        * WEIGHT_RISK
    ) / total_weight

    return clip_score(
        raw_score
    )


# ============================================================
# CLASSE PRINCIPAL
# ============================================================

class TechnicalScore:
    """
    Motor responsável pelo Technical Entry Score.
    """

    def __init__(
        self,
        output_file: str | Path = TECHNICAL_OUTPUT_FILE,
        minimum_entry_score: float = 70.0,
    ) -> None:
        self.output_file = Path(
            output_file
        )

        self.minimum_entry_score = float(
            minimum_entry_score
        )

        self.history = pd.DataFrame()

        self.latest = pd.DataFrame()

        self.failures = pd.DataFrame()

    def calculate(
        self,
        indicators: pd.DataFrame,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Calcula o Technical Entry Score para o último
        registro disponível de cada ticker.

        Parameters
        ----------
        indicators:
            DataFrame produzido por technical_indicators.py.

        save:
            Quando True, salva data/technical_score.csv.

        Returns
        -------
        pandas.DataFrame
            Ranking técnico mais recente.
        """

        validated_data = validate_input(
            indicators
        )

        self.history = validated_data.copy()

        latest = (
            validated_data
            .sort_values(
                [
                    "ticker",
                    "date",
                ]
            )
            .groupby(
                "ticker",
                as_index=False,
            )
            .tail(1)
            .reset_index(drop=True)
        )

        if latest.empty:
            raise RuntimeError(
                "Nenhum registro recente foi encontrado."
            )

        # ----------------------------------------------------
        # COMPONENTES DO SCORE
        # ----------------------------------------------------

        latest[
            "score_discount"
        ] = latest.apply(
            calculate_discount_score,
            axis=1,
        )

        latest[
            "score_momentum"
        ] = latest.apply(
            calculate_momentum_score,
            axis=1,
        )

        latest[
            "score_trend"
        ] = latest.apply(
            calculate_trend_score,
            axis=1,
        )

        latest[
            "score_volume_flow"
        ] = latest.apply(
            calculate_volume_flow_score,
            axis=1,
        )

        latest[
            "score_risk"
        ] = latest.apply(
            calculate_risk_score,
            axis=1,
        )

        # ----------------------------------------------------
        # SCORE BRUTO
        # ----------------------------------------------------

        latest[
            "technical_entry_score_raw"
        ] = latest.apply(
            calculate_final_technical_score,
            axis=1,
        )

        # ----------------------------------------------------
        # PENALIDADES
        # ----------------------------------------------------

        penalties = latest.apply(
            calculate_technical_penalties,
            axis=1,
        )

        latest[
            "technical_penalty"
        ] = [
            result[0]
            for result in penalties
        ]

        latest[
            "technical_penalty_reasons"
        ] = [
            result[1]
            for result in penalties
        ]

        # ----------------------------------------------------
        # SCORE FINAL
        # ----------------------------------------------------

        latest[
            "technical_entry_score"
        ] = (
            latest[
                "technical_entry_score_raw"
            ]
            -
            latest[
                "technical_penalty"
            ]
        ).clip(
            lower=0,
            upper=100,
        ).round(2)

        # ----------------------------------------------------
        # CONFIRMAÇÕES
        # ----------------------------------------------------

        latest[
            "macd_confirmation"
        ] = latest.apply(
            bullish_macd_confirmation,
            axis=1,
        )

        latest[
            "trend_confirmation"
        ] = latest.apply(
            trend_confirmation,
            axis=1,
        )

        latest[
            "volume_confirmation"
        ] = latest.apply(
            volume_confirmation,
            axis=1,
        )

        latest[
            "risk_confirmation"
        ] = latest.apply(
            risk_confirmation,
            axis=1,
        )

        # ----------------------------------------------------
        # ENTRADA APROVADA
        # ----------------------------------------------------

        latest[
            "technical_entry_approved"
        ] = (
            latest[
                "technical_entry_score"
            ] >= self.minimum_entry_score
        ) & (
            latest[
                "macd_confirmation"
            ]
        ) & (
            latest[
                "trend_confirmation"
            ]
        ) & (
            latest[
                "volume_confirmation"
            ]
        ) & (
            latest[
                "risk_confirmation"
            ]
        )

        # ----------------------------------------------------
        # CLASSIFICAÇÃO
        # ----------------------------------------------------

        latest[
            "technical_classification"
        ] = latest[
            "technical_entry_score"
        ].apply(
            classify_technical_score
        )

        latest[
            "technical_diagnosis"
        ] = latest.apply(
            diagnose_technical_entry,
            axis=1,
        )

        latest[
            "positive_technical_factors"
        ] = latest.apply(
            list_positive_factors,
            axis=1,
        )

        latest[
            "pending_technical_conditions"
        ] = latest.apply(
            list_pending_conditions,
            axis=1,
        )

        # ----------------------------------------------------
        # COMPATIBILIDADE COM O COLAB
        # ----------------------------------------------------

        latest[
            "score_desconto"
        ] = latest[
            "score_discount"
        ]

        latest[
            "score_momentum"
        ] = latest[
            "score_momentum"
        ]

        latest[
            "score_tendencia"
        ] = latest[
            "score_trend"
        ]

        latest[
            "score_volume_fluxo"
        ] = latest[
            "score_volume_flow"
        ]

        latest[
            "score_risco"
        ] = latest[
            "score_risk"
        ]

        latest[
            "penalidade_tecnica"
        ] = latest[
            "technical_penalty"
        ]

        latest[
            "motivos_penalidade"
        ] = latest[
            "technical_penalty_reasons"
        ]

        latest[
            "classificacao_tecnica"
        ] = latest[
            "technical_classification"
        ]

        latest[
            "diagnostico_entrada"
        ] = latest[
            "technical_diagnosis"
        ]

        latest[
            "entrada_tecnica_aprovada"
        ] = latest[
            "technical_entry_approved"
        ]

        # ----------------------------------------------------
        # ORDENAÇÃO
        # ----------------------------------------------------

        latest = (
            latest
            .sort_values(
                [
                    "technical_entry_approved",
                    "technical_entry_score",
                    "score_volume_flow",
                    "score_momentum",
                    "score_trend",
                ],
                ascending=[
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
            )
            .reset_index(drop=True)
        )

        latest[
            "technical_ranking"
        ] = np.arange(
            1,
            len(latest) + 1,
        )

        latest[
            "ranking_tecnico"
        ] = latest[
            "technical_ranking"
        ]

        self.latest = latest

        if save:
            self.save()

        self.print_summary()

        return self.latest.copy()

    def save(self) -> None:
        """
        Salva o ranking técnico.
        """

        if self.latest.empty:
            raise ValueError(
                "Não há resultado técnico para salvar."
            )

        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.latest.to_csv(
            self.output_file,
            index=False,
            encoding="utf-8-sig",
        )

    def get_latest(self) -> pd.DataFrame:
        """
        Retorna o ranking técnico completo.
        """

        return self.latest.copy()

    def get_approved_entries(
        self,
    ) -> pd.DataFrame:
        """
        Retorna somente entradas técnicas aprovadas.
        """

        if self.latest.empty:
            return pd.DataFrame()

        return (
            self.latest.loc[
                self.latest[
                    "technical_entry_approved"
                ]
            ]
            .copy()
            .reset_index(drop=True)
        )

    def get_best_by_sector(
        self,
    ) -> pd.DataFrame:
        """
        Retorna a melhor configuração técnica de cada setor.
        """

        if self.latest.empty:
            return pd.DataFrame()

        sector_column = (
            "setor"
            if "setor" in self.latest.columns
            else "segmento"
        )

        if sector_column not in self.latest.columns:
            raise KeyError(
                "A coluna de setor não foi encontrada."
            )

        return (
            self.latest
            .sort_values(
                [
                    "technical_entry_approved",
                    "technical_entry_score",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .groupby(
                sector_column,
                as_index=False,
            )
            .first()
            .sort_values(
                [
                    "technical_entry_approved",
                    "technical_entry_score",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .reset_index(drop=True)
        )

    def print_summary(self) -> None:
        """
        Exibe o resumo do Technical Entry Score.
        """

        total = len(
            self.latest
        )

        approved = (
            int(
                self.latest[
                    "technical_entry_approved"
                ].sum()
            )
            if not self.latest.empty
            else 0
        )

        print()
        print("=" * 110)
        print(
            "TECHNICAL ENTRY SCORE — "
            "SWING TRADE DE ATÉ 6 MESES"
        )
        print("=" * 110)

        print(
            f"Ações analisadas: {total}"
        )

        print(
            f"Entradas tecnicamente aprovadas: "
            f"{approved}"
        )

        print(
            f"Score mínimo de entrada: "
            f"{self.minimum_entry_score:.2f}"
        )

        print(
            f"Arquivo: {self.output_file}"
        )

        if not self.latest.empty:
            print()
            print(
                "TOP 15 — MELHORES "
                "CONFIGURAÇÕES TÉCNICAS"
            )

            columns = [
                "technical_ranking",
                "ticker",
                "setor",
                "close",
                "technical_entry_score",
                "technical_classification",
                "technical_diagnosis",
                "technical_entry_approved",
                "score_discount",
                "score_momentum",
                "score_trend",
                "score_volume_flow",
                "score_risk",
                "technical_penalty",
                "rsi_14",
                "macd_hist",
                "adx_14",
                "atr_percentual",
                "distancia_maxima_52s",
                "volume_relativo_20d",
                "technical_penalty_reasons",
            ]

            available_columns = [
                column
                for column in columns
                if column in self.latest.columns
            ]

            print(
                self.latest[
                    available_columns
                ]
                .head(15)
                .to_string(
                    index=False
                )
            )

        print("=" * 110)


# ============================================================
# FUNÇÃO SIMPLIFICADA
# ============================================================

def calculate_technical_score(
    indicators: pd.DataFrame,
    save: bool = True,
    minimum_entry_score: float = 70.0,
) -> pd.DataFrame:
    """
    Interface simplificada para calcular o score técnico.
    """

    engine = TechnicalScore(
        minimum_entry_score=minimum_entry_score
    )

    return engine.calculate(
        indicators=indicators,
        save=save,
    )


# ============================================================
# EXECUÇÃO DIRETA
# ============================================================

if __name__ == "__main__":

    if not Path(
        INDICATOR_FILE
    ).exists():
        raise FileNotFoundError(
            "O arquivo indicadores.csv não foi encontrado. "
            "Execute primeiro technical_indicators.py."
        )

    indicator_data = pd.read_csv(
        INDICATOR_FILE
    )

    technical_engine = TechnicalScore(
        minimum_entry_score=70.0
    )

    technical_ranking = (
        technical_engine.calculate(
            indicators=indicator_data,
            save=True,
        )
    )

    print()
    print(
        "MELHOR CONFIGURAÇÃO TÉCNICA "
        "DE CADA SETOR"
    )

    best_by_sector = (
        technical_engine
        .get_best_by_sector()
    )

    columns_to_display = [
        "ticker",
        "setor",
        "close",
        "technical_entry_score",
        "technical_classification",
        "technical_diagnosis",
        "distancia_maxima_52s",
        "rsi_14",
        "macd_hist",
        "volume_relativo_20d",
    ]

    available_columns = [
        column
        for column in columns_to_display
        if column in best_by_sector.columns
    ]

    print(
        best_by_sector[
            available_columns
        ]
        .to_string(
            index=False
        )
    )
