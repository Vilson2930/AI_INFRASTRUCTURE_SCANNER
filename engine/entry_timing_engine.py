# ============================================================
# AI INFRASTRUCTURE SCANNER
# entry_timing_engine.py
#
# Motor exclusivo de timing de entrada.
#
# Objetivo:
# - separar qualidade da empresa de qualidade do ponto de entrada;
# - impedir compra após movimentos excessivamente esticados;
# - classificar a ação em:
#   * ENTRAR AGORA
#   * PRÉ-ENTRADA
#   * AGUARDAR PULLBACK
#   * AGUARDAR ROMPIMENTO
#   * MUITO ESTICADA
#   * ALTO RISCO
#   * LATERAL / OBSERVAÇÃO
#
# Entradas esperadas:
# DataFrame produzido por technical_indicators.py contendo,
# preferencialmente:
# - retorno_5d
# - retorno_10d
# - distancia_sma_10
# - distancia_sma_20
# - distancia_sma_50
# - distancia_maxima_20d
# - distancia_maxima_52s
# - rsi_14
# - macd_hist
# - macd_hist_variacao
# - adx_14
# - plus_di_14
# - minus_di_14
# - atr_percentual
# - volume_relativo_5d
# - volume_relativo_20d
# - cmf_20
# - weekly_extension_risk
# - parabolic_move_risk
# - pullback_required
# - extension_score
# - trend_quality_score
# - trend_structure_quality
# - candle_stability_score
# - pullback_quality_score
# - volatility_expansion_score
#
# Saídas principais:
# - entry_timing_score
# - timing_status
# - pullback_probability
# - parabolic_risk
# - acceleration_factor_5d
# - acceleration_factor_10d
# - explosion_score
# - extension_quality
# - trend_structure_timing_score
# - timing_confidence
# - timing_approved
# - timing_positive_factors
# - timing_pending_conditions
# - timing_rejection_reasons
# ============================================================

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import DATA_PATH

warnings.filterwarnings("ignore")


# ============================================================
# ARQUIVOS
# ============================================================

ENTRY_TIMING_OUTPUT_FILE = (
    Path(DATA_PATH)
    / "entry_timing_score.csv"
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

MAX_SCORE = 100.0

MIN_TIMING_SCORE = 70.0

MIN_ENTRY_SCORE = 75.0

MIN_CONFIRMATIONS = 2

MAX_ACCEPTABLE_PULLBACK_PROBABILITY = 65.0

MAX_ACCEPTABLE_ATR = 10.0

MAX_ACCEPTABLE_RSI = 74.0

MAX_ACCEPTABLE_DISTANCE_SMA20 = 12.0

MAX_ACCEPTABLE_RETURN_5D = 20.0

MAX_ACCEPTABLE_RETURN_10D = 35.0

MAX_ACCEPTABLE_ACCELERATION_5D = 3.00
MAX_ACCEPTABLE_ACCELERATION_10D = 4.50
MAX_ACCEPTABLE_EXPLOSION_SCORE = 72.0

STRONG_ACCELERATION_5D = 2.50
EXTREME_ACCELERATION_5D = 3.50

STRONG_ACCELERATION_10D = 3.50
EXTREME_ACCELERATION_10D = 5.00

MIN_TREND_QUALITY_FOR_ENTRY = 42.0
MIN_TREND_QUALITY_FOR_BONUS = 72.0
STRONG_TREND_QUALITY = 85.0


MINIMUM_REQUIRED_COLUMNS = {
    "ticker",
    "date",
    "close",
    "rsi_14",
    "macd_hist",
    "macd_hist_variacao",
    "adx_14",
    "atr_percentual",
    "distancia_sma_20",
    "distancia_sma_50",
    "distancia_maxima_52s",
    "volume_relativo_20d",
}


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def is_valid_number(
    value: Any,
) -> bool:
    """
    Verifica se o valor é numérico e finito.
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
    Limita uma nota entre 0 e 100.
    """

    if not is_valid_number(
        value
    ):
        return 0.0

    return float(
        np.clip(
            float(value),
            0,
            MAX_SCORE,
        )
    )


def boolean_value(
    value: Any,
) -> bool:
    """
    Converte formatos diversos para booleano.
    """

    if isinstance(
        value,
        bool,
    ):
        return value

    if value is None:
        return False

    if isinstance(
        value,
        str,
    ):
        return (
            value.strip().lower()
            in {
                "true",
                "1",
                "sim",
                "yes",
                "verdadeiro",
            }
        )

    try:
        return bool(
            value
        )

    except Exception:
        return False


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

        if is_valid_number(
            value
        ):

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
    Padroniza os nomes das colunas sem gerar duplicidades.
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
        "segmento":
            "setor",

        "empresa":
            "company",

        "nome_empresa":
            "company",

        "return_5d":
            "retorno_5d",

        "return_10d":
            "retorno_10d",

        "distance_sma_10":
            "distancia_sma_10",

        "distance_sma_20":
            "distancia_sma_20",

        "distance_sma_50":
            "distancia_sma_50",

        "distance_from_20d_high":
            "distancia_maxima_20d",

        "distance_from_52w_high":
            "distancia_maxima_52s",

        "relative_volume_5d":
            "volume_relativo_5d",

        "relative_volume_20d":
            "volume_relativo_20d",

        "macd_hist_change":
            "macd_hist_variacao",
    }

    for source_column, target_column in aliases.items():

        if source_column not in df.columns:
            continue

        if target_column in df.columns:
            df = df.drop(
                columns=[
                    source_column
                ]
            )

        else:
            df = df.rename(
                columns={
                    source_column:
                        target_column
                }
            )

    df = df.loc[
        :,
        ~df.columns.duplicated(
            keep="last"
        )
    ].copy()

    return df


def validate_input(
    indicators: pd.DataFrame,
) -> pd.DataFrame:
    """
    Valida os indicadores recebidos.
    """

    if not isinstance(
        indicators,
        pd.DataFrame,
    ):
        raise TypeError(
            "indicators deve ser um pandas DataFrame."
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
            "Colunas obrigatórias ausentes no Entry Timing Engine: "
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
        "retorno_5d",
        "retorno_10d",
        "distancia_sma_10",
        "distancia_sma_20",
        "distancia_sma_50",
        "distancia_maxima_20d",
        "distancia_maxima_52s",
        "rsi_14",
        "macd_hist",
        "macd_hist_variacao",
        "adx_14",
        "plus_di_14",
        "minus_di_14",
        "atr_percentual",
        "volume_relativo_5d",
        "volume_relativo_20d",
        "cmf_20",
        "extension_score",
        "trend_quality_score",
        "candle_stability_score",
        "pullback_quality_score",
        "volatility_expansion_score",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    boolean_columns = [
        "weekly_extension_risk",
        "parabolic_move_risk",
        "pullback_required",
    ]

    for column in boolean_columns:

        if column in df.columns:

            df[column] = (
                df[column]
                .apply(
                    boolean_value
                )
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
        .reset_index(
            drop=True
        )
    )

    if df.empty:
        raise ValueError(
            "Nenhuma linha válida permaneceu após a validação."
        )

    return df


# ============================================================
# SCORES DOS COMPONENTES
# ============================================================

def score_return_5d(
    value: float,
) -> float:
    """
    Avalia o retorno de cinco pregões.

    O melhor ponto é força moderada, não movimento parabólico.
    """

    if not is_valid_number(
        value
    ):
        return 45.0

    value = float(
        value
    )

    if -3 <= value <= 6:
        return 100.0

    if 6 < value <= 12:
        return 88.0

    if -7 <= value < -3:
        return 78.0

    if 12 < value <= 15:
        return 65.0

    if -12 <= value < -7:
        return 58.0

    if 15 < value <= 20:
        return 40.0

    if 20 < value <= 30:
        return 15.0

    if value > 30:
        return 0.0

    return 20.0


def score_return_10d(
    value: float,
) -> float:
    """
    Avalia o retorno de dez pregões.
    """

    if not is_valid_number(
        value
    ):
        return 45.0

    value = float(
        value
    )

    if -5 <= value <= 10:
        return 100.0

    if 10 < value <= 18:
        return 82.0

    if -10 <= value < -5:
        return 72.0

    if 18 < value <= 25:
        return 55.0

    if -18 <= value < -10:
        return 48.0

    if 25 < value <= 35:
        return 25.0

    if value > 35:
        return 0.0

    return 18.0


def score_distance_sma20(
    value: float,
) -> float:
    """
    Avalia a distância da SMA20 para timing de entrada.
    """

    if not is_valid_number(
        value
    ):
        return 45.0

    value = float(
        value
    )

    if -3 <= value <= 4:
        return 100.0

    if 4 < value <= 8:
        return 82.0

    if -7 <= value < -3:
        return 78.0

    if 8 < value <= 12:
        return 55.0

    if -12 <= value < -7:
        return 58.0

    if 12 < value <= 18:
        return 20.0

    if value > 18:
        return 0.0

    return 25.0


def score_distance_recent_high(
    value: float,
) -> float:
    """
    Avalia distância da máxima de 20 pregões.

    Pullbacks moderados dentro da tendência recebem a maior nota.
    """

    if not is_valid_number(
        value
    ):
        return 45.0

    value = float(
        value
    )

    if -8 <= value <= -3:
        return 100.0

    if -3 < value <= 0:
        return 75.0

    if -15 <= value < -8:
        return 82.0

    if -22 <= value < -15:
        return 58.0

    if value < -22:
        return 30.0

    return 35.0


def score_rsi_timing(
    value: float,
) -> float:
    """
    Avalia o RSI exclusivamente como timing de entrada.
    """

    if not is_valid_number(
        value
    ):
        return 45.0

    value = float(
        value
    )

    if 42 <= value <= 58:
        return 100.0

    if 58 < value <= 65:
        return 85.0

    if 35 <= value < 42:
        return 78.0

    if 65 < value <= 70:
        return 60.0

    if 30 <= value < 35:
        return 55.0

    if 70 < value <= 74:
        return 35.0

    if 74 < value <= 80:
        return 12.0

    if value > 80:
        return 0.0

    return 20.0


def score_macd_timing(
    histogram: float,
    variation: float,
) -> float:
    """
    Avalia confirmação e aceleração do MACD.
    """

    if not is_valid_number(
        histogram
    ):
        return 40.0

    histogram = float(
        histogram
    )

    variation = (
        float(
            variation
        )
        if is_valid_number(
            variation
        )
        else 0.0
    )

    if (
        histogram > 0
        and variation > 0
    ):
        return 100.0

    if (
        histogram <= 0
        and variation > 0
    ):
        return 72.0

    if (
        histogram > 0
        and variation <= 0
    ):
        return 62.0

    return 25.0


def score_adx_timing(
    adx: float,
    plus_di: float,
    minus_di: float,
) -> float:
    """
    Avalia a tendência sem premiar movimentos excessivamente maduros.
    """

    if not is_valid_number(
        adx
    ):
        return 45.0

    adx = float(
        adx
    )

    bullish_direction = bool(
        is_valid_number(
            plus_di
        )
        and is_valid_number(
            minus_di
        )
        and float(
            plus_di
        ) > float(
            minus_di
        )
    )

    if 18 <= adx <= 30 and bullish_direction:
        return 100.0

    if 12 <= adx < 18 and bullish_direction:
        return 80.0

    if 30 < adx <= 40 and bullish_direction:
        return 75.0

    if adx > 40 and bullish_direction:
        return 50.0

    if adx < 12:
        return 45.0

    return 25.0


def score_volume_timing(
    relative_volume_5d: float,
    relative_volume_20d: float,
    cmf: float,
) -> float:
    """
    Avalia participação de volume e fluxo.
    """

    values = []

    if is_valid_number(
        relative_volume_5d
    ):

        rv5 = float(
            relative_volume_5d
        )

        if 0.90 <= rv5 <= 1.60:
            values.append(
                100.0
            )

        elif 0.70 <= rv5 < 0.90:
            values.append(
                72.0
            )

        elif 1.60 < rv5 <= 2.50:
            values.append(
                82.0
            )

        elif rv5 > 2.50:
            values.append(
                55.0
            )

        else:
            values.append(
                35.0
            )

    if is_valid_number(
        relative_volume_20d
    ):

        rv20 = float(
            relative_volume_20d
        )

        if 0.80 <= rv20 <= 1.50:
            values.append(
                100.0
            )

        elif 0.60 <= rv20 < 0.80:
            values.append(
                65.0
            )

        elif rv20 > 1.50:
            values.append(
                78.0
            )

        else:
            values.append(
                35.0
            )

    if is_valid_number(
        cmf
    ):

        cmf = float(
            cmf
        )

        if cmf >= 0.10:
            values.append(
                100.0
            )

        elif cmf >= 0:
            values.append(
                78.0
            )

        elif cmf >= -0.10:
            values.append(
                48.0
            )

        else:
            values.append(
                20.0
            )

    if not values:
        return 40.0

    return float(
        np.mean(
            values
        )
    )


def score_risk_timing(
    atr_percentual: float,
) -> float:
    """
    Avalia o risco de volatilidade.
    """

    if not is_valid_number(
        atr_percentual
    ):
        return 40.0

    atr = float(
        atr_percentual
    )

    if atr <= 3:
        return 100.0

    if atr <= 5:
        return 90.0

    if atr <= 7:
        return 72.0

    if atr <= 10:
        return 48.0

    if atr <= 13:
        return 20.0

    return 0.0


# ============================================================
# EXTENSÃO NORMALIZADA PELA VOLATILIDADE
# ============================================================

def calculate_acceleration_factor(
    return_percent: float,
    atr_percent: float,
) -> float:
    """
    Mede quantos ATRs o preço percorreu no período.

    Exemplo:
    retorno 5d = 12%
    ATR = 4%
    fator = 3,0 ATR

    Quanto maior o fator, maior a aceleração relativa
    à volatilidade normal do ativo.
    """

    if not (
        is_valid_number(return_percent)
        and is_valid_number(atr_percent)
    ):
        return np.nan

    atr = float(atr_percent)

    if atr <= 0:
        return np.nan

    return round(
        float(return_percent) / atr,
        3,
    )


def score_acceleration_quality(
    acceleration_factor: float,
) -> float:
    """
    Avalia a qualidade da aceleração.

    A melhor região é força positiva sem explosão.
    """

    if not is_valid_number(
        acceleration_factor
    ):
        return 45.0

    value = float(
        acceleration_factor
    )

    if -0.75 <= value <= 1.25:
        return 100.0

    if 1.25 < value <= 2.00:
        return 90.0

    if 2.00 < value <= 2.50:
        return 72.0

    if 2.50 < value <= 3.00:
        return 48.0

    if 3.00 < value <= 3.50:
        return 25.0

    if value > 3.50:
        return 5.0

    if -1.50 <= value < -0.75:
        return 72.0

    if -2.50 <= value < -1.50:
        return 45.0

    return 20.0


def calculate_explosion_score(
    row: pd.Series,
) -> float:
    """
    Mede a extensão/explosão do movimento recente.

    Combina:
    - aceleração 5d normalizada pelo ATR;
    - aceleração 10d normalizada pelo ATR;
    - distância da SMA20;
    - posição perto da máxima recente;
    - RSI;
    - flags já existentes de extensão/parabolicidade.

    0–25   -> movimento saudável
    25–50  -> movimento forte
    50–72  -> movimento esticado
    72–100 -> movimento explosivo
    """

    atr = row.get(
        "atr_percentual",
        np.nan,
    )

    acc5 = calculate_acceleration_factor(
        row.get(
            "retorno_5d",
            np.nan,
        ),
        atr,
    )

    acc10 = calculate_acceleration_factor(
        row.get(
            "retorno_10d",
            np.nan,
        ),
        atr,
    )

    distance_sma20 = row.get(
        "distancia_sma_20",
        np.nan,
    )

    distance_recent_high = row.get(
        "distancia_maxima_20d",
        np.nan,
    )

    rsi = row.get(
        "rsi_14",
        np.nan,
    )

    components: list[float] = []
    weights: list[float] = []

    # aceleração 5d — principal fator
    if is_valid_number(acc5):
        value = float(acc5)

        if value <= 1.25:
            score = 10.0
        elif value <= 2.00:
            score = 25.0
        elif value <= 2.50:
            score = 45.0
        elif value <= 3.00:
            score = 65.0
        elif value <= 3.50:
            score = 82.0
        else:
            score = 100.0

        components.append(score)
        weights.append(0.30)

    # aceleração 10d
    if is_valid_number(acc10):
        value = float(acc10)

        if value <= 1.75:
            score = 10.0
        elif value <= 2.50:
            score = 25.0
        elif value <= 3.50:
            score = 45.0
        elif value <= 4.50:
            score = 68.0
        elif value <= 5.00:
            score = 85.0
        else:
            score = 100.0

        components.append(score)
        weights.append(0.20)

    # distância SMA20
    if is_valid_number(distance_sma20):
        value = float(distance_sma20)

        if value <= 4:
            score = 10.0
        elif value <= 8:
            score = 30.0
        elif value <= 12:
            score = 55.0
        elif value <= 18:
            score = 80.0
        else:
            score = 100.0

        components.append(score)
        weights.append(0.20)

    # proximidade da máxima recente
    if is_valid_number(distance_recent_high):
        value = float(distance_recent_high)

        if value <= -8:
            score = 10.0
        elif value <= -3:
            score = 25.0
        elif value <= -1:
            score = 45.0
        else:
            score = 65.0

        components.append(score)
        weights.append(0.10)

    # RSI
    if is_valid_number(rsi):
        value = float(rsi)

        if value <= 65:
            score = 10.0
        elif value <= 70:
            score = 35.0
        elif value <= 74:
            score = 60.0
        elif value <= 80:
            score = 82.0
        else:
            score = 100.0

        components.append(score)
        weights.append(0.10)

    # flags existentes
    flag_score = 0.0

    if boolean_value(
        row.get(
            "weekly_extension_risk",
            False,
        )
    ):
        flag_score += 45.0

    if boolean_value(
        row.get(
            "parabolic_move_risk",
            False,
        )
    ):
        flag_score += 55.0

    flag_score = min(
        flag_score,
        100.0,
    )

    components.append(flag_score)
    weights.append(0.10)

    if not components:
        return 40.0

    return round(
        clip_score(
            np.average(
                components,
                weights=weights,
            )
        ),
        2,
    )


def classify_extension_quality(
    explosion_score: float,
) -> str:
    """
    Classifica a extensão do movimento.
    """

    if not is_valid_number(
        explosion_score
    ):
        return "INDEFINIDA"

    score = float(
        explosion_score
    )

    if score < 25:
        return "SAUDÁVEL"

    if score < 50:
        return "FORTE"

    if score < 72:
        return "ESTICADA"

    return "EXPLOSIVA"


# ============================================================
# EXTENSÃO E PULLBACK
# ============================================================

def calculate_pullback_probability(
    row: pd.Series,
) -> float:
    """
    Estima risco heurístico de pullback.

    IMPORTANTE:
    não representa probabilidade estatística calibrada.

    A versão refinada usa principalmente extensão
    normalizada pela volatilidade, evitando tratar
    todos os ativos com os mesmos limites absolutos.
    """

    probability = 15.0

    explosion_score = row.get(
        "explosion_score",
        np.nan,
    )

    if not is_valid_number(
        explosion_score
    ):
        explosion_score = calculate_explosion_score(
            row
        )

    if is_valid_number(
        explosion_score
    ):

        explosion_score = float(
            explosion_score
        )

        # converte extensão em risco incremental
        if explosion_score >= 85:
            probability += 55

        elif explosion_score >= 72:
            probability += 40

        elif explosion_score >= 60:
            probability += 28

        elif explosion_score >= 50:
            probability += 18

        elif explosion_score >= 35:
            probability += 8

    atr = row.get(
        "atr_percentual",
        np.nan,
    )

    # volatilidade excessiva continua sendo risco adicional
    if is_valid_number(
        atr
    ):

        atr = float(
            atr
        )

        if atr >= 13:
            probability += 15

        elif atr >= 10:
            probability += 10

        elif atr >= 7:
            probability += 5

    if boolean_value(
        row.get(
            "pullback_required",
            False,
        )
    ):
        probability += 15

    return round(
        clip_score(
            probability
        ),
        2,
    )


def define_parabolic_risk(
    row: pd.Series,
) -> str:
    """
    Classifica o risco de movimento parabólico.
    """

    probability = calculate_pullback_probability(
        row
    )

    if boolean_value(
        row.get(
            "parabolic_move_risk",
            False,
        )
    ):
        return "EXTREMO"

    if probability >= 80:
        return "EXTREMO"

    if probability >= 65:
        return "ALTO"

    if probability >= 45:
        return "MÉDIO"

    return "BAIXO"


def count_timing_confirmations(
    row: pd.Series,
) -> int:
    """
    Conta confirmações independentes de timing.
    """

    confirmations = 0

    if (
        is_valid_number(
            row.get(
                "macd_hist"
            )
        )
        and is_valid_number(
            row.get(
                "macd_hist_variacao"
            )
        )
        and float(
            row.get(
                "macd_hist"
            )
        ) > 0
        and float(
            row.get(
                "macd_hist_variacao"
            )
        ) > 0
    ):
        confirmations += 1

    if (
        is_valid_number(
            row.get(
                "adx_14"
            )
        )
        and float(
            row.get(
                "adx_14"
            )
        ) >= 15
        and (
            not is_valid_number(
                row.get(
                    "plus_di_14"
                )
            )
            or not is_valid_number(
                row.get(
                    "minus_di_14"
                )
            )
            or float(
                row.get(
                    "plus_di_14"
                )
            ) > float(
                row.get(
                    "minus_di_14"
                )
            )
        )
    ):
        confirmations += 1

    if (
        is_valid_number(
            row.get(
                "volume_relativo_20d"
            )
        )
        and float(
            row.get(
                "volume_relativo_20d"
            )
        ) >= 0.80
    ):
        confirmations += 1

    return confirmations


def score_structural_trend_for_timing(
    trend_quality_score: float,
    candle_stability_score: float,
    pullback_quality_score: float,
    volatility_expansion_score: float,
) -> float:
    """
    Converte a qualidade estrutural da tendência em score de timing.

    A função usa principalmente trend_quality_score, mas confirma
    a leitura com estabilidade dos candles, qualidade do pullback
    e expansão de volatilidade.

    Objetivo:
    - premiar tendências limpas e sustentáveis;
    - não punir excessivamente tendências apenas moderadas;
    - penalizar estruturas instáveis.
    """

    values: list[float] = []
    weights: list[float] = []

    if is_valid_number(trend_quality_score):
        values.append(
            clip_score(
                float(trend_quality_score)
            )
        )
        weights.append(0.55)

    if is_valid_number(candle_stability_score):
        values.append(
            clip_score(
                float(candle_stability_score)
            )
        )
        weights.append(0.20)

    if is_valid_number(pullback_quality_score):
        values.append(
            clip_score(
                float(pullback_quality_score)
            )
        )
        weights.append(0.15)

    if is_valid_number(volatility_expansion_score):
        values.append(
            clip_score(
                100.0
                -
                float(volatility_expansion_score)
            )
        )
        weights.append(0.10)

    if not values:
        return 50.0

    return round(
        weighted_average_available(
            values=values,
            weights=weights,
            fallback=50.0,
        ),
        2,
    )


# ============================================================
# SCORE FINAL DE TIMING
# ============================================================

def calculate_entry_timing_score(
    row: pd.Series,
) -> float:
    """
    Calcula o score final de timing.

    A versão refinada reduz a duplicidade entre retorno,
    RSI e distância das médias e dá peso explícito à
    qualidade da extensão do movimento.
    """

    atr = row.get(
        "atr_percentual",
        np.nan,
    )

    acceleration_5d = row.get(
        "acceleration_factor_5d",
        np.nan,
    )

    acceleration_10d = row.get(
        "acceleration_factor_10d",
        np.nan,
    )

    explosion_score = row.get(
        "explosion_score",
        np.nan,
    )

    if not is_valid_number(
        acceleration_5d
    ):
        acceleration_5d = calculate_acceleration_factor(
            row.get(
                "retorno_5d",
                np.nan,
            ),
            atr,
        )

    if not is_valid_number(
        acceleration_10d
    ):
        acceleration_10d = calculate_acceleration_factor(
            row.get(
                "retorno_10d",
                np.nan,
            ),
            atr,
        )

    if not is_valid_number(
        explosion_score
    ):
        explosion_score = calculate_explosion_score(
            row
        )

    extension_quality_score = (
        100.0
        -
        float(explosion_score)
        if is_valid_number(
            explosion_score
        )
        else 50.0
    )

    structural_timing_score = (
        score_structural_trend_for_timing(
            trend_quality_score=row.get(
                "trend_quality_score",
                np.nan,
            ),
            candle_stability_score=row.get(
                "candle_stability_score",
                np.nan,
            ),
            pullback_quality_score=row.get(
                "pullback_quality_score",
                np.nan,
            ),
            volatility_expansion_score=row.get(
                "volatility_expansion_score",
                np.nan,
            ),
        )
    )

    base_score = weighted_average_available(
        values=[
            # extensão / aceleração
            score_acceleration_quality(
                acceleration_5d
            ),
            score_acceleration_quality(
                acceleration_10d
            ),
            extension_quality_score,

            # posição do preço
            score_distance_sma20(
                row.get(
                    "distancia_sma_20"
                )
            ),
            score_distance_recent_high(
                row.get(
                    "distancia_maxima_20d"
                )
            ),

            # momentum
            score_rsi_timing(
                row.get(
                    "rsi_14"
                )
            ),
            score_macd_timing(
                row.get(
                    "macd_hist"
                ),
                row.get(
                    "macd_hist_variacao"
                ),
            ),

            # tendência
            score_adx_timing(
                row.get(
                    "adx_14"
                ),
                row.get(
                    "plus_di_14"
                ),
                row.get(
                    "minus_di_14"
                ),
            ),

            # fluxo
            score_volume_timing(
                row.get(
                    "volume_relativo_5d"
                ),
                row.get(
                    "volume_relativo_20d"
                ),
                row.get(
                    "cmf_20"
                ),
            ),

            # qualidade estrutural da tendência
            structural_timing_score,

            # risco
            score_risk_timing(
                atr
            ),
        ],
        weights=[
            0.10,
            0.07,
            0.17,
            0.10,
            0.07,
            0.09,
            0.07,
            0.06,
            0.07,
            0.13,
            0.07,
        ],
        fallback=40.0,
    )

    penalty = 0.0

    pullback_probability = row.get(
        "pullback_probability",
        np.nan,
    )

    if not is_valid_number(
        pullback_probability
    ):
        pullback_probability = calculate_pullback_probability(
            row
        )

    if is_valid_number(
        pullback_probability
    ):

        pullback_probability = float(
            pullback_probability
        )

        if pullback_probability >= 80:
            penalty += 25

        elif pullback_probability >= 65:
            penalty += 15

        elif pullback_probability >= 50:
            penalty += 8

    # penalidade adicional apenas para explosão real
    if is_valid_number(
        explosion_score
    ):

        explosion_score = float(
            explosion_score
        )

        if explosion_score >= 85:
            penalty += 20

        elif explosion_score >= 72:
            penalty += 12

        elif explosion_score >= 60:
            penalty += 5

    if boolean_value(
        row.get(
            "parabolic_move_risk",
            False,
        )
    ):
        penalty += 15

    if boolean_value(
        row.get(
            "pullback_required",
            False,
        )
    ):
        penalty += 8

    trend_quality_score = row.get(
        "trend_quality_score",
        np.nan,
    )

    if is_valid_number(
        trend_quality_score
    ):
        trend_quality_score = float(
            trend_quality_score
        )

        if trend_quality_score < 42:
            penalty += 12

        elif trend_quality_score < 58:
            penalty += 6

        elif trend_quality_score >= 85:
            base_score += 3

        elif trend_quality_score >= 72:
            base_score += 1.5

    return round(
        clip_score(
            base_score
            -
            penalty
        ),
        2,
    )


def calculate_timing_confidence(
    row: pd.Series,
) -> float:
    """
    Mede a confiança na leitura do timing.
    """

    required_fields = [
        "retorno_5d",
        "retorno_10d",
        "distancia_sma_20",
        "distancia_maxima_20d",
        "rsi_14",
        "macd_hist",
        "macd_hist_variacao",
        "adx_14",
        "atr_percentual",
        "volume_relativo_20d",
        "trend_quality_score",
    ]

    valid_count = sum(
        1
        for field
        in required_fields
        if is_valid_number(
            row.get(
                field
            )
        )
    )

    data_completeness = (
        valid_count
        /
        len(
            required_fields
        )
        *
        100
    )

    confirmations = count_timing_confirmations(
        row
    )

    confirmation_component = (
        confirmations
        /
        3
        *
        100
    )

    confidence = (
        data_completeness * 0.70
        +
        confirmation_component * 0.30
    )

    return round(
        clip_score(
            confidence
        ),
        2,
    )


def timing_risk_is_acceptable(
    row: pd.Series,
) -> bool:
    """
    Confirma se o ativo não está excessivamente esticado.

    A decisão usa extensão normalizada pelo ATR, e não
    apenas limites absolutos de retorno.
    """

    pullback_probability = row.get(
        "pullback_probability",
        np.nan,
    )

    parabolic_risk = str(
        row.get(
            "parabolic_risk",
            "",
        )
    ).upper()

    atr = row.get(
        "atr_percentual",
        np.nan,
    )

    rsi = row.get(
        "rsi_14",
        np.nan,
    )

    distance_sma20 = row.get(
        "distancia_sma_20",
        np.nan,
    )

    acceleration_5d = row.get(
        "acceleration_factor_5d",
        np.nan,
    )

    acceleration_10d = row.get(
        "acceleration_factor_10d",
        np.nan,
    )

    explosion_score = row.get(
        "explosion_score",
        np.nan,
    )

    trend_quality_score = row.get(
        "trend_quality_score",
        np.nan,
    )

    checks = [
        (
            not is_valid_number(
                pullback_probability
            )
            or float(
                pullback_probability
            )
            <=
            MAX_ACCEPTABLE_PULLBACK_PROBABILITY
        ),

        (
            parabolic_risk
            not in {
                "ALTO",
                "EXTREMO",
            }
        ),

        (
            not is_valid_number(
                atr
            )
            or float(
                atr
            )
            <=
            MAX_ACCEPTABLE_ATR
        ),

        (
            not is_valid_number(
                rsi
            )
            or float(
                rsi
            )
            <=
            MAX_ACCEPTABLE_RSI
        ),

        (
            not is_valid_number(
                distance_sma20
            )
            or float(
                distance_sma20
            )
            <=
            MAX_ACCEPTABLE_DISTANCE_SMA20
        ),

        (
            not is_valid_number(
                acceleration_5d
            )
            or float(
                acceleration_5d
            )
            <=
            MAX_ACCEPTABLE_ACCELERATION_5D
        ),

        (
            not is_valid_number(
                acceleration_10d
            )
            or float(
                acceleration_10d
            )
            <=
            MAX_ACCEPTABLE_ACCELERATION_10D
        ),

        (
            not is_valid_number(
                explosion_score
            )
            or float(
                explosion_score
            )
            <
            MAX_ACCEPTABLE_EXPLOSION_SCORE
        ),

        (
            not is_valid_number(
                trend_quality_score
            )
            or float(
                trend_quality_score
            )
            >=
            MIN_TREND_QUALITY_FOR_ENTRY
        ),
    ]

    return bool(
        all(
            checks
        )
    )


def calculate_timing_approval(
    row: pd.Series,
) -> bool:
    """
    Aprova a entrada somente com bom timing e risco aceitável.
    """

    timing_score = row.get(
        "entry_timing_score",
        0,
    )

    confirmations = row.get(
        "timing_confirmations",
        0,
    )

    return bool(
        is_valid_number(
            timing_score
        )
        and float(
            timing_score
        )
        >=
        MIN_ENTRY_SCORE
        and int(
            confirmations
        )
        >=
        MIN_CONFIRMATIONS
        and timing_risk_is_acceptable(
            row
        )
    )


# ============================================================
# STATUS
# ============================================================

def define_timing_status(
    row: pd.Series,
) -> str:
    """
    Define o status operacional do timing.
    """

    score = float(
        row.get(
            "entry_timing_score",
            0,
        )
    )

    pullback_probability = float(
        row.get(
            "pullback_probability",
            0,
        )
    )

    parabolic_risk = str(
        row.get(
            "parabolic_risk",
            "BAIXO",
        )
    ).upper()

    timing_approved = boolean_value(
        row.get(
            "timing_approved",
            False,
        )
    )

    distance_recent_high = row.get(
        "distancia_maxima_20d",
        np.nan,
    )

    explosion_score = row.get(
        "explosion_score",
        np.nan,
    )

    extension_quality = str(
        row.get(
            "extension_quality",
            "",
        )
    ).upper()

    if timing_approved:
        return "ENTRAR AGORA"

    if (
        parabolic_risk == "EXTREMO"
        or extension_quality == "EXPLOSIVA"
        or (
            is_valid_number(
                explosion_score
            )
            and float(
                explosion_score
            ) >= 85
        )
    ):
        return "MUITO ESTICADA"

    if (
        parabolic_risk == "ALTO"
        or pullback_probability >= 65
        or extension_quality == "ESTICADA"
        or boolean_value(
            row.get(
                "pullback_required",
                False,
            )
        )
    ):
        return "AGUARDAR PULLBACK"

    if (
        score >= 68
        and is_valid_number(
            distance_recent_high
        )
        and float(
            distance_recent_high
        ) <= -3
    ):
        return "PRÉ-ENTRADA"

    if (
        score >= 60
        and is_valid_number(
            distance_recent_high
        )
        and float(
            distance_recent_high
        ) > -3
    ):
        return "AGUARDAR ROMPIMENTO"

    if (
        not timing_risk_is_acceptable(
            row
        )
    ):
        return "ALTO RISCO"

    return "LATERAL / OBSERVAÇÃO"


# ============================================================
# EXPLICAÇÕES
# ============================================================

def list_timing_positive_factors(
    row: pd.Series,
) -> str:
    """
    Lista fatores positivos do ponto de entrada.
    """

    factors: list[str] = []

    if (
        is_valid_number(
            row.get(
                "retorno_5d"
            )
        )
        and -3
        <= float(
            row.get(
                "retorno_5d"
            )
        )
        <= 12
    ):
        factors.append(
            "movimento semanal saudável"
        )

    if (
        is_valid_number(
            row.get(
                "distancia_sma_20"
            )
        )
        and -7
        <= float(
            row.get(
                "distancia_sma_20"
            )
        )
        <= 8
    ):
        factors.append(
            "preço próximo da SMA20"
        )

    if (
        is_valid_number(
            row.get(
                "rsi_14"
            )
        )
        and 42
        <= float(
            row.get(
                "rsi_14"
            )
        )
        <= 65
    ):
        factors.append(
            "RSI em faixa favorável"
        )

    if (
        is_valid_number(
            row.get(
                "macd_hist"
            )
        )
        and is_valid_number(
            row.get(
                "macd_hist_variacao"
            )
        )
        and float(
            row.get(
                "macd_hist"
            )
        ) > 0
        and float(
            row.get(
                "macd_hist_variacao"
            )
        ) > 0
    ):
        factors.append(
            "MACD acelerando"
        )

    if (
        is_valid_number(
            row.get(
                "volume_relativo_20d"
            )
        )
        and float(
            row.get(
                "volume_relativo_20d"
            )
        ) >= 0.80
    ):
        factors.append(
            "volume suficiente"
        )

    if (
        str(
            row.get(
                "parabolic_risk",
                ""
            )
        ).upper()
        ==
        "BAIXO"
    ):
        factors.append(
            "baixo risco parabólico"
        )

    trend_quality_score = row.get(
        "trend_quality_score",
        np.nan,
    )

    if (
        is_valid_number(trend_quality_score)
        and float(trend_quality_score) >= 72
    ):
        factors.append(
            "qualidade estrutural da tendência favorável"
        )

    if (
        is_valid_number(trend_quality_score)
        and float(trend_quality_score) >= 85
    ):
        factors.append(
            "tendência estrutural excelente"
        )

    return "; ".join(
        factors
    )


def list_timing_pending_conditions(
    row: pd.Series,
) -> str:
    """
    Lista o que ainda impede a entrada.
    """

    pending: list[str] = []

    if (
        is_valid_number(
            row.get(
                "retorno_5d"
            )
        )
        and float(
            row.get(
                "retorno_5d"
            )
        ) > 15
    ):
        pending.append(
            "aguardar redução da extensão semanal"
        )

    if (
        is_valid_number(
            row.get(
                "distancia_sma_20"
            )
        )
        and float(
            row.get(
                "distancia_sma_20"
            )
        ) > 8
    ):
        pending.append(
            "aguardar aproximação da SMA20"
        )

    if (
        is_valid_number(
            row.get(
                "rsi_14"
            )
        )
        and float(
            row.get(
                "rsi_14"
            )
        ) > 70
    ):
        pending.append(
            "aguardar alívio do RSI"
        )

    if (
        is_valid_number(
            row.get(
                "volume_relativo_20d"
            )
        )
        and float(
            row.get(
                "volume_relativo_20d"
            )
        ) < 0.80
    ):
        pending.append(
            "aguardar confirmação de volume"
        )

    if int(
        row.get(
            "timing_confirmations",
            0,
        )
    ) < MIN_CONFIRMATIONS:
        pending.append(
            "faltam confirmações independentes"
        )

    acceleration_5d = row.get(
        "acceleration_factor_5d",
        np.nan,
    )

    explosion_score = row.get(
        "explosion_score",
        np.nan,
    )

    if (
        is_valid_number(acceleration_5d)
        and float(acceleration_5d) > 2.50
    ):
        pending.append(
            "aguardar desaceleração do movimento recente"
        )

    if (
        is_valid_number(explosion_score)
        and float(explosion_score) >= 50
    ):
        pending.append(
            "aguardar redução da extensão do preço"
        )

    trend_quality_score = row.get(
        "trend_quality_score",
        np.nan,
    )

    if (
        is_valid_number(trend_quality_score)
        and float(trend_quality_score) < 58
    ):
        pending.append(
            "qualidade estrutural da tendência insuficiente"
        )

    return "; ".join(
        dict.fromkeys(
            pending
        )
    )


def list_timing_rejection_reasons(
    row: pd.Series,
) -> str:
    """
    Lista motivos graves para evitar entrada imediata.
    """

    reasons: list[str] = []

    if boolean_value(
        row.get(
            "parabolic_move_risk",
            False,
        )
    ):
        reasons.append(
            "movimento parabólico"
        )

    if (
        is_valid_number(
            row.get(
                "retorno_5d"
            )
        )
        and float(
            row.get(
                "retorno_5d"
            )
        ) >= 30
    ):
        reasons.append(
            "alta semanal excessiva"
        )

    if (
        is_valid_number(
            row.get(
                "retorno_10d"
            )
        )
        and float(
            row.get(
                "retorno_10d"
            )
        ) >= 35
    ):
        reasons.append(
            "alta quinzenal excessiva"
        )

    if (
        is_valid_number(
            row.get(
                "atr_percentual"
            )
        )
        and float(
            row.get(
                "atr_percentual"
            )
        ) > 10
    ):
        reasons.append(
            "volatilidade excessiva"
        )

    if (
        is_valid_number(
            row.get(
                "rsi_14"
            )
        )
        and float(
            row.get(
                "rsi_14"
            )
        ) > 80
    ):
        reasons.append(
            "RSI extremamente elevado"
        )

    acceleration_5d = row.get(
        "acceleration_factor_5d",
        np.nan,
    )

    explosion_score = row.get(
        "explosion_score",
        np.nan,
    )

    if (
        is_valid_number(acceleration_5d)
        and float(acceleration_5d) >= EXTREME_ACCELERATION_5D
    ):
        reasons.append(
            "aceleração extrema em 5 pregões"
        )

    if (
        is_valid_number(explosion_score)
        and float(explosion_score) >= 85
    ):
        reasons.append(
            "movimento explosivo"
        )

    trend_quality_score = row.get(
        "trend_quality_score",
        np.nan,
    )

    if (
        is_valid_number(trend_quality_score)
        and float(trend_quality_score) < 42
    ):
        reasons.append(
            "estrutura de tendência fraca"
        )

    return "; ".join(
        dict.fromkeys(
            reasons
        )
    )


def generate_timing_decision(
    row: pd.Series,
) -> str:
    """
    Gera decisão curta e operacional.
    """

    ticker = row.get(
        "ticker",
        ""
    )

    status = row.get(
        "timing_status",
        "LATERAL / OBSERVAÇÃO",
    )

    score = row.get(
        "entry_timing_score",
        0,
    )

    pullback = row.get(
        "pullback_probability",
        0,
    )

    if status == "ENTRAR AGORA":

        return (
            f"{ticker}: timing aprovado para entrada. "
            f"Score {float(score):.2f}; "
            f"risco de pullback {float(pullback):.2f}%."
        )

    if status == "PRÉ-ENTRADA":

        return (
            f"{ticker}: ponto próximo do ideal; "
            f"aguardar gatilho adicional."
        )

    if status == "AGUARDAR PULLBACK":

        return (
            f"{ticker}: empresa forte, mas o preço está esticado; "
            f"aguardar correção ou consolidação."
        )

    if status == "AGUARDAR ROMPIMENTO":

        return (
            f"{ticker}: aguardar rompimento confirmado "
            f"com volume."
        )

    if status == "MUITO ESTICADA":

        return (
            f"{ticker}: não perseguir o preço; "
            f"movimento recente excessivo."
        )

    if status == "ALTO RISCO":

        return (
            f"{ticker}: risco técnico elevado para entrada imediata."
        )

    return (
        f"{ticker}: acompanhar sem entrada imediata."
    )


# ============================================================
# CLASSE PRINCIPAL
# ============================================================

class EntryTimingEngine:
    """
    Motor responsável exclusivamente pelo ponto de entrada.
    """

    def __init__(
        self,
        output_file: str | Path = ENTRY_TIMING_OUTPUT_FILE,
    ) -> None:

        self.output_file = Path(
            output_file
        )

        self.result = pd.DataFrame()

    def calculate(
        self,
        indicators: pd.DataFrame,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Calcula o timing de entrada para cada ticker.

        Aceita histórico completo ou apenas a última linha
        de cada ticker.
        """

        data = validate_input(
            indicators
        )

        latest = (
            data
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
            .reset_index(
                drop=True
            )
        )

        latest[
            "acceleration_factor_5d"
        ] = latest.apply(
            lambda row:
                calculate_acceleration_factor(
                    row.get(
                        "retorno_5d",
                        np.nan,
                    ),
                    row.get(
                        "atr_percentual",
                        np.nan,
                    ),
                ),
            axis=1,
        )

        latest[
            "acceleration_factor_10d"
        ] = latest.apply(
            lambda row:
                calculate_acceleration_factor(
                    row.get(
                        "retorno_10d",
                        np.nan,
                    ),
                    row.get(
                        "atr_percentual",
                        np.nan,
                    ),
                ),
            axis=1,
        )

        latest[
            "explosion_score"
        ] = latest.apply(
            calculate_explosion_score,
            axis=1,
        )

        latest[
            "trend_structure_timing_score"
        ] = latest.apply(
            lambda row:
                score_structural_trend_for_timing(
                    trend_quality_score=row.get(
                        "trend_quality_score",
                        np.nan,
                    ),
                    candle_stability_score=row.get(
                        "candle_stability_score",
                        np.nan,
                    ),
                    pullback_quality_score=row.get(
                        "pullback_quality_score",
                        np.nan,
                    ),
                    volatility_expansion_score=row.get(
                        "volatility_expansion_score",
                        np.nan,
                    ),
                ),
            axis=1,
        )

        latest[
            "extension_quality"
        ] = latest[
            "explosion_score"
        ].apply(
            classify_extension_quality
        )

        latest[
            "pullback_probability"
        ] = latest.apply(
            calculate_pullback_probability,
            axis=1,
        )

        latest[
            "parabolic_risk"
        ] = latest.apply(
            define_parabolic_risk,
            axis=1,
        )

        latest[
            "timing_confirmations"
        ] = latest.apply(
            count_timing_confirmations,
            axis=1,
        )

        latest[
            "entry_timing_score"
        ] = latest.apply(
            calculate_entry_timing_score,
            axis=1,
        )

        latest[
            "timing_confidence"
        ] = latest.apply(
            calculate_timing_confidence,
            axis=1,
        )

        latest[
            "timing_approved"
        ] = latest.apply(
            calculate_timing_approval,
            axis=1,
        )

        latest[
            "timing_status"
        ] = latest.apply(
            define_timing_status,
            axis=1,
        )

        latest[
            "timing_positive_factors"
        ] = latest.apply(
            list_timing_positive_factors,
            axis=1,
        )

        latest[
            "timing_pending_conditions"
        ] = latest.apply(
            list_timing_pending_conditions,
            axis=1,
        )

        latest[
            "timing_rejection_reasons"
        ] = latest.apply(
            list_timing_rejection_reasons,
            axis=1,
        )

        latest[
            "timing_decision"
        ] = latest.apply(
            generate_timing_decision,
            axis=1,
        )

        status_order = {
            "ENTRAR AGORA": 1,
            "PRÉ-ENTRADA": 2,
            "AGUARDAR PULLBACK": 3,
            "AGUARDAR ROMPIMENTO": 4,
            "LATERAL / OBSERVAÇÃO": 5,
            "ALTO RISCO": 6,
            "MUITO ESTICADA": 7,
        }

        latest[
            "timing_status_order"
        ] = (
            latest[
                "timing_status"
            ]
            .map(
                status_order
            )
            .fillna(
                99
            )
        )

        self.result = (
            latest
            .sort_values(
                [
                    "timing_approved",
                    "timing_status_order",
                    "entry_timing_score",
                    "timing_confidence",
                    "pullback_probability",
                ],
                ascending=[
                    False,
                    True,
                    False,
                    False,
                    True,
                ],
            )
            .reset_index(
                drop=True
            )
        )

        self.result[
            "timing_ranking"
        ] = np.arange(
            1,
            len(
                self.result
            )
            +
            1,
        )

        if save:
            self.save()

        self.print_summary()

        return self.result.copy()

    def save(
        self,
    ) -> None:
        """
        Salva o ranking de timing.
        """

        if self.result.empty:
            raise ValueError(
                "Não há dados de timing para salvar."
            )

        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.result.to_csv(
            self.output_file,
            index=False,
            encoding="utf-8-sig",
        )

    def get_result(
        self,
    ) -> pd.DataFrame:
        """
        Retorna o resultado completo.
        """

        return self.result.copy()

    def get_enter_now(
        self,
    ) -> pd.DataFrame:
        """
        Retorna somente entradas aprovadas pelo timing.
        """

        if self.result.empty:
            return pd.DataFrame()

        return (
            self.result.loc[
                self.result[
                    "timing_approved"
                ]
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

    def get_pullback_watchlist(
        self,
    ) -> pd.DataFrame:
        """
        Retorna ações fortes que exigem pullback.
        """

        if self.result.empty:
            return pd.DataFrame()

        return (
            self.result.loc[
                self.result[
                    "timing_status"
                ].isin(
                    {
                        "AGUARDAR PULLBACK",
                        "MUITO ESTICADA",
                    }
                )
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

    def print_summary(
        self,
    ) -> None:
        """
        Exibe resumo do Entry Timing Engine.
        """

        total = len(
            self.result
        )

        approved = (
            int(
                self.result[
                    "timing_approved"
                ].sum()
            )
            if not self.result.empty
            else 0
        )

        pullback_count = (
            int(
                self.result[
                    "timing_status"
                ]
                .isin(
                    {
                        "AGUARDAR PULLBACK",
                        "MUITO ESTICADA",
                    }
                )
                .sum()
            )
            if not self.result.empty
            else 0
        )

        print()
        print("=" * 115)
        print("ENTRY TIMING ENGINE — QUALIDADE DO PONTO DE ENTRADA")
        print("=" * 115)

        print(
            f"Empresas analisadas: {total}"
        )

        print(
            f"Entrar agora: {approved}"
        )

        print(
            f"Aguardar pullback: {pullback_count}"
        )

        print(
            f"Score mínimo: {MIN_ENTRY_SCORE}"
        )

        print(
            f"Arquivo: {self.output_file}"
        )

        if not self.result.empty:

            columns = [
                "timing_ranking",
                "ticker",
                "setor",
                "close",
                "entry_timing_score",
                "timing_status",
                "timing_approved",
                "pullback_probability",
                "parabolic_risk",
                "timing_confidence",
                "retorno_5d",
                "retorno_10d",
                "acceleration_factor_5d",
                "acceleration_factor_10d",
                "explosion_score",
                "extension_quality",
                "trend_quality_score",
                "trend_structure_quality",
                "trend_structure_timing_score",
                "candle_stability_score",
                "pullback_quality_score",
                "volatility_expansion_score",
                "distancia_sma_20",
                "rsi_14",
                "atr_percentual",
                "timing_decision",
            ]

            available_columns = [
                column
                for column in columns
                if column
                in self.result.columns
            ]

            print()
            print(
                "TOP 15 — MELHORES TIMINGS"
            )

            print(
                self.result[
                    available_columns
                ]
                .head(
                    15
                )
                .to_string(
                    index=False
                )
            )

        print("=" * 115)


# ============================================================
# FUNÇÃO SIMPLIFICADA
# ============================================================

def calculate_entry_timing(
    indicators: pd.DataFrame,
    save: bool = True,
) -> pd.DataFrame:
    """
    Interface simplificada do Entry Timing Engine.
    """

    engine = EntryTimingEngine()

    return engine.calculate(
        indicators=indicators,
        save=save,
    )


# ============================================================
# EXECUÇÃO DIRETA
# ============================================================

if __name__ == "__main__":

    from config.settings import INDICATOR_FILE

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

    timing_engine = EntryTimingEngine()

    timing_engine.calculate(
        indicators=indicator_data,
        save=True,
    )
