# ============================================================
# AI INFRASTRUCTURE SCANNER
# institutional_score.py
#
# Institutional Growth & Money Flow Score
#
# Objetivo:
# Identificar ações que estejam recebendo fluxo de capital,
# apresentem força relativa, liquidez e liderança dentro
# dos setores ligados à infraestrutura de IA.
#
# O score não representa análise fundamentalista tradicional.
# Ele é orientado a swing trade de até 6 meses.
# ============================================================

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from config.settings import (
    INDICATOR_FILE,
    SHOW_PROGRESS,
    WEIGHT_GROWTH,
    WEIGHT_MARKET_LEADER,
    WEIGHT_LIQUIDITY,
    WEIGHT_HYPE,
    WEIGHT_SECTOR,
    MAX_SCORE,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURAÇÕES
# ============================================================

INSTITUTIONAL_OUTPUT_FILE = (
    Path(INDICATOR_FILE).parent
    / "institutional_score.csv"
)

MINIMUM_REQUIRED_COLUMNS = {
    "ticker",
    "date",
    "close",
    "volume",
    "setor",
    "retorno_1m",
    "retorno_3m",
    "retorno_6m",
    "volume_relativo_20d",
    "average_dollar_volume_20d",
    "mfi_14",
    "cmf_20",
    "obv",
    "obv_sma_20",
    "vwap_20",
    "sma_50",
    "sma_200",
    "distancia_maxima_52s",
}

SCORE_COMPONENTS = [
    "score_growth_momentum",
    "score_money_flow",
    "score_liquidity",
    "score_market_leadership",
    "score_sector_strength",
]


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def is_valid_number(value: object) -> bool:
    """
    Verifica se o valor é numérico e finito.
    """

    try:
        return bool(
            np.isfinite(
                float(value)
            )
        )
    except (TypeError, ValueError):
        return False


def clip_score(
    value: float,
) -> float:
    """
    Limita uma nota ao intervalo de 0 a MAX_SCORE.
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
        "sector": "setor",
        "empresa": "company",
        "volume_financeiro_medio_20d":
            "average_dollar_volume_20d",
        "retorno_1_month": "retorno_1m",
        "retorno_3_months": "retorno_3m",
        "retorno_6_months": "retorno_6m",
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
    Valida os dados recebidos do technical_indicators.py.
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

    missing = (
        MINIMUM_REQUIRED_COLUMNS
        .difference(
            df.columns
        )
    )

    if missing:
        raise KeyError(
            "Colunas obrigatórias ausentes: "
            f"{sorted(missing)}"
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

    df["setor"] = (
        df["setor"]
        .astype(str)
        .str.strip()
    )

    numeric_columns = [
        "close",
        "volume",
        "retorno_1m",
        "retorno_3m",
        "retorno_6m",
        "volume_relativo_20d",
        "average_dollar_volume_20d",
        "mfi_14",
        "cmf_20",
        "obv",
        "obv_sma_20",
        "vwap_20",
        "sma_50",
        "sma_200",
        "distancia_maxima_52s",
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
                "setor",
            ]
        )
        .sort_values(
            ["ticker", "date"]
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
            "Nenhum registro válido permaneceu "
            "após a validação."
        )

    return df


def percentile_score(
    series: pd.Series,
    higher_is_better: bool = True,
) -> pd.Series:
    """
    Converte uma métrica em score percentílico de 0 a 100.
    """

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )

    result = pd.Series(
        np.nan,
        index=series.index,
        dtype=float,
    )

    valid = values.notna()

    if valid.sum() == 0:
        return result

    if valid.sum() == 1:
        result.loc[valid] = 50.0
        return result

    ranks = values.loc[valid].rank(
        pct=True,
        method="average",
        ascending=higher_is_better,
    )

    if higher_is_better:
        result.loc[valid] = (
            ranks * 100
        )

    else:
        result.loc[valid] = (
            1
            -
            ranks
            +
            1 / valid.sum()
        ) * 100

    return result.clip(
        lower=0,
        upper=100,
    )


def weighted_average_available(
    values: list[float],
    weights: list[float],
    fallback: float = 40.0,
) -> float:
    """
    Calcula média ponderada ignorando valores ausentes.
    """

    valid_values = []
    valid_weights = []

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


# ============================================================
# SCORES DE MOMENTUM E CRESCIMENTO
# ============================================================

def score_return_1m(
    value: float,
) -> float:
    """
    Nota para retorno de 1 mês.

    Premia avanço moderado e penaliza ações muito esticadas.
    """

    if not is_valid_number(value):
        return 40.0

    value = float(value)

    if -3 <= value <= 8:
        return 85.0

    if 8 < value <= 15:
        return 100.0

    if 15 < value <= 25:
        return 85.0

    if 25 < value <= 40:
        return 60.0

    if value > 40:
        return 35.0

    if -8 <= value < -3:
        return 55.0

    if -15 <= value < -8:
        return 30.0

    return 10.0


def score_return_3m(
    value: float,
) -> float:
    """
    Nota para retorno de 3 meses.
    """

    if not is_valid_number(value):
        return 40.0

    value = float(value)

    if 10 <= value <= 35:
        return 100.0

    if 5 <= value < 10:
        return 80.0

    if 35 < value <= 60:
        return 80.0

    if 0 <= value < 5:
        return 65.0

    if 60 < value <= 100:
        return 55.0

    if -10 <= value < 0:
        return 40.0

    if value < -10:
        return 15.0

    return 30.0


def score_return_6m(
    value: float,
) -> float:
    """
    Nota para retorno de 6 meses.
    """

    if not is_valid_number(value):
        return 40.0

    value = float(value)

    if 20 <= value <= 80:
        return 100.0

    if 10 <= value < 20:
        return 80.0

    if 80 < value <= 150:
        return 75.0

    if 0 <= value < 10:
        return 60.0

    if 150 < value <= 250:
        return 50.0

    if -15 <= value < 0:
        return 35.0

    if value < -15:
        return 10.0

    return 30.0


def calculate_growth_momentum_score(
    row: pd.Series,
) -> float:
    """
    Score de crescimento de preço e momentum intermediário.
    """

    return clip_score(
        weighted_average_available(
            values=[
                score_return_1m(
                    row.get(
                        "retorno_1m"
                    )
                ),
                score_return_3m(
                    row.get(
                        "retorno_3m"
                    )
                ),
                score_return_6m(
                    row.get(
                        "retorno_6m"
                    )
                ),
            ],
            weights=[
                0.20,
                0.35,
                0.45,
            ],
            fallback=40.0,
        )
    )


# ============================================================
# MONEY FLOW SCORE
# ============================================================

def score_relative_volume(
    value: float,
) -> float:
    """
    Nota para volume relativo de 20 dias.
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
        return 50.0

    if value >= 0.6:
        return 30.0

    return 10.0


def score_mfi(
    value: float,
) -> float:
    """
    Nota para Money Flow Index.
    """

    if not is_valid_number(value):
        return 40.0

    value = float(value)

    if 50 <= value <= 70:
        return 100.0

    if 40 <= value < 50:
        return 80.0

    if 70 < value <= 80:
        return 75.0

    if 30 <= value < 40:
        return 55.0

    if 80 < value <= 90:
        return 45.0

    if value < 30:
        return 25.0

    return 15.0


def score_cmf(
    value: float,
) -> float:
    """
    Nota para Chaikin Money Flow.
    """

    if not is_valid_number(value):
        return 35.0

    value = float(value)

    if value >= 0.25:
        return 100.0

    if value >= 0.15:
        return 90.0

    if value >= 0.05:
        return 78.0

    if value >= 0:
        return 65.0

    if value >= -0.05:
        return 45.0

    if value >= -0.15:
        return 25.0

    return 8.0


def calculate_money_flow_score(
    row: pd.Series,
) -> float:
    """
    Avalia entrada de capital utilizando volume, MFI, CMF,
    OBV e posição em relação ao VWAP.
    """

    obv_above_average = (
        is_valid_number(
            row.get("obv")
        )
        and
        is_valid_number(
            row.get("obv_sma_20")
        )
        and
        float(row["obv"])
        >
        float(row["obv_sma_20"])
    )

    above_vwap = (
        is_valid_number(
            row.get("close")
        )
        and
        is_valid_number(
            row.get("vwap_20")
        )
        and
        float(row["close"])
        >
        float(row["vwap_20"])
    )

    obv_score = (
        90.0
        if obv_above_average
        else 35.0
    )

    vwap_score = (
        85.0
        if above_vwap
        else 35.0
    )

    score = weighted_average_available(
        values=[
            score_relative_volume(
                row.get(
                    "volume_relativo_20d"
                )
            ),
            score_mfi(
                row.get(
                    "mfi_14"
                )
            ),
            score_cmf(
                row.get(
                    "cmf_20"
                )
            ),
            obv_score,
            vwap_score,
        ],
        weights=[
            0.25,
            0.20,
            0.20,
            0.20,
            0.15,
        ],
        fallback=35.0,
    )

    return clip_score(
        score
    )


# ============================================================
# LIQUIDITY SCORE
# ============================================================

def score_dollar_volume(
    value: float,
) -> float:
    """
    Avalia liquidez financeira média de 20 dias.
    """

    if not is_valid_number(value):
        return 20.0

    value = float(value)

    if value >= 5_000_000_000:
        return 100.0

    if value >= 2_000_000_000:
        return 92.0

    if value >= 1_000_000_000:
        return 85.0

    if value >= 500_000_000:
        return 75.0

    if value >= 250_000_000:
        return 65.0

    if value >= 100_000_000:
        return 55.0

    if value >= 50_000_000:
        return 35.0

    return 15.0


def calculate_liquidity_score(
    row: pd.Series,
) -> float:
    """
    Score de liquidez operacional.
    """

    dollar_volume_score = score_dollar_volume(
        row.get(
            "average_dollar_volume_20d"
        )
    )

    relative_volume_score = score_relative_volume(
        row.get(
            "volume_relativo_20d"
        )
    )

    return clip_score(
        weighted_average_available(
            values=[
                dollar_volume_score,
                relative_volume_score,
            ],
            weights=[
                0.75,
                0.25,
            ],
            fallback=30.0,
        )
    )


# ============================================================
# LEADERSHIP SCORE
# ============================================================

def calculate_sector_percentiles(
    latest_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula rankings percentílicos dentro de cada setor.
    """

    df = latest_data.copy()

    df[
        "sector_return_3m_percentile"
    ] = (
        df.groupby(
            "setor"
        )["retorno_3m"]
        .transform(
            lambda series:
                percentile_score(
                    series,
                    higher_is_better=True,
                )
        )
    )

    df[
        "sector_return_6m_percentile"
    ] = (
        df.groupby(
            "setor"
        )["retorno_6m"]
        .transform(
            lambda series:
                percentile_score(
                    series,
                    higher_is_better=True,
                )
        )
    )

    df[
        "sector_volume_percentile"
    ] = (
        df.groupby(
            "setor"
        )[
            "average_dollar_volume_20d"
        ]
        .transform(
            lambda series:
                percentile_score(
                    series,
                    higher_is_better=True,
                )
        )
    )

    df[
        "sector_money_flow_percentile"
    ] = (
        df.groupby(
            "setor"
        )["score_money_flow"]
        .transform(
            lambda series:
                percentile_score(
                    series,
                    higher_is_better=True,
                )
        )
    )

    return df


def calculate_market_leadership_score(
    row: pd.Series,
) -> float:
    """
    Avalia se a ação lidera o próprio setor.
    """

    score = weighted_average_available(
        values=[
            row.get(
                "sector_return_3m_percentile"
            ),
            row.get(
                "sector_return_6m_percentile"
            ),
            row.get(
                "sector_volume_percentile"
            ),
            row.get(
                "sector_money_flow_percentile"
            ),
        ],
        weights=[
            0.25,
            0.35,
            0.20,
            0.20,
        ],
        fallback=50.0,
    )

    return clip_score(
        score
    )


# ============================================================
# SECTOR STRENGTH SCORE
# ============================================================

def calculate_sector_strength_table(
    latest_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula a força agregada de cada setor.
    """

    sector_table = (
        latest_data
        .groupby(
            "setor",
            as_index=False,
        )
        .agg(
            sector_return_1m=(
                "retorno_1m",
                "median",
            ),
            sector_return_3m=(
                "retorno_3m",
                "median",
            ),
            sector_return_6m=(
                "retorno_6m",
                "median",
            ),
            sector_money_flow=(
                "score_money_flow",
                "mean",
            ),
            sector_liquidity=(
                "score_liquidity",
                "mean",
            ),
            sector_companies=(
                "ticker",
                "nunique",
            ),
        )
    )

    sector_table[
        "score_sector_return_1m"
    ] = percentile_score(
        sector_table[
            "sector_return_1m"
        ],
        higher_is_better=True,
    )

    sector_table[
        "score_sector_return_3m"
    ] = percentile_score(
        sector_table[
            "sector_return_3m"
        ],
        higher_is_better=True,
    )

    sector_table[
        "score_sector_return_6m"
    ] = percentile_score(
        sector_table[
            "sector_return_6m"
        ],
        higher_is_better=True,
    )

    sector_table[
        "score_sector_money_flow"
    ] = percentile_score(
        sector_table[
            "sector_money_flow"
        ],
        higher_is_better=True,
    )

    sector_table[
        "score_sector_liquidity"
    ] = percentile_score(
        sector_table[
            "sector_liquidity"
        ],
        higher_is_better=True,
    )

    sector_table[
        "score_sector_strength"
    ] = (
        sector_table[
            "score_sector_return_1m"
        ] * 0.10
        +
        sector_table[
            "score_sector_return_3m"
        ] * 0.25
        +
        sector_table[
            "score_sector_return_6m"
        ] * 0.30
        +
        sector_table[
            "score_sector_money_flow"
        ] * 0.25
        +
        sector_table[
            "score_sector_liquidity"
        ] * 0.10
    ).clip(
        lower=0,
        upper=100,
    )

    return sector_table


# ============================================================
# HYPE / ATTENTION SCORE
# ============================================================

def score_distance_from_high(
    value: float,
) -> float:
    """
    Premia ações próximas da máxima, mas evita ações esticadas.
    """

    if not is_valid_number(value):
        return 40.0

    value = float(value)

    if -20 <= value <= -8:
        return 100.0

    if -8 < value <= -3:
        return 85.0

    if -30 <= value < -20:
        return 80.0

    if -3 < value <= 0:
        return 65.0

    if -40 <= value < -30:
        return 55.0

    if value < -40:
        return 25.0

    return 40.0


def calculate_hype_score(
    row: pd.Series,
) -> float:
    """
    Score de atenção de mercado.

    Utiliza volume, momentum e proximidade da máxima.
    """

    return clip_score(
        weighted_average_available(
            values=[
                score_relative_volume(
                    row.get(
                        "volume_relativo_20d"
                    )
                ),
                score_return_3m(
                    row.get(
                        "retorno_3m"
                    )
                ),
                score_return_6m(
                    row.get(
                        "retorno_6m"
                    )
                ),
                score_distance_from_high(
                    row.get(
                        "distancia_maxima_52s"
                    )
                ),
            ],
            weights=[
                0.30,
                0.25,
                0.25,
                0.20,
            ],
            fallback=40.0,
        )
    )


# ============================================================
# PENALIDADES
# ============================================================

def calculate_penalties(
    row: pd.Series,
) -> tuple[float, str]:
    """
    Penalidades por deterioração de fluxo ou tendência.
    """

    penalty = 0.0
    reasons: list[str] = []

    cmf = row.get(
        "cmf_20"
    )

    mfi = row.get(
        "mfi_14"
    )

    volume_relative = row.get(
        "volume_relativo_20d"
    )

    return_3m = row.get(
        "retorno_3m"
    )

    return_6m = row.get(
        "retorno_6m"
    )

    close = row.get(
        "close"
    )

    sma_50 = row.get(
        "sma_50"
    )

    sma_200 = row.get(
        "sma_200"
    )

    if (
        is_valid_number(cmf)
        and float(cmf) < -0.15
    ):
        penalty += 12
        reasons.append(
            "CMF fortemente negativo"
        )

    if (
        is_valid_number(mfi)
        and float(mfi) < 25
    ):
        penalty += 8
        reasons.append(
            "MFI muito fraco"
        )

    if (
        is_valid_number(volume_relative)
        and float(volume_relative) < 0.60
    ):
        penalty += 6
        reasons.append(
            "volume relativo muito baixo"
        )

    if (
        is_valid_number(return_3m)
        and float(return_3m) < -15
    ):
        penalty += 8
        reasons.append(
            "retorno de 3 meses negativo"
        )

    if (
        is_valid_number(return_6m)
        and float(return_6m) < -25
    ):
        penalty += 8
        reasons.append(
            "retorno de 6 meses negativo"
        )

    if (
        is_valid_number(close)
        and is_valid_number(sma_50)
        and float(close) < float(sma_50)
    ):
        penalty += 4
        reasons.append(
            "preço abaixo da SMA50"
        )

    if (
        is_valid_number(close)
        and is_valid_number(sma_200)
        and float(close) < float(sma_200)
    ):
        penalty += 8
        reasons.append(
            "preço abaixo da SMA200"
        )

    return (
        float(penalty),
        "; ".join(reasons),
    )


# ============================================================
# CLASSIFICAÇÃO
# ============================================================

def classify_institutional_score(
    score: float,
) -> str:
    """
    Classifica o Institutional Score.
    """

    if score >= 85:
        return "FLUXO INSTITUCIONAL EXCEPCIONAL"

    if score >= 75:
        return "FLUXO INSTITUCIONAL FORTE"

    if score >= 65:
        return "FLUXO INSTITUCIONAL FAVORÁVEL"

    if score >= 55:
        return "FLUXO MODERADO"

    if score >= 45:
        return "FLUXO NEUTRO"

    return "FLUXO FRACO"


def institutional_diagnosis(
    row: pd.Series,
) -> str:
    """
    Diagnóstico executivo da ação.
    """

    score = row.get(
        "institutional_score",
        0
    )

    money_flow = row.get(
        "score_money_flow",
        0
    )

    leadership = row.get(
        "score_market_leadership",
        0
    )

    sector_strength = row.get(
        "score_sector_strength",
        0
    )

    growth = row.get(
        "score_growth_momentum",
        0
    )

    if (
        score >= 80
        and money_flow >= 75
        and leadership >= 70
    ):
        return (
            "FORTE ENTRADA DE CAPITAL E "
            "LIDERANÇA SETORIAL"
        )

    if (
        money_flow >= 80
        and growth < 55
    ):
        return (
            "ACUMULAÇÃO INSTITUCIONAL "
            "ANTES DA ACELERAÇÃO"
        )

    if (
        growth >= 80
        and money_flow < 50
    ):
        return (
            "PREÇO FORTE, MAS FLUXO "
            "AINDA NÃO CONFIRMADO"
        )

    if (
        sector_strength >= 75
        and leadership >= 65
    ):
        return (
            "LÍDER EM SETOR FORTE"
        )

    if score >= 65:
        return (
            "FLUXO E MOMENTUM FAVORÁVEIS"
        )

    if score >= 55:
        return (
            "ACOMPANHAMENTO QUALIFICADO"
        )

    return (
        "SEM EVIDÊNCIA SUFICIENTE "
        "DE ENTRADA INSTITUCIONAL"
    )


# ============================================================
# CLASSE PRINCIPAL
# ============================================================

class InstitutionalScore:
    """
    Motor responsável pelo Institutional Growth Score.
    """

    def __init__(
        self,
        output_file: str | Path = (
            INSTITUTIONAL_OUTPUT_FILE
        ),
    ) -> None:
        self.output_file = Path(
            output_file
        )

        self.history = pd.DataFrame()
        self.latest = pd.DataFrame()
        self.sector_table = pd.DataFrame()
        self.failures = pd.DataFrame()

    def calculate(
        self,
        indicators: pd.DataFrame,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Calcula o Institutional Score para o registro mais
        recente de cada ticker.

        Parameters
        ----------
        indicators:
            DataFrame produzido por technical_indicators.py.

        save:
            Salva o resultado em CSV quando True.

        Returns
        -------
        pandas.DataFrame
            Ranking institucional por ticker.
        """

        validated = validate_input(
            indicators
        )

        self.history = validated.copy()

        latest = (
            validated
            .sort_values(
                ["ticker", "date"]
            )
            .groupby(
                "ticker",
                as_index=False,
            )
            .tail(1)
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # SCORES INDIVIDUAIS
        # ----------------------------------------------------

        latest[
            "score_growth_momentum"
        ] = latest.apply(
            calculate_growth_momentum_score,
            axis=1,
        )

        latest[
            "score_money_flow"
        ] = latest.apply(
            calculate_money_flow_score,
            axis=1,
        )

        latest[
            "score_liquidity"
        ] = latest.apply(
            calculate_liquidity_score,
            axis=1,
        )

        latest[
            "score_hype"
        ] = latest.apply(
            calculate_hype_score,
            axis=1,
        )

        # ----------------------------------------------------
        # LIDERANÇA NO SETOR
        # ----------------------------------------------------

        latest = calculate_sector_percentiles(
            latest
        )

        latest[
            "score_market_leadership"
        ] = latest.apply(
            calculate_market_leadership_score,
            axis=1,
        )

        # ----------------------------------------------------
        # FORÇA DO SETOR
        # ----------------------------------------------------

        self.sector_table = (
            calculate_sector_strength_table(
                latest
            )
        )

        latest = latest.merge(
            self.sector_table[
                [
                    "setor",
                    "score_sector_strength",
                    "sector_return_1m",
                    "sector_return_3m",
                    "sector_return_6m",
                    "sector_money_flow",
                ]
            ],
            on="setor",
            how="left",
            validate="many_to_one",
        )

        # ----------------------------------------------------
        # SCORE BRUTO
        # ----------------------------------------------------

        total_weight = (
            WEIGHT_GROWTH
            +
            WEIGHT_MARKET_LEADER
            +
            WEIGHT_LIQUIDITY
            +
            WEIGHT_HYPE
            +
            WEIGHT_SECTOR
        )

        if total_weight <= 0:
            raise ValueError(
                "A soma dos pesos institucionais "
                "deve ser maior que zero."
            )

        latest[
            "institutional_score_raw"
        ] = (
            latest[
                "score_growth_momentum"
            ] * WEIGHT_GROWTH
            +
            latest[
                "score_market_leadership"
            ] * WEIGHT_MARKET_LEADER
            +
            latest[
                "score_liquidity"
            ] * WEIGHT_LIQUIDITY
            +
            latest[
                "score_hype"
            ] * WEIGHT_HYPE
            +
            latest[
                "score_sector_strength"
            ] * WEIGHT_SECTOR
        ) / total_weight

        # ----------------------------------------------------
        # PENALIDADES
        # ----------------------------------------------------

        penalties = latest.apply(
            calculate_penalties,
            axis=1,
        )

        latest[
            "institutional_penalty"
        ] = [
            result[0]
            for result in penalties
        ]

        latest[
            "institutional_penalty_reasons"
        ] = [
            result[1]
            for result in penalties
        ]

        # ----------------------------------------------------
        # SCORE FINAL
        # ----------------------------------------------------

        latest[
            "institutional_score"
        ] = (
            latest[
                "institutional_score_raw"
            ]
            -
            latest[
                "institutional_penalty"
            ]
        ).clip(
            lower=0,
            upper=100,
        ).round(2)

        latest[
            "institutional_classification"
        ] = latest[
            "institutional_score"
        ].apply(
            classify_institutional_score
        )

        latest[
            "institutional_diagnosis"
        ] = latest.apply(
            institutional_diagnosis,
            axis=1,
        )

        latest[
            "institutional_flow_approved"
        ] = (
            latest[
                "institutional_score"
            ] >= 65
        )

        latest = (
            latest
            .sort_values(
                [
                    "institutional_score",
                    "score_money_flow",
                    "score_market_leadership",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
            )
            .reset_index(drop=True)
        )

        latest[
            "institutional_ranking"
        ] = np.arange(
            1,
            len(latest) + 1,
        )

        self.latest = latest

        if save:
            self.save()

        self.print_summary()

        return self.latest.copy()

    def save(self) -> None:
        """
        Salva o ranking institucional.
        """

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
        Retorna o ranking institucional.
        """

        return self.latest.copy()

    def get_sector_table(self) -> pd.DataFrame:
        """
        Retorna o ranking de força por setor.
        """

        return self.sector_table.copy()

    def print_summary(self) -> None:
        """
        Exibe resumo da execução.
        """

        approved = (
            int(
                self.latest[
                    "institutional_flow_approved"
                ].sum()
            )
            if not self.latest.empty
            else 0
        )

        total = len(
            self.latest
        )

        print()
        print("=" * 95)
        print("INSTITUTIONAL GROWTH SCORE")
        print("=" * 95)
        print(
            f"Empresas analisadas: {total}"
        )
        print(
            f"Fluxo institucional aprovado: {approved}"
        )
        print(
            f"Arquivo: {self.output_file}"
        )

        if not self.latest.empty:
            print()
            print(
                "TOP 10 — FLUXO INSTITUCIONAL"
            )

            columns = [
                "institutional_ranking",
                "ticker",
                "setor",
                "institutional_score",
                "institutional_classification",
                "score_money_flow",
                "score_market_leadership",
                "score_sector_strength",
                "institutional_diagnosis",
            ]

            available = [
                column
                for column in columns
                if column in self.latest.columns
            ]

            print(
                self.latest[
                    available
                ]
                .head(10)
                .to_string(
                    index=False
                )
            )

        print("=" * 95)


# ============================================================
# FUNÇÃO SIMPLIFICADA
# ============================================================

def calculate_institutional_score(
    indicators: pd.DataFrame,
    save: bool = True,
) -> pd.DataFrame:
    """
    Interface simplificada do Institutional Score.
    """

    engine = InstitutionalScore()

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

    engine = InstitutionalScore()

    institutional_ranking = (
        engine.calculate(
            indicators=indicator_data,
            save=True,
        )
    )

    print()
    print(
        institutional_ranking[
            [
                "institutional_ranking",
                "ticker",
                "setor",
                "institutional_score",
                "institutional_classification",
                "institutional_diagnosis",
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )
