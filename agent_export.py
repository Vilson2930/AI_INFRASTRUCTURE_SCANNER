# ============================================================
# AI INFRASTRUCTURE SCANNER
# agent_export.py
# ============================================================
#
# Exportador de dados para o INVESTMENT CIO AGENT.
#
# OBJETIVO:
# Transformar os resultados já calculados pelo scanner em um
# JSON bruto, estável e rastreável para consumo pelo CIO.
#
# IMPORTANTE:
# - NÃO recalcula indicadores;
# - NÃO recalcula scores;
# - NÃO altera sinais;
# - NÃO altera ranking;
# - NÃO cria recomendação;
# - NÃO executa ordens;
# - apenas serializa resultados produzidos pelo motor.
#
# ============================================================

from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# IDENTIDADE DA EXPORTAÇÃO
# ============================================================

SOURCE_SYSTEM = "AI_INFRASTRUCTURE_SCANNER"
EXPORT_VERSION = "1.0"

OUTPUT_DIR = Path("outputs")
OUTPUT_FILE = OUTPUT_DIR / "agent_output_raw.json"


# ============================================================
# CAMPOS PRIORITÁRIOS
# ============================================================
#
# Estes campos NÃO são usados para recalcular nada.
#
# Servem apenas para organizar o conteúdo mais importante
# produzido pelo ranking executivo.
#
# Caso existam outras colunas no DataFrame, elas também serão
# preservadas.
# ============================================================

PRIORITY_RANKING_FIELDS = [
    "ranking",
    "ticker",
    "company",
    "setor",
    "signal_status",
    "ranking_quality",
    "priority_score",
    "final_score",
    "institutional_score",
    "technical_entry_score",
    "entry_timing_score",
    "timing_status",
    "timing_approved",
    "timing_confidence",
    "pullback_probability",
    "parabolic_risk",
    "executive_decision",
]


# ============================================================
# SERIALIZAÇÃO SEGURA
# ============================================================

def _is_null(value: Any) -> bool:
    """
    Detecta valores nulos sem gerar ambiguidades com listas,
    dicionários ou DataFrames.
    """

    if value is None:
        return True

    if value is pd.NA:
        return True

    if isinstance(value, (float, np.floating)):
        try:
            return math.isnan(float(value))
        except (TypeError, ValueError):
            return False

    return False


def _json_safe(value: Any) -> Any:
    """
    Converte objetos comuns do pandas/numpy/Python para tipos
    serializáveis em JSON.

    Não altera semanticamente scores, sinais ou decisões.
    """

    if _is_null(value):
        return None

    if isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        numeric_value = float(value)

        if math.isfinite(numeric_value):
            return numeric_value

        return None

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(
        value,
        (
            datetime,
            date,
            pd.Timestamp,
        ),
    ):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            _json_safe(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return [
            _json_safe(item)
            for item in value.tolist()
        ]

    if hasattr(value, "item"):
        try:
            return _json_safe(
                value.item()
            )
        except Exception:
            pass

    return str(value)


# ============================================================
# DATAFRAME -> REGISTROS
# ============================================================

def _dataframe_to_records(
    dataframe: Any,
) -> list[dict[str, Any]]:
    """
    Converte DataFrame para lista de registros JSON.

    Todas as colunas existentes são preservadas.
    """

    if dataframe is None:
        return []

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise TypeError(
            "Era esperado um pandas DataFrame, "
            f"mas foi recebido {type(dataframe).__name__}."
        )

    if dataframe.empty:
        return []

    clean_dataframe = (
        dataframe
        .copy()
        .reset_index(drop=True)
    )

    records = clean_dataframe.to_dict(
        orient="records"
    )

    return [
        {
            str(key): _json_safe(value)
            for key, value in record.items()
        }
        for record in records
    ]


# ============================================================
# COLUNAS
# ============================================================

def _get_columns(
    dataframe: Any,
) -> list[str]:
    """
    Retorna as colunas existentes no DataFrame.
    """

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        return []

    return [
        str(column)
        for column in dataframe.columns
    ]


# ============================================================
# CAMPOS PRIORITÁRIOS PRESENTES
# ============================================================

def _get_available_priority_fields(
    dataframe: Any,
) -> list[str]:
    """
    Informa quais campos prioritários realmente existem no
    ranking produzido pelo motor.

    Não cria campos ausentes.
    """

    columns = set(
        _get_columns(dataframe)
    )

    return [
        field
        for field in PRIORITY_RANKING_FIELDS
        if field in columns
    ]


# ============================================================
# CONTAGEM DE SINAIS
# ============================================================

def _count_existing_values(
    dataframe: Any,
    column: str,
) -> dict[str, int]:
    """
    Conta valores já produzidos pelo motor.

    É apenas uma agregação descritiva.
    Não cria ou modifica sinais.
    """

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        return {}

    if dataframe.empty:
        return {}

    if column not in dataframe.columns:
        return {}

    series = dataframe[column]

    counts: dict[str, int] = {}

    for value in series:

        safe_value = _json_safe(value)

        if safe_value is None:
            continue

        key = str(safe_value)

        counts[key] = (
            counts.get(key, 0)
            + 1
        )

    return counts


# ============================================================
# TICKERS
# ============================================================

def _extract_tickers(
    dataframe: Any,
) -> list[str]:
    """
    Extrai tickers existentes preservando a ordem.
    """

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        return []

    if dataframe.empty:
        return []

    if "ticker" not in dataframe.columns:
        return []

    tickers: list[str] = []

    for value in dataframe["ticker"]:

        safe_value = _json_safe(value)

        if safe_value is None:
            continue

        ticker = str(
            safe_value
        ).strip()

        if (
            ticker
            and ticker not in tickers
        ):
            tickers.append(
                ticker
            )

    return tickers


# ============================================================
# VALIDAÇÃO DO RESULTADO RECEBIDO
# ============================================================

def _validate_results(
    results: dict[str, object],
) -> None:
    """
    Valida apenas a estrutura necessária para exportação.

    Não valida a lógica quantitativa do scanner.
    """

    if not isinstance(
        results,
        dict,
    ):
        raise TypeError(
            "results deve ser um dicionário retornado "
            "por run_scanner()."
        )

    required_keys = [
        "ranking",
        "executive_ranking",
        "approved_entries",
        "watchlist",
        "best_by_sector",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in results
    ]

    if missing_keys:
        raise ValueError(
            "Resultados incompletos para exportação. "
            "Chaves ausentes: "
            + ", ".join(missing_keys)
        )

    dataframe_keys = [
        "ranking",
        "executive_ranking",
        "approved_entries",
        "watchlist",
        "best_by_sector",
    ]

    for key in dataframe_keys:

        value = results.get(key)

        if not isinstance(
            value,
            pd.DataFrame,
        ):
            raise TypeError(
                f"results['{key}'] deve ser "
                "um pandas DataFrame."
            )


# ============================================================
# METADADOS DO MOTOR
# ============================================================

def _build_engine_metadata(
    results: dict[str, object],
) -> dict[str, Any]:

    ranking = results.get(
        "ranking"
    )

    executive_ranking = results.get(
        "executive_ranking"
    )

    approved_entries = results.get(
        "approved_entries"
    )

    watchlist = results.get(
        "watchlist"
    )

    best_by_sector = results.get(
        "best_by_sector"
    )

    return {
        "objective": (
            "Identificar ações de infraestrutura de IA "
            "com entrada de capital e boa oportunidade "
            "técnica para swing trade de até 6 meses."
        ),

        "pipeline": [
            "MARKET_DATA",
            "TECHNICAL_INDICATORS",
            "INSTITUTIONAL_SCORE",
            "TECHNICAL_ENTRY_SCORE",
            "ENTRY_TIMING_ENGINE",
            "SIGNAL_ENGINE",
            "RANKING_ENGINE",
        ],

        "ranking_rows":
            len(ranking),

        "executive_ranking_rows":
            len(executive_ranking),

        "approved_entries_rows":
            len(approved_entries),

        "watchlist_rows":
            len(watchlist),

        "best_by_sector_rows":
            len(best_by_sector),

        "ranking_columns":
            _get_columns(
                ranking
            ),

        "executive_ranking_columns":
            _get_columns(
                executive_ranking
            ),

        "available_priority_fields":
            _get_available_priority_fields(
                executive_ranking
            ),
    }


# ============================================================
# RESUMO DESCRITIVO
# ============================================================

def _build_summary(
    results: dict[str, object],
) -> dict[str, Any]:
    """
    Gera somente contagens e listas descritivas baseadas nos
    resultados já calculados.
    """

    executive_ranking = results[
        "executive_ranking"
    ]

    approved_entries = results[
        "approved_entries"
    ]

    watchlist = results[
        "watchlist"
    ]

    best_by_sector = results[
        "best_by_sector"
    ]

    return {
        "classified_companies":
            len(executive_ranking),

        "approved_entries":
            len(approved_entries),

        "watchlist_size":
            len(watchlist),

        "represented_sectors":
            len(best_by_sector),

        "signal_counts":
            _count_existing_values(
                executive_ranking,
                "signal_status",
            ),

        "ranking_quality_counts":
            _count_existing_values(
                executive_ranking,
                "ranking_quality",
            ),

        "timing_status_counts":
            _count_existing_values(
                executive_ranking,
                "timing_status",
            ),

        "executive_decision_counts":
            _count_existing_values(
                executive_ranking,
                "executive_decision",
            ),

        "approved_tickers":
            _extract_tickers(
                approved_entries
            ),

        "watchlist_tickers":
            _extract_tickers(
                watchlist
            ),
    }


# ============================================================
# AUDITORIA DE PRESERVAÇÃO
# ============================================================

def _build_export_policy() -> dict[str, bool]:
    """
    Declara explicitamente o comportamento deste exportador.
    """

    return {
        "indicators_recalculated":
            False,

        "institutional_score_recalculated":
            False,

        "technical_score_recalculated":
            False,

        "entry_timing_recalculated":
            False,

        "signals_recalculated":
            False,

        "ranking_recalculated":
            False,

        "executive_decisions_modified":
            False,

        "broker_execution_allowed":
            False,

        "source_results_preserved":
            True,
    }


# ============================================================
# PAYLOAD
# ============================================================

def build_agent_payload(
    results: dict[str, object],
) -> dict[str, Any]:
    """
    Constrói o payload bruto destinado ao Investment CIO.
    """

    _validate_results(
        results
    )

    ranking = results[
        "ranking"
    ]

    executive_ranking = results[
        "executive_ranking"
    ]

    approved_entries = results[
        "approved_entries"
    ]

    watchlist = results[
        "watchlist"
    ]

    best_by_sector = results[
        "best_by_sector"
    ]

    payload = {
        "source_system":
            SOURCE_SYSTEM,

        "export_version":
            EXPORT_VERSION,

        "generated_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "summary":
            _build_summary(
                results
            ),

        "engine":
            _build_engine_metadata(
                results
            ),

        # ----------------------------------------------------
        # RESULTADOS PRINCIPAIS
        # ----------------------------------------------------

        "ranking":
            _dataframe_to_records(
                ranking
            ),

        "executive_ranking":
            _dataframe_to_records(
                executive_ranking
            ),

        "approved_entries":
            _dataframe_to_records(
                approved_entries
            ),

        "watchlist":
            _dataframe_to_records(
                watchlist
            ),

        "best_by_sector":
            _dataframe_to_records(
                best_by_sector
            ),

        # ----------------------------------------------------
        # CAMADAS INTERMEDIÁRIAS
        #
        # Preservadas para rastreabilidade do CIO.
        # ----------------------------------------------------

        "institutional_ranking":
            _dataframe_to_records(
                results.get(
                    "institutional_ranking"
                )
            ),

        "sector_strength":
            _dataframe_to_records(
                results.get(
                    "sector_strength"
                )
            ),

        "technical_ranking":
            _dataframe_to_records(
                results.get(
                    "technical_ranking"
                )
            ),

        "timing_ranking":
            _dataframe_to_records(
                results.get(
                    "timing_ranking"
                )
            ),

        "signals":
            _dataframe_to_records(
                results.get(
                    "signals"
                )
            ),

        # ----------------------------------------------------
        # METADADOS
        # ----------------------------------------------------

        "metadata": {
            "source_report":
                _json_safe(
                    results.get(
                        "report_path"
                    )
                ),

            "export_policy":
                _build_export_policy(),
        },
    }

    return payload


# ============================================================
# EXPORTAÇÃO
# ============================================================

def export_agent_output(
    results: dict[str, object],
    output_file: str | Path = OUTPUT_FILE,
) -> str:
    """
    Gera outputs/agent_output_raw.json.
    """

    payload = build_agent_payload(
        results
    )

    output_path = Path(
        output_file
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    print()
    print("=" * 120)
    print(
        "EXPORTAÇÃO PARA INVESTMENT CIO AGENT"
    )
    print("=" * 120)

    print(
        f"Sistema: {SOURCE_SYSTEM}"
    )

    print(
        f"Versão da exportação: {EXPORT_VERSION}"
    )

    print(
        "Empresas classificadas: "
        f"{payload['summary']['classified_companies']}"
    )

    print(
        "Entradas aprovadas: "
        f"{payload['summary']['approved_entries']}"
    )

    print(
        "Watchlist: "
        f"{payload['summary']['watchlist_size']}"
    )

    print(
        "Setores representados: "
        f"{payload['summary']['represented_sectors']}"
    )

    print(
        "Sinais: "
        f"{payload['summary']['signal_counts']}"
    )

    print(
        f"Arquivo: {output_path}"
    )

    print("=" * 120)

    return str(
        output_path
    )
