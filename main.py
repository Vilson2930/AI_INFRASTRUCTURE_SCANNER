# ============================================================
# AI INFRASTRUCTURE SCANNER
# main.py
#
# Motor principal do projeto.
#
# Fluxo:
# 1. Coleta de preços
# 2. Indicadores técnicos
# 3. Institutional Score
# 4. Technical Entry Score
# 5. Signal Engine
# 6. Ranking final
# 7. Relatório Excel
# ============================================================

from __future__ import annotations

import sys
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

from config.settings import (
    PROJECT_NAME,
    VERSION,
    DATA_PATH,
    PRICE_FILE,
    INDICATOR_FILE,
    RANKING_FILE,
    REPORT_FILE,
)

from engine.market_data import MarketData
from engine.technical_indicators import TechnicalIndicators
from engine.institutional_score import InstitutionalScore
from engine.technical_score import TechnicalScore
from engine.signal_engine import SignalEngine
from engine.ranking_engine import RankingEngine
from engine.report import ReportEngine


# ============================================================
# CONFIGURAÇÕES DA EXECUÇÃO
# ============================================================

USE_EXISTING_PRICES = False
USE_EXISTING_INDICATORS = False

SAVE_INTERMEDIATE_FILES = True
GENERATE_EXCEL_REPORT = True

TOP_N_DISPLAY = 20


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def print_header() -> None:
    """
    Exibe o cabeçalho do scanner.
    """

    print("=" * 120)
    print(PROJECT_NAME.upper())
    print(f"VERSÃO {VERSION}")
    print("=" * 120)
    print(
        "Objetivo: identificar ações de infraestrutura de IA "
        "com entrada de capital e boa oportunidade técnica "
        "para swing trade de até 6 meses."
    )
    print("=" * 120)
    print(
        f"Início da execução: "
        f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
    )
    print("=" * 120)


def print_stage(
    stage_number: int,
    stage_name: str,
) -> None:
    """
    Exibe o início de cada etapa.
    """

    print()
    print("=" * 120)
    print(
        f"ETAPA {stage_number} — {stage_name}"
    )
    print("=" * 120)


def validate_dataframe(
    data: pd.DataFrame,
    dataframe_name: str,
) -> None:
    """
    Confirma se o resultado de uma etapa é válido.
    """

    if not isinstance(
        data,
        pd.DataFrame,
    ):
        raise TypeError(
            f"{dataframe_name} não é um pandas DataFrame."
        )

    if data.empty:
        raise ValueError(
            f"{dataframe_name} está vazio."
        )


def load_csv(
    file_path: str | Path,
    file_description: str,
) -> pd.DataFrame:
    """
    Carrega um CSV existente com validação.
    """

    path = Path(
        file_path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"{file_description} não encontrado: {path}"
        )

    data = pd.read_csv(
        path
    )

    validate_dataframe(
        data=data,
        dataframe_name=file_description,
    )

    print(
        f"✅ {file_description} carregado: "
        f"{len(data):,} linhas"
    )

    return data


def show_top_ranking(
    ranking: pd.DataFrame,
    top_n: int = TOP_N_DISPLAY,
) -> None:
    """
    Exibe as melhores oportunidades.
    """

    if ranking.empty:
        print(
            "Nenhuma empresa disponível no ranking."
        )
        return

    desired_columns = [
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
        "executive_decision",
    ]

    available_columns = [
        column
        for column in desired_columns
        if column in ranking.columns
    ]

    print()
    print(
        f"TOP {min(top_n, len(ranking))} "
        "— MELHORES OPORTUNIDADES"
    )
    print("-" * 120)

    print(
        ranking[
            available_columns
        ]
        .head(top_n)
        .to_string(
            index=False
        )
    )


def print_final_summary(
    ranking_engine: RankingEngine,
    report_path: Path | None,
    execution_start: datetime,
) -> None:
    """
    Exibe o resumo final da execução.
    """

    full_ranking = (
        ranking_engine
        .get_full_ranking()
    )

    approved_entries = (
        ranking_engine
        .get_approved_entries()
    )

    watchlist = (
        ranking_engine
        .get_watchlist()
    )

    best_by_sector = (
        ranking_engine
        .get_best_by_sector()
    )

    execution_end = datetime.now()

    duration = (
        execution_end
        -
        execution_start
    ).total_seconds()

    print()
    print("=" * 120)
    print("RESUMO FINAL")
    print("=" * 120)

    print(
        f"Empresas classificadas: "
        f"{len(full_ranking)}"
    )

    print(
        f"Entradas aprovadas: "
        f"{len(approved_entries)}"
    )

    print(
        f"Ações na watchlist: "
        f"{len(watchlist)}"
    )

    print(
        f"Setores representados: "
        f"{len(best_by_sector)}"
    )

    print(
        f"Tempo total: "
        f"{duration:.2f} segundos"
    )

    print()
    print("Arquivos principais:")

    print(
        f"- Preços: {PRICE_FILE}"
    )

    print(
        f"- Indicadores: {INDICATOR_FILE}"
    )

    print(
        f"- Ranking: {RANKING_FILE}"
    )

    if report_path is not None:

        print(
            f"- Relatório: {report_path}"
        )

    print("=" * 120)

    if approved_entries.empty:

        print(
            "⚠️ Nenhuma entrada foi aprovada neste pregão."
        )

        print(
            "Isso não representa falha do sistema. "
            "Significa que nenhuma ação reuniu simultaneamente "
            "fluxo institucional e confirmação técnica."
        )

    else:

        print(
            "✅ Existem entradas aprovadas pelo scanner."
        )

    print("=" * 120)


# ============================================================
# MOTOR PRINCIPAL
# ============================================================

def run_scanner() -> dict[str, object]:
    """
    Executa todo o pipeline do scanner.

    Returns
    -------
    dict
        Resultados principais de cada etapa.
    """

    execution_start = datetime.now()

    print_header()

    DATA_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. PREÇOS
    # --------------------------------------------------------

    print_stage(
        1,
        "COLETA E VALIDAÇÃO DOS PREÇOS",
    )

    if USE_EXISTING_PRICES:

        price_history = load_csv(
            file_path=PRICE_FILE,
            file_description=(
                "Histórico de preços"
            ),
        )

        market_summary = pd.DataFrame()

    else:

        market_engine = MarketData()

        price_history = (
            market_engine
            .download_prices()
        )

        market_summary = (
            market_engine
            .get_summary()
        )

    validate_dataframe(
        data=price_history,
        dataframe_name=(
            "Histórico de preços"
        ),
    )

    print(
        f"✅ Histórico disponível: "
        f"{len(price_history):,} linhas"
    )

    # --------------------------------------------------------
    # 2. INDICADORES
    # --------------------------------------------------------

    print_stage(
        2,
        "CÁLCULO DOS INDICADORES TÉCNICOS",
    )

    if USE_EXISTING_INDICATORS:

        indicator_history = load_csv(
            file_path=INDICATOR_FILE,
            file_description=(
                "Histórico de indicadores"
            ),
        )

        indicator_engine = TechnicalIndicators()

        indicator_engine.history = (
            indicator_history.copy()
        )

        latest_indicators = (
            indicator_history
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

        indicator_engine.latest = (
            latest_indicators
        )

    else:

        indicator_engine = (
            TechnicalIndicators()
        )

        indicator_history = (
            indicator_engine.calculate(
                price_history=price_history,
                save=SAVE_INTERMEDIATE_FILES,
            )
        )

        latest_indicators = (
            indicator_engine
            .get_latest()
        )

    validate_dataframe(
        data=indicator_history,
        dataframe_name=(
            "Histórico de indicadores"
        ),
    )

    validate_dataframe(
        data=latest_indicators,
        dataframe_name=(
            "Indicadores atuais"
        ),
    )

    # --------------------------------------------------------
    # 3. INSTITUTIONAL SCORE
    # --------------------------------------------------------

    print_stage(
        3,
        "INSTITUTIONAL GROWTH & MONEY FLOW SCORE",
    )

    institutional_engine = (
        InstitutionalScore()
    )

    institutional_ranking = (
        institutional_engine.calculate(
            indicators=indicator_history,
            save=SAVE_INTERMEDIATE_FILES,
        )
    )

    validate_dataframe(
        data=institutional_ranking,
        dataframe_name=(
            "Ranking institucional"
        ),
    )

    sector_strength = (
        institutional_engine
        .get_sector_table()
    )

    # --------------------------------------------------------
    # 4. TECHNICAL SCORE
    # --------------------------------------------------------

    print_stage(
        4,
        "TECHNICAL ENTRY SCORE",
    )

    technical_engine = (
        TechnicalScore()
    )

    technical_ranking = (
        technical_engine.calculate(
            indicators=indicator_history,
            save=SAVE_INTERMEDIATE_FILES,
        )
    )

    validate_dataframe(
        data=technical_ranking,
        dataframe_name=(
            "Ranking técnico"
        ),
    )

    # --------------------------------------------------------
    # 5. SIGNAL ENGINE
    # --------------------------------------------------------

    print_stage(
        5,
        "COMBINAÇÃO DOS MOTORES E GERAÇÃO DE SINAIS",
    )

    signal_engine = (
        SignalEngine()
    )

    signals = signal_engine.calculate(
        institutional_data=(
            institutional_ranking
        ),
        technical_data=(
            technical_ranking
        ),
        save=SAVE_INTERMEDIATE_FILES,
    )

    validate_dataframe(
        data=signals,
        dataframe_name=(
            "Sinais consolidados"
        ),
    )

    # --------------------------------------------------------
    # 6. RANKING
    # --------------------------------------------------------

    print_stage(
        6,
        "RANKING FINAL",
    )

    ranking_engine = (
        RankingEngine()
    )

    executive_ranking = (
        ranking_engine.calculate(
            signals=signals,
            save=SAVE_INTERMEDIATE_FILES,
        )
    )

    validate_dataframe(
        data=executive_ranking,
        dataframe_name=(
            "Ranking executivo"
        ),
    )

    show_top_ranking(
        ranking=executive_ranking,
        top_n=TOP_N_DISPLAY,
    )

    # --------------------------------------------------------
    # 7. RELATÓRIO
    # --------------------------------------------------------

    report_path: Path | None = None

    if GENERATE_EXCEL_REPORT:

        print_stage(
            7,
            "GERAÇÃO DO RELATÓRIO EXCEL",
        )

        report_engine = (
            ReportEngine()
        )

        report_path = (
            report_engine.generate(
                ranking=(
                    ranking_engine
                    .get_full_ranking()
                ),
                approved_entries=(
                    ranking_engine
                    .get_approved_entries()
                ),
                watchlist=(
                    ranking_engine
                    .get_watchlist()
                ),
                best_by_sector=(
                    ranking_engine
                    .get_best_by_sector()
                ),
            )
        )

    # --------------------------------------------------------
    # RESUMO
    # --------------------------------------------------------

    print_final_summary(
        ranking_engine=ranking_engine,
        report_path=report_path,
        execution_start=execution_start,
    )

    return {
        "market_summary":
            market_summary,

        "price_history":
            price_history,

        "indicator_history":
            indicator_history,

        "latest_indicators":
            latest_indicators,

        "institutional_ranking":
            institutional_ranking,

        "sector_strength":
            sector_strength,

        "technical_ranking":
            technical_ranking,

        "signals":
            signals,

        "ranking":
            ranking_engine
            .get_full_ranking(),

        "executive_ranking":
            executive_ranking,

        "approved_entries":
            ranking_engine
            .get_approved_entries(),

        "watchlist":
            ranking_engine
            .get_watchlist(),

        "best_by_sector":
            ranking_engine
            .get_best_by_sector(),

        "report_path":
            report_path,
    }


# ============================================================
# EXECUÇÃO
# ============================================================

def main() -> int:
    """
    Ponto de entrada do programa.
    """

    try:

        run_scanner()

        return 0

    except KeyboardInterrupt:

        print()
        print(
            "Execução interrompida pelo usuário."
        )

        return 130

    except Exception as error:

        print()
        print("=" * 120)
        print("ERRO NA EXECUÇÃO DO SCANNER")
        print("=" * 120)

        print(
            f"Tipo: {type(error).__name__}"
        )

        print(
            f"Mensagem: {error}"
        )

        print()
        print("Detalhes técnicos:")

        traceback.print_exc()

        print("=" * 120)

        return 1


if __name__ == "__main__":

    sys.exit(
        main()
    )
