# ============================================================
# AI INFRASTRUCTURE SCANNER
# report.py
#
# Gerador do relatório executivo em Excel.
#
# Abas:
# - RESUMO
# - TOP_20
# - RANKING_COMPLETO
# - ENTRADAS_APROVADAS
# - WATCHLIST
# - MELHOR_POR_SETOR
#
# Arquivo final:
# data/relatorio.xlsx
# ============================================================

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import (
    DATA_PATH,
    RANKING_FILE,
    REPORT_FILE,
    TOP_N,
)

warnings.filterwarnings("ignore")


# ============================================================
# ARQUIVOS
# ============================================================

APPROVED_ENTRIES_FILE = (
    Path(DATA_PATH)
    / "entradas_aprovadas.csv"
)

WATCHLIST_FILE = (
    Path(DATA_PATH)
    / "watchlist.csv"
)

BEST_BY_SECTOR_FILE = (
    Path(DATA_PATH)
    / "melhor_por_setor.csv"
)


# ============================================================
# CONFIGURAÇÕES DO EXCEL
# ============================================================

SHEET_SUMMARY = "RESUMO"
SHEET_TOP = "TOP_20"
SHEET_RANKING = "RANKING_COMPLETO"
SHEET_APPROVED = "ENTRADAS_APROVADAS"
SHEET_WATCHLIST = "WATCHLIST"
SHEET_SECTOR = "MELHOR_POR_SETOR"

HEADER_COLOR = "#1F4E78"
HEADER_FONT_COLOR = "#FFFFFF"

BUY_COLOR = "#C6EFCE"
BUY_FONT_COLOR = "#006100"

WAIT_COLOR = "#FFEB9C"
WAIT_FONT_COLOR = "#9C6500"

AVOID_COLOR = "#FFC7CE"
AVOID_FONT_COLOR = "#9C0006"

NEUTRAL_COLOR = "#D9EAF7"
NEUTRAL_FONT_COLOR = "#1F1F1F"


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


def boolean_value(
    value: Any,
) -> bool:
    """
    Converte diferentes formatos para booleano.
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
        return bool(value)

    except Exception:
        return False


def normalize_columns(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Padroniza nomes de colunas.
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
        "segmento": "setor",
        "empresa": "company",
        "nome_empresa": "company",
        "score_final": "final_score",
        "status_sinal": "signal_status",
        "entrada_aprovada": "signal_approved",
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


def prepare_dataframe(
    data: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Prepara um DataFrame para exportação.
    """

    if data is None:
        return pd.DataFrame()

    if not isinstance(
        data,
        pd.DataFrame,
    ):
        raise TypeError(
            "O objeto informado deve ser um pandas DataFrame."
        )

    if data.empty:
        return pd.DataFrame()

    df = normalize_columns(
        data
    )

    if "ticker" in df.columns:

        df["ticker"] = (
            df["ticker"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

    if "signal_approved" in df.columns:

        df["signal_approved"] = (
            df["signal_approved"]
            .apply(
                boolean_value
            )
        )

    date_columns = [
        column
        for column in df.columns
        if (
            column == "date"
            or column.startswith("data_")
            or column.endswith("_date")
        )
    ]

    for column in date_columns:

        df[column] = pd.to_datetime(
            df[column],
            errors="coerce",
        )

    return df.reset_index(
        drop=True
    )


def load_csv_if_exists(
    file_path: str | Path,
) -> pd.DataFrame:
    """
    Carrega um CSV caso ele exista.
    """

    path = Path(
        file_path
    )

    if not path.exists():
        return pd.DataFrame()

    try:

        return prepare_dataframe(
            pd.read_csv(
                path
            )
        )

    except Exception as error:

        print(
            f"⚠️ Falha ao carregar {path.name}: "
            f"{error}"
        )

        return pd.DataFrame()


def safe_mean(
    data: pd.DataFrame,
    column: str,
) -> float:
    """
    Calcula média com segurança.
    """

    if (
        data.empty
        or column not in data.columns
    ):
        return np.nan

    values = pd.to_numeric(
        data[column],
        errors="coerce",
    )

    return float(
        values.mean()
    )


def safe_max(
    data: pd.DataFrame,
    column: str,
) -> float:
    """
    Calcula máximo com segurança.
    """

    if (
        data.empty
        or column not in data.columns
    ):
        return np.nan

    values = pd.to_numeric(
        data[column],
        errors="coerce",
    )

    return float(
        values.max()
    )


def safe_count_true(
    data: pd.DataFrame,
    column: str,
) -> int:
    """
    Conta valores verdadeiros.
    """

    if (
        data.empty
        or column not in data.columns
    ):
        return 0

    return int(
        data[column]
        .apply(
            boolean_value
        )
        .sum()
    )


# ============================================================
# RESUMO EXECUTIVO
# ============================================================

def build_summary(
    ranking: pd.DataFrame,
    approved_entries: pd.DataFrame,
    watchlist: pd.DataFrame,
    best_by_sector: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cria o resumo executivo do scanner.
    """

    total_companies = len(
        ranking
    )

    total_sectors = (
        ranking["setor"].nunique()
        if (
            not ranking.empty
            and "setor" in ranking.columns
        )
        else 0
    )

    approved_count = (
        len(approved_entries)
        if not approved_entries.empty
        else safe_count_true(
            ranking,
            "signal_approved",
        )
    )

    watchlist_count = len(
        watchlist
    )

    best_ticker = ""

    best_status = ""

    best_score = np.nan

    if not ranking.empty:

        first_row = ranking.iloc[0]

        best_ticker = str(
            first_row.get(
                "ticker",
                "",
            )
        )

        best_status = str(
            first_row.get(
                "signal_status",
                "",
            )
        )

        best_score = first_row.get(
            "final_score",
            np.nan,
        )

    summary_records = [
        {
            "Métrica": "Data da execução",
            "Valor": datetime.now().strftime(
                "%d/%m/%Y %H:%M"
            ),
        },
        {
            "Métrica": "Empresas analisadas",
            "Valor": total_companies,
        },
        {
            "Métrica": "Setores analisados",
            "Valor": total_sectors,
        },
        {
            "Métrica": "Entradas aprovadas",
            "Valor": approved_count,
        },
        {
            "Métrica": "Ações na watchlist",
            "Valor": watchlist_count,
        },
        {
            "Métrica": "Setores com representante",
            "Valor": len(
                best_by_sector
            ),
        },
        {
            "Métrica": "Melhor ticker atual",
            "Valor": best_ticker,
        },
        {
            "Métrica": "Status da melhor ação",
            "Valor": best_status,
        },
        {
            "Métrica": "Maior Final Score",
            "Valor": (
                round(
                    float(best_score),
                    2,
                )
                if is_valid_number(
                    best_score
                )
                else ""
            ),
        },
        {
            "Métrica": "Final Score médio",
            "Valor": round(
                safe_mean(
                    ranking,
                    "final_score",
                ),
                2,
            ),
        },
        {
            "Métrica": "Institutional Score médio",
            "Valor": round(
                safe_mean(
                    ranking,
                    "institutional_score",
                ),
                2,
            ),
        },
        {
            "Métrica": "Technical Score médio",
            "Valor": round(
                safe_mean(
                    ranking,
                    "technical_entry_score",
                ),
                2,
            ),
        },
        {
            "Métrica": "Maior Institutional Score",
            "Valor": round(
                safe_max(
                    ranking,
                    "institutional_score",
                ),
                2,
            ),
        },
        {
            "Métrica": "Maior Technical Score",
            "Valor": round(
                safe_max(
                    ranking,
                    "technical_entry_score",
                ),
                2,
            ),
        },
    ]

    summary = pd.DataFrame(
        summary_records
    )

    summary["Valor"] = (
        summary["Valor"]
        .replace(
            {
                np.nan: "",
                np.inf: "",
                -np.inf: "",
            }
        )
    )

    return summary


# ============================================================
# SELEÇÃO DAS COLUNAS
# ============================================================

def select_report_columns(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Seleciona e organiza as colunas mais relevantes.
    """

    if data.empty:
        return pd.DataFrame()

    desired_columns = [
        "ranking",
        "ticker",
        "company",
        "setor",
        "signal_status",
        "signal_approved",
        "ranking_quality",
        "opportunity_profile",
        "priority_score",
        "final_score",
        "signal_strength_score",
        "institutional_score",
        "technical_entry_score",
        "institutional_classification",
        "technical_classification",
        "institutional_diagnosis",
        "technical_diagnosis",
        "close",
        "rsi_14",
        "macd_hist",
        "adx_14",
        "mfi_14",
        "atr_percentual",
        "distancia_maxima_52s",
        "distancia_sma_50",
        "distancia_sma_200",
        "volume_relativo_20d",
        "cmf_20",
        "score_money_flow",
        "score_market_leadership",
        "score_sector_strength",
        "score_discount",
        "score_momentum",
        "score_trend",
        "score_volume_flow",
        "score_risk",
        "signal_positive_factors",
        "signal_pending_conditions",
        "signal_rejection_reasons",
        "executive_decision",
    ]

    available_columns = [
        column
        for column in desired_columns
        if column in data.columns
    ]

    remaining_columns = [
        column
        for column in data.columns
        if column not in available_columns
    ]

    ordered_columns = (
        available_columns
        +
        remaining_columns
    )

    return data[
        ordered_columns
    ].copy()


# ============================================================
# FORMATAÇÃO DO EXCEL
# ============================================================

def calculate_column_width(
    data: pd.DataFrame,
    column: str,
    maximum_width: int = 55,
) -> int:
    """
    Calcula uma largura adequada para a coluna.
    """

    header_length = len(
        str(column)
    )

    if data.empty:
        return min(
            max(
                header_length + 2,
                12,
            ),
            maximum_width,
        )

    values = (
        data[column]
        .astype(str)
        .replace(
            {
                "nan": "",
                "NaT": "",
                "None": "",
            }
        )
    )

    maximum_content = (
        values.str.len().max()
    )

    if pd.isna(
        maximum_content
    ):
        maximum_content = 0

    width = max(
        header_length + 2,
        int(maximum_content) + 2,
        12,
    )

    return min(
        width,
        maximum_width,
    )


def get_status_format(
    workbook: Any,
    status: str,
) -> Any:
    """
    Retorna a formatação conforme o status.
    """

    normalized = str(
        status
    ).upper()

    if "ENTRADA APROVADA" in normalized:

        return workbook.add_format(
            {
                "bg_color": BUY_COLOR,
                "font_color": BUY_FONT_COLOR,
                "bold": True,
                "border": 1,
            }
        )

    if (
        "AGUARDAR" in normalized
        or "OBSERVAÇÃO PRIORITÁRIA" in normalized
    ):

        return workbook.add_format(
            {
                "bg_color": WAIT_COLOR,
                "font_color": WAIT_FONT_COLOR,
                "bold": True,
                "border": 1,
            }
        )

    if (
        "NÃO COMPRAR" in normalized
        or "FRACA" in normalized
    ):

        return workbook.add_format(
            {
                "bg_color": AVOID_COLOR,
                "font_color": AVOID_FONT_COLOR,
                "bold": True,
                "border": 1,
            }
        )

    return workbook.add_format(
        {
            "bg_color": NEUTRAL_COLOR,
            "font_color": NEUTRAL_FONT_COLOR,
            "border": 1,
        }
    )


def format_worksheet(
    writer: pd.ExcelWriter,
    sheet_name: str,
    data: pd.DataFrame,
) -> None:
    """
    Aplica formatação profissional a uma planilha.
    """

    workbook = writer.book
    worksheet = writer.sheets[
        sheet_name
    ]

    header_format = workbook.add_format(
        {
            "bold": True,
            "font_color": HEADER_FONT_COLOR,
            "bg_color": HEADER_COLOR,
            "border": 1,
            "align": "center",
            "valign": "vcenter",
            "text_wrap": True,
        }
    )

    integer_format = workbook.add_format(
        {
            "num_format": "0",
            "border": 1,
        }
    )

    decimal_format = workbook.add_format(
        {
            "num_format": "0.00",
            "border": 1,
        }
    )

    percentage_format = workbook.add_format(
        {
            "num_format": "0.00",
            "border": 1,
        }
    )

    date_format = workbook.add_format(
        {
            "num_format": "dd/mm/yyyy",
            "border": 1,
        }
    )

    text_format = workbook.add_format(
        {
            "border": 1,
            "valign": "top",
            "text_wrap": True,
        }
    )

    center_format = workbook.add_format(
        {
            "border": 1,
            "align": "center",
            "valign": "vcenter",
        }
    )

    for column_index, column in enumerate(
        data.columns
    ):

        worksheet.write(
            0,
            column_index,
            column,
            header_format,
        )

        width = calculate_column_width(
            data=data,
            column=column,
        )

        worksheet.set_column(
            column_index,
            column_index,
            width,
        )

        column_lower = str(
            column
        ).lower()

        if (
            column_lower == "ranking"
            or column_lower.endswith("_ranking")
        ):

            worksheet.set_column(
                column_index,
                column_index,
                width,
                integer_format,
            )

        elif (
            "score" in column_lower
            or "rsi" in column_lower
            or "adx" in column_lower
            or "mfi" in column_lower
        ):

            worksheet.set_column(
                column_index,
                column_index,
                width,
                decimal_format,
            )

        elif (
            "percent" in column_lower
            or "distancia" in column_lower
            or "retorno" in column_lower
            or "volume_relativo" in column_lower
            or column_lower == "cmf_20"
        ):

            worksheet.set_column(
                column_index,
                column_index,
                width,
                percentage_format,
            )

        elif (
            "date" in column_lower
            or column_lower.startswith(
                "data_"
            )
        ):

            worksheet.set_column(
                column_index,
                column_index,
                width,
                date_format,
            )

        elif (
            column_lower
            in {
                "ticker",
                "setor",
                "signal_status",
                "signal_approved",
            }
        ):

            worksheet.set_column(
                column_index,
                column_index,
                width,
                center_format,
            )

        else:

            worksheet.set_column(
                column_index,
                column_index,
                width,
                text_format,
            )

    worksheet.freeze_panes(
        1,
        0,
    )

    worksheet.autofilter(
        0,
        0,
        max(
            len(data),
            1,
        ),
        max(
            len(data.columns) - 1,
            0,
        ),
    )

    worksheet.set_row(
        0,
        34,
    )

    if (
        not data.empty
        and "signal_status"
        in data.columns
    ):

        status_column_index = (
            data.columns.get_loc(
                "signal_status"
            )
        )

        for row_index, status in enumerate(
            data["signal_status"],
            start=1,
        ):

            status_format = (
                get_status_format(
                    workbook=workbook,
                    status=status,
                )
            )

            worksheet.write(
                row_index,
                status_column_index,
                status,
                status_format,
            )

    if (
        not data.empty
        and "signal_approved"
        in data.columns
    ):

        approved_column_index = (
            data.columns.get_loc(
                "signal_approved"
            )
        )

        approved_format = workbook.add_format(
            {
                "bg_color": BUY_COLOR,
                "font_color": BUY_FONT_COLOR,
                "bold": True,
                "border": 1,
                "align": "center",
            }
        )

        rejected_format = workbook.add_format(
            {
                "bg_color": AVOID_COLOR,
                "font_color": AVOID_FONT_COLOR,
                "border": 1,
                "align": "center",
            }
        )

        for row_index, value in enumerate(
            data["signal_approved"],
            start=1,
        ):

            approved = boolean_value(
                value
            )

            worksheet.write(
                row_index,
                approved_column_index,
                "SIM" if approved else "NÃO",
                (
                    approved_format
                    if approved
                    else rejected_format
                ),
            )

    if (
        not data.empty
        and "final_score"
        in data.columns
    ):

        final_score_column = (
            data.columns.get_loc(
                "final_score"
            )
        )

        worksheet.conditional_format(
            1,
            final_score_column,
            len(data),
            final_score_column,
            {
                "type": "3_color_scale",
                "min_color": "#F8696B",
                "mid_color": "#FFEB84",
                "max_color": "#63BE7B",
            },
        )


def format_summary_sheet(
    writer: pd.ExcelWriter,
    summary: pd.DataFrame,
) -> None:
    """
    Aplica formatação à aba RESUMO.
    """

    workbook = writer.book
    worksheet = writer.sheets[
        SHEET_SUMMARY
    ]

    title_format = workbook.add_format(
        {
            "bold": True,
            "font_size": 18,
            "font_color": "#FFFFFF",
            "bg_color": HEADER_COLOR,
            "align": "center",
            "valign": "vcenter",
            "border": 1,
        }
    )

    subtitle_format = workbook.add_format(
        {
            "bold": True,
            "font_size": 11,
            "font_color": "#1F1F1F",
            "bg_color": "#D9EAF7",
            "align": "center",
            "border": 1,
        }
    )

    metric_format = workbook.add_format(
        {
            "bold": True,
            "bg_color": "#EAF2F8",
            "border": 1,
        }
    )

    value_format = workbook.add_format(
        {
            "border": 1,
            "align": "left",
        }
    )

    worksheet.merge_range(
        "A1:B1",
        "AI INFRASTRUCTURE SCANNER",
        title_format,
    )

    worksheet.merge_range(
        "A2:B2",
        "Relatório executivo de oportunidades para swing trade",
        subtitle_format,
    )

    start_row = 3

    worksheet.write(
        start_row,
        0,
        "Métrica",
        metric_format,
    )

    worksheet.write(
        start_row,
        1,
        "Valor",
        metric_format,
    )

    for index, row in summary.iterrows():

        excel_row = (
            start_row
            +
            index
            +
            1
        )

        worksheet.write(
            excel_row,
            0,
            row["Métrica"],
            metric_format,
        )

        worksheet.write(
            excel_row,
            1,
            row["Valor"],
            value_format,
        )

    worksheet.set_column(
        "A:A",
        34,
    )

    worksheet.set_column(
        "B:B",
        35,
    )

    worksheet.set_row(
        0,
        32,
    )

    worksheet.freeze_panes(
        4,
        0,
    )


# ============================================================
# CLASSE PRINCIPAL
# ============================================================

class ReportEngine:
    """
    Motor responsável pelo relatório Excel.
    """

    def __init__(
        self,
        output_file: str | Path = REPORT_FILE,
        top_n: int = TOP_N,
    ) -> None:
        self.output_file = Path(
            output_file
        )

        self.top_n = int(
            top_n
        )

        self.summary = pd.DataFrame()

    def generate(
        self,
        ranking: pd.DataFrame,
        approved_entries: pd.DataFrame | None = None,
        watchlist: pd.DataFrame | None = None,
        best_by_sector: pd.DataFrame | None = None,
    ) -> Path:
        """
        Gera o relatório completo em Excel.

        Parameters
        ----------
        ranking:
            Ranking produzido por ranking_engine.py.

        approved_entries:
            Entradas aprovadas.

        watchlist:
            Ações que aguardam confirmação.

        best_by_sector:
            Melhor ação de cada setor.

        Returns
        -------
        pathlib.Path
            Caminho do relatório gerado.
        """

        ranking = prepare_dataframe(
            ranking
        )

        if ranking.empty:
            raise ValueError(
                "O ranking está vazio. "
                "Não é possível gerar o relatório."
            )

        approved_entries = (
            prepare_dataframe(
                approved_entries
            )
        )

        watchlist = prepare_dataframe(
            watchlist
        )

        best_by_sector = (
            prepare_dataframe(
                best_by_sector
            )
        )

        if approved_entries.empty:

            if "signal_approved" in ranking.columns:

                approved_entries = (
                    ranking.loc[
                        ranking[
                            "signal_approved"
                        ].apply(
                            boolean_value
                        )
                    ]
                    .copy()
                    .reset_index(drop=True)
                )

        if watchlist.empty:

            watch_status = {
                "AGUARDAR GATILHO",
                "AGUARDAR CONFIRMAÇÃO TÉCNICA",
                "OBSERVAÇÃO PRIORITÁRIA",
            }

            if "signal_status" in ranking.columns:

                watchlist = (
                    ranking.loc[
                        ranking[
                            "signal_status"
                        ].isin(
                            watch_status
                        )
                    ]
                    .copy()
                    .reset_index(drop=True)
                )

        if (
            best_by_sector.empty
            and "setor" in ranking.columns
        ):

            best_by_sector = (
                ranking
                .sort_values(
                    [
                        column
                        for column in [
                            "signal_approved",
                            "priority_score",
                            "final_score",
                        ]
                        if column in ranking.columns
                    ],
                    ascending=[
                        False,
                        False,
                        False,
                    ][
                        :len(
                            [
                                column
                                for column in [
                                    "signal_approved",
                                    "priority_score",
                                    "final_score",
                                ]
                                if column in ranking.columns
                            ]
                        )
                    ],
                )
                .groupby(
                    "setor",
                    as_index=False,
                )
                .first()
                .reset_index(drop=True)
            )

        self.summary = build_summary(
            ranking=ranking,
            approved_entries=approved_entries,
            watchlist=watchlist,
            best_by_sector=best_by_sector,
        )

        ranking_report = (
            select_report_columns(
                ranking
            )
        )

        top_report = (
            ranking_report
            .head(
                self.top_n
            )
            .copy()
            .reset_index(drop=True)
        )

        approved_report = (
            select_report_columns(
                approved_entries
            )
        )

        watchlist_report = (
            select_report_columns(
                watchlist
            )
        )

        sector_report = (
            select_report_columns(
                best_by_sector
            )
        )

        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with pd.ExcelWriter(
            self.output_file,
            engine="xlsxwriter",
            datetime_format="dd/mm/yyyy",
            date_format="dd/mm/yyyy",
        ) as writer:

            self.summary.to_excel(
                writer,
                sheet_name=SHEET_SUMMARY,
                index=False,
                startrow=3,
            )

            top_report.to_excel(
                writer,
                sheet_name=SHEET_TOP,
                index=False,
            )

            ranking_report.to_excel(
                writer,
                sheet_name=SHEET_RANKING,
                index=False,
            )

            approved_report.to_excel(
                writer,
                sheet_name=SHEET_APPROVED,
                index=False,
            )

            watchlist_report.to_excel(
                writer,
                sheet_name=SHEET_WATCHLIST,
                index=False,
            )

            sector_report.to_excel(
                writer,
                sheet_name=SHEET_SECTOR,
                index=False,
            )

            format_summary_sheet(
                writer=writer,
                summary=self.summary,
            )

            format_worksheet(
                writer=writer,
                sheet_name=SHEET_TOP,
                data=top_report,
            )

            format_worksheet(
                writer=writer,
                sheet_name=SHEET_RANKING,
                data=ranking_report,
            )

            format_worksheet(
                writer=writer,
                sheet_name=SHEET_APPROVED,
                data=approved_report,
            )

            format_worksheet(
                writer=writer,
                sheet_name=SHEET_WATCHLIST,
                data=watchlist_report,
            )

            format_worksheet(
                writer=writer,
                sheet_name=SHEET_SECTOR,
                data=sector_report,
            )

        self.print_summary(
            ranking=ranking,
            approved_entries=approved_entries,
            watchlist=watchlist,
            best_by_sector=best_by_sector,
        )

        return self.output_file

    def print_summary(
        self,
        ranking: pd.DataFrame,
        approved_entries: pd.DataFrame,
        watchlist: pd.DataFrame,
        best_by_sector: pd.DataFrame,
    ) -> None:
        """
        Exibe o resumo da geração do relatório.
        """

        print()
        print("=" * 105)
        print("REPORT ENGINE — RELATÓRIO EXECUTIVO")
        print("=" * 105)

        print(
            f"Empresas no ranking: "
            f"{len(ranking)}"
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
            f"Relatório criado em: "
            f"{self.output_file}"
        )

        print("=" * 105)


# ============================================================
# FUNÇÃO SIMPLIFICADA
# ============================================================

def generate_report(
    ranking: pd.DataFrame,
    approved_entries: pd.DataFrame | None = None,
    watchlist: pd.DataFrame | None = None,
    best_by_sector: pd.DataFrame | None = None,
    output_file: str | Path = REPORT_FILE,
) -> Path:
    """
    Interface simplificada para gerar o relatório.
    """

    engine = ReportEngine(
        output_file=output_file
    )

    return engine.generate(
        ranking=ranking,
        approved_entries=approved_entries,
        watchlist=watchlist,
        best_by_sector=best_by_sector,
    )


# ============================================================
# EXECUÇÃO DIRETA
# ============================================================

if __name__ == "__main__":

    if not Path(
        RANKING_FILE
    ).exists():

        raise FileNotFoundError(
            "O arquivo ranking.csv não foi encontrado. "
            "Execute primeiro ranking_engine.py."
        )

    ranking_data = pd.read_csv(
        RANKING_FILE
    )

    approved_data = load_csv_if_exists(
        APPROVED_ENTRIES_FILE
    )

    watchlist_data = load_csv_if_exists(
        WATCHLIST_FILE
    )

    sector_data = load_csv_if_exists(
        BEST_BY_SECTOR_FILE
    )

    report_engine = ReportEngine()

    report_path = (
        report_engine.generate(
            ranking=ranking_data,
            approved_entries=approved_data,
            watchlist=watchlist_data,
            best_by_sector=sector_data,
        )
    )

    print()
    print(
        f"✅ Relatório final disponível em: "
        f"{report_path}"
    )
