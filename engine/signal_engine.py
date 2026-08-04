# ============================================================
# AI INFRASTRUCTURE SCANNER
# signal_engine.py
#
# Motor de decisão do scanner.
#
# Objetivo:
# Combinar o Institutional Score e o Technical Entry Score
# para produzir uma decisão prática de swing trade.
#
# Saídas principais:
# - final_score
# - signal_status
# - signal_approved
# - signal_strength
# - positive_factors
# - pending_conditions
# - rejection_reasons
# ============================================================

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import (
    DATA_PATH,
    MIN_TECHNICAL_SCORE,
    MIN_INSTITUTIONAL_SCORE,
    MIN_FINAL_SCORE,
    WEIGHT_INSTITUTIONAL,
    WEIGHT_TECHNICAL,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURAÇÕES
# ============================================================

SIGNAL_OUTPUT_FILE = (
    Path(DATA_PATH)
    / "signals.csv"
)

MAX_SCORE = 100.0

MINIMUM_TECHNICAL_COLUMNS = {
    "ticker",
    "technical_entry_score",
    "technical_entry_approved",
    "technical_classification",
    "technical_diagnosis",
}

MINIMUM_INSTITUTIONAL_COLUMNS = {
    "ticker",
    "institutional_score",
    "institutional_flow_approved",
    "institutional_classification",
    "institutional_diagnosis",
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
    Limita uma nota ao intervalo entre 0 e 100.
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
        "entrada_tecnica_aprovada":
            "technical_entry_approved",

        "classificacao_tecnica":
            "technical_classification",

        "diagnostico_entrada":
            "technical_diagnosis",

        "institutional_flow_approved":
            "institutional_flow_approved",
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


def validate_dataframe(
    data: pd.DataFrame,
    required_columns: set[str],
    dataframe_name: str,
) -> pd.DataFrame:
    """
    Valida um DataFrame utilizado pelo Signal Engine.
    """

    if not isinstance(
        data,
        pd.DataFrame,
    ):
        raise TypeError(
            f"{dataframe_name} deve ser um pandas DataFrame."
        )

    if data.empty:
        raise ValueError(
            f"{dataframe_name} está vazio."
        )

    df = normalize_columns(
        data
    )

    missing_columns = (
        required_columns
        .difference(
            df.columns
        )
    )

    if missing_columns:
        raise KeyError(
            f"Colunas ausentes em {dataframe_name}: "
            f"{sorted(missing_columns)}"
        )

    df["ticker"] = (
        df["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df = (
        df.dropna(
            subset=["ticker"]
        )
        .drop_duplicates(
            subset=["ticker"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError(
            f"Nenhuma linha válida permaneceu em "
            f"{dataframe_name}."
        )

    return df


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
                "yes",
                "sim",
                "verdadeiro",
            }
        )

    try:
        return bool(value)

    except Exception:
        return False


# ============================================================
# SCORE FINAL
# ============================================================

def calculate_combined_score(
    row: pd.Series,
) -> float:
    """
    Combina Institutional Score e Technical Entry Score.
    """

    institutional_score = row.get(
        "institutional_score",
        np.nan,
    )

    technical_score = row.get(
        "technical_entry_score",
        np.nan,
    )

    total_weight = (
        WEIGHT_INSTITUTIONAL
        +
        WEIGHT_TECHNICAL
    )

    if total_weight <= 0:
        raise ValueError(
            "A soma dos pesos do ranking final "
            "deve ser maior que zero."
        )

    available_scores = []
    available_weights = []

    if is_valid_number(
        institutional_score
    ):
        available_scores.append(
            float(institutional_score)
        )

        available_weights.append(
            float(WEIGHT_INSTITUTIONAL)
        )

    if is_valid_number(
        technical_score
    ):
        available_scores.append(
            float(technical_score)
        )

        available_weights.append(
            float(WEIGHT_TECHNICAL)
        )

    if not available_scores:
        return 0.0

    final_score = np.average(
        available_scores,
        weights=available_weights,
    )

    return clip_score(
        final_score
    )


# ============================================================
# CONFLUÊNCIA DOS MOTORES
# ============================================================

def calculate_confluence_score(
    row: pd.Series,
) -> float:
    """
    Mede o grau de concordância entre os motores.

    Quanto mais próximos os dois scores, maior a confluência.
    """

    institutional_score = row.get(
        "institutional_score",
        np.nan,
    )

    technical_score = row.get(
        "technical_entry_score",
        np.nan,
    )

    if not (
        is_valid_number(
            institutional_score
        )
        and
        is_valid_number(
            technical_score
        )
    ):
        return 0.0

    difference = abs(
        float(institutional_score)
        -
        float(technical_score)
    )

    confluence = (
        100
        -
        difference
    )

    return clip_score(
        confluence
    )


def calculate_signal_strength(
    row: pd.Series,
) -> float:
    """
    Combina a nota final com a confluência dos motores.
    """

    final_score = row.get(
        "final_score",
        0,
    )

    confluence_score = row.get(
        "confluence_score",
        0,
    )

    institutional_approved = boolean_value(
        row.get(
            "institutional_flow_approved",
            False,
        )
    )

    technical_approved = boolean_value(
        row.get(
            "technical_entry_approved",
            False,
        )
    )

    score = (
        float(final_score) * 0.80
        +
        float(confluence_score) * 0.20
    )

    if (
        institutional_approved
        and technical_approved
    ):
        score += 5

    return clip_score(
        score
    )


# ============================================================
# CONDIÇÕES DE APROVAÇÃO
# ============================================================

def institutional_condition(
    row: pd.Series,
) -> bool:
    """
    Verifica se o filtro institucional foi aprovado.
    """

    score = row.get(
        "institutional_score",
        np.nan,
    )

    approved = boolean_value(
        row.get(
            "institutional_flow_approved",
            False,
        )
    )

    return bool(
        is_valid_number(score)
        and float(score)
        >= MIN_INSTITUTIONAL_SCORE
        and approved
    )


def technical_condition(
    row: pd.Series,
) -> bool:
    """
    Verifica se o filtro técnico foi aprovado.
    """

    score = row.get(
        "technical_entry_score",
        np.nan,
    )

    approved = boolean_value(
        row.get(
            "technical_entry_approved",
            False,
        )
    )

    return bool(
        is_valid_number(score)
        and float(score)
        >= MIN_TECHNICAL_SCORE
        and approved
    )


def final_score_condition(
    row: pd.Series,
) -> bool:
    """
    Verifica se o score combinado passou do mínimo.
    """

    score = row.get(
        "final_score",
        np.nan,
    )

    return bool(
        is_valid_number(score)
        and float(score)
        >= MIN_FINAL_SCORE
    )


def calculate_signal_approval(
    row: pd.Series,
) -> bool:
    """
    Aprovação final do sinal.
    """

    return bool(
        institutional_condition(row)
        and technical_condition(row)
        and final_score_condition(row)
    )


# ============================================================
# STATUS DO SINAL
# ============================================================

def define_signal_status(
    row: pd.Series,
) -> str:
    """
    Define o status operacional da ação.
    """

    institutional_score = row.get(
        "institutional_score",
        0,
    )

    technical_score = row.get(
        "technical_entry_score",
        0,
    )

    final_score = row.get(
        "final_score",
        0,
    )

    institutional_approved = (
        institutional_condition(row)
    )

    technical_approved = (
        technical_condition(row)
    )

    signal_approved = boolean_value(
        row.get(
            "signal_approved",
            False,
        )
    )

    technical_diagnosis = str(
        row.get(
            "technical_diagnosis",
            "",
        )
    ).upper()

    if signal_approved:
        return "ENTRADA APROVADA"

    if (
        institutional_approved
        and
        is_valid_number(technical_score)
        and float(technical_score)
        >= MIN_TECHNICAL_SCORE
        and not technical_approved
    ):
        return "AGUARDAR CONFIRMAÇÃO TÉCNICA"

    if (
        institutional_approved
        and
        (
            "AGUARDAR GATILHO"
            in technical_diagnosis
            or
            "REVERSÃO EM FORMAÇÃO"
            in technical_diagnosis
        )
    ):
        return "AGUARDAR GATILHO"

    if (
        technical_approved
        and
        is_valid_number(institutional_score)
        and float(institutional_score)
        < MIN_INSTITUTIONAL_SCORE
    ):
        return "TÉCNICA BOA — FLUXO INSUFICIENTE"

    if (
        is_valid_number(final_score)
        and float(final_score) >= 65
    ):
        return "OBSERVAÇÃO PRIORITÁRIA"

    if (
        is_valid_number(institutional_score)
        and float(institutional_score) < 45
        and
        is_valid_number(technical_score)
        and float(technical_score) < 45
    ):
        return "NÃO COMPRAR"

    return "OBSERVAÇÃO"


# ============================================================
# CLASSIFICAÇÃO DO SINAL
# ============================================================

def classify_signal_strength(
    score: float,
) -> str:
    """
    Classifica a força global do sinal.
    """

    if not is_valid_number(score):
        return "INDEFINIDA"

    score = float(score)

    if score >= 85:
        return "MUITO FORTE"

    if score >= 75:
        return "FORTE"

    if score >= 65:
        return "MODERADA"

    if score >= 55:
        return "NEUTRA"

    if score >= 45:
        return "FRACA"

    return "MUITO FRACA"


# ============================================================
# FATORES POSITIVOS
# ============================================================

def list_signal_positive_factors(
    row: pd.Series,
) -> str:
    """
    Lista os principais fatores positivos.
    """

    factors: list[str] = []

    institutional_score = row.get(
        "institutional_score",
        0,
    )

    technical_score = row.get(
        "technical_entry_score",
        0,
    )

    money_flow_score = row.get(
        "score_money_flow",
        np.nan,
    )

    leadership_score = row.get(
        "score_market_leadership",
        np.nan,
    )

    sector_strength = row.get(
        "score_sector_strength",
        np.nan,
    )

    momentum_score = row.get(
        "score_momentum",
        np.nan,
    )

    trend_score = row.get(
        "score_trend",
        np.nan,
    )

    volume_score = row.get(
        "score_volume_flow",
        np.nan,
    )

    if (
        is_valid_number(
            institutional_score
        )
        and float(institutional_score)
        >= MIN_INSTITUTIONAL_SCORE
    ):
        factors.append(
            "fluxo institucional aprovado"
        )

    if (
        is_valid_number(
            technical_score
        )
        and float(technical_score)
        >= MIN_TECHNICAL_SCORE
    ):
        factors.append(
            "score técnico aprovado"
        )

    if (
        is_valid_number(
            money_flow_score
        )
        and float(money_flow_score)
        >= 70
    ):
        factors.append(
            "entrada de capital favorável"
        )

    if (
        is_valid_number(
            leadership_score
        )
        and float(leadership_score)
        >= 70
    ):
        factors.append(
            "liderança no setor"
        )

    if (
        is_valid_number(
            sector_strength
        )
        and float(sector_strength)
        >= 70
    ):
        factors.append(
            "setor forte"
        )

    if (
        is_valid_number(
            momentum_score
        )
        and float(momentum_score)
        >= 70
    ):
        factors.append(
            "momentum positivo"
        )

    if (
        is_valid_number(
            trend_score
        )
        and float(trend_score)
        >= 70
    ):
        factors.append(
            "tendência favorável"
        )

    if (
        is_valid_number(
            volume_score
        )
        and float(volume_score)
        >= 70
    ):
        factors.append(
            "volume técnico favorável"
        )

    if boolean_value(
        row.get(
            "macd_confirmation",
            False,
        )
    ):
        factors.append(
            "MACD confirmado"
        )

    if boolean_value(
        row.get(
            "trend_confirmation",
            False,
        )
    ):
        factors.append(
            "estrutura de tendência confirmada"
        )

    if boolean_value(
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


# ============================================================
# PENDÊNCIAS
# ============================================================

def list_signal_pending_conditions(
    row: pd.Series,
) -> str:
    """
    Lista o que impede a aprovação da entrada.
    """

    pending: list[str] = []

    institutional_score = row.get(
        "institutional_score",
        0,
    )

    technical_score = row.get(
        "technical_entry_score",
        0,
    )

    final_score = row.get(
        "final_score",
        0,
    )

    if (
        not is_valid_number(
            institutional_score
        )
        or float(institutional_score)
        < MIN_INSTITUTIONAL_SCORE
    ):
        pending.append(
            "Institutional Score abaixo do mínimo"
        )

    if not boolean_value(
        row.get(
            "institutional_flow_approved",
            False,
        )
    ):
        pending.append(
            "fluxo institucional não aprovado"
        )

    if (
        not is_valid_number(
            technical_score
        )
        or float(technical_score)
        < MIN_TECHNICAL_SCORE
    ):
        pending.append(
            "Technical Score abaixo do mínimo"
        )

    if not boolean_value(
        row.get(
            "technical_entry_approved",
            False,
        )
    ):
        pending.append(
            "entrada técnica ainda não confirmada"
        )

    if (
        not is_valid_number(
            final_score
        )
        or float(final_score)
        < MIN_FINAL_SCORE
    ):
        pending.append(
            "Final Score abaixo do mínimo"
        )

    technical_pending = row.get(
        "pending_technical_conditions",
        "",
    )

    if (
        isinstance(
            technical_pending,
            str,
        )
        and technical_pending.strip()
    ):
        pending.append(
            technical_pending.strip()
        )

    return "; ".join(
        dict.fromkeys(
            pending
        )
    )


# ============================================================
# MOTIVOS DE REJEIÇÃO
# ============================================================

def list_rejection_reasons(
    row: pd.Series,
) -> str:
    """
    Lista fatores graves que justificam evitar a compra.
    """

    reasons: list[str] = []

    institutional_penalty = row.get(
        "institutional_penalty",
        0,
    )

    technical_penalty = row.get(
        "technical_penalty",
        0,
    )

    institutional_penalty_reasons = row.get(
        "institutional_penalty_reasons",
        "",
    )

    technical_penalty_reasons = row.get(
        "technical_penalty_reasons",
        "",
    )

    if (
        is_valid_number(
            institutional_penalty
        )
        and float(institutional_penalty)
        >= 15
    ):
        reasons.append(
            "penalidade institucional elevada"
        )

    if (
        is_valid_number(
            technical_penalty
        )
        and float(technical_penalty)
        >= 15
    ):
        reasons.append(
            "penalidade técnica elevada"
        )

    if (
        isinstance(
            institutional_penalty_reasons,
            str,
        )
        and institutional_penalty_reasons.strip()
    ):
        reasons.append(
            institutional_penalty_reasons.strip()
        )

    if (
        isinstance(
            technical_penalty_reasons,
            str,
        )
        and technical_penalty_reasons.strip()
    ):
        reasons.append(
            technical_penalty_reasons.strip()
        )

    return "; ".join(
        dict.fromkeys(
            reasons
        )
    )


# ============================================================
# DECISÃO EXECUTIVA
# ============================================================

def generate_executive_decision(
    row: pd.Series,
) -> str:
    """
    Gera uma decisão curta e direta.
    """

    status = row.get(
        "signal_status",
        "OBSERVAÇÃO",
    )

    ticker = row.get(
        "ticker",
        "",
    )

    final_score = row.get(
        "final_score",
        0,
    )

    if status == "ENTRADA APROVADA":
        return (
            f"{ticker}: compra tecnicamente confirmada, "
            f"com fluxo institucional favorável. "
            f"Nota final {float(final_score):.2f}."
        )

    if status == "AGUARDAR GATILHO":
        return (
            f"{ticker}: empresa qualificada, mas ainda "
            f"aguarda confirmação de entrada."
        )

    if status == "AGUARDAR CONFIRMAÇÃO TÉCNICA":
        return (
            f"{ticker}: fluxo institucional aprovado; "
            f"aguardar confirmação completa da análise técnica."
        )

    if status == "TÉCNICA BOA — FLUXO INSUFICIENTE":
        return (
            f"{ticker}: configuração técnica favorável, "
            f"mas sem fluxo institucional suficiente."
        )

    if status == "OBSERVAÇÃO PRIORITÁRIA":
        return (
            f"{ticker}: permanece na lista prioritária "
            f"para acompanhamento."
        )

    if status == "NÃO COMPRAR":
        return (
            f"{ticker}: filtros institucional e técnico "
            f"insuficientes para entrada."
        )

    return (
        f"{ticker}: acompanhar, sem entrada confirmada."
    )


# ============================================================
# CLASSE PRINCIPAL
# ============================================================

class SignalEngine:
    """
    Motor responsável por combinar os scores e gerar sinais.
    """

    def __init__(
        self,
        output_file: str | Path = SIGNAL_OUTPUT_FILE,
    ) -> None:
        self.output_file = Path(
            output_file
        )

        self.result = pd.DataFrame()

    def calculate(
        self,
        institutional_data: pd.DataFrame,
        technical_data: pd.DataFrame,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Combina os dois motores e gera a decisão final.

        Parameters
        ----------
        institutional_data:
            Resultado produzido por institutional_score.py.

        technical_data:
            Resultado produzido por technical_score.py.

        save:
            Quando True, salva data/signals.csv.

        Returns
        -------
        pandas.DataFrame
            Resultado consolidado por ticker.
        """

        institutional = validate_dataframe(
            data=institutional_data,
            required_columns=(
                MINIMUM_INSTITUTIONAL_COLUMNS
            ),
            dataframe_name=(
                "institutional_data"
            ),
        )

        technical = validate_dataframe(
            data=technical_data,
            required_columns=(
                MINIMUM_TECHNICAL_COLUMNS
            ),
            dataframe_name=(
                "technical_data"
            ),
        )

        # ----------------------------------------------------
        # EVITAR COLUNAS DUPLICADAS
        # ----------------------------------------------------

        common_columns = (
            set(institutional.columns)
            .intersection(
                technical.columns
            )
            -
            {"ticker"}
        )

        technical_columns_to_keep = [
            column
            for column in technical.columns
            if (
                column == "ticker"
                or column not in common_columns
            )
        ]

        technical_clean = technical[
            technical_columns_to_keep
        ].copy()

        result = institutional.merge(
            technical_clean,
            on="ticker",
            how="inner",
            validate="one_to_one",
        )

        if result.empty:
            raise RuntimeError(
                "Nenhum ticker coincidente foi encontrado "
                "entre os motores."
            )

        # ----------------------------------------------------
        # SCORE FINAL
        # ----------------------------------------------------

        result[
            "final_score"
        ] = result.apply(
            calculate_combined_score,
            axis=1,
        ).round(2)

        result[
            "confluence_score"
        ] = result.apply(
            calculate_confluence_score,
            axis=1,
        ).round(2)

        result[
            "signal_strength_score"
        ] = result.apply(
            calculate_signal_strength,
            axis=1,
        ).round(2)

        # ----------------------------------------------------
        # CONDIÇÕES
        # ----------------------------------------------------

        result[
            "institutional_condition_approved"
        ] = result.apply(
            institutional_condition,
            axis=1,
        )

        result[
            "technical_condition_approved"
        ] = result.apply(
            technical_condition,
            axis=1,
        )

        result[
            "final_score_condition_approved"
        ] = result.apply(
            final_score_condition,
            axis=1,
        )

        result[
            "signal_approved"
        ] = result.apply(
            calculate_signal_approval,
            axis=1,
        )

        # ----------------------------------------------------
        # CLASSIFICAÇÃO
        # ----------------------------------------------------

        result[
            "signal_status"
        ] = result.apply(
            define_signal_status,
            axis=1,
        )

        result[
            "signal_strength"
        ] = result[
            "signal_strength_score"
        ].apply(
            classify_signal_strength
        )

        # ----------------------------------------------------
        # EXPLICAÇÕES
        # ----------------------------------------------------

        result[
            "signal_positive_factors"
        ] = result.apply(
            list_signal_positive_factors,
            axis=1,
        )

        result[
            "signal_pending_conditions"
        ] = result.apply(
            list_signal_pending_conditions,
            axis=1,
        )

        result[
            "signal_rejection_reasons"
        ] = result.apply(
            list_rejection_reasons,
            axis=1,
        )

        result[
            "executive_decision"
        ] = result.apply(
            generate_executive_decision,
            axis=1,
        )

        # ----------------------------------------------------
        # ORDEM DO STATUS
        # ----------------------------------------------------

        status_order = {
            "ENTRADA APROVADA": 1,
            "AGUARDAR GATILHO": 2,
            "AGUARDAR CONFIRMAÇÃO TÉCNICA": 3,
            "OBSERVAÇÃO PRIORITÁRIA": 4,
            "TÉCNICA BOA — FLUXO INSUFICIENTE": 5,
            "OBSERVAÇÃO": 6,
            "NÃO COMPRAR": 7,
        }

        result[
            "signal_status_order"
        ] = (
            result[
                "signal_status"
            ]
            .map(status_order)
            .fillna(99)
        )

        # ----------------------------------------------------
        # ORDENAÇÃO
        # ----------------------------------------------------

        result = (
            result
            .sort_values(
                [
                    "signal_approved",
                    "signal_status_order",
                    "signal_strength_score",
                    "final_score",
                    "institutional_score",
                    "technical_entry_score",
                ],
                ascending=[
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                ],
            )
            .reset_index(drop=True)
        )

        result[
            "signal_ranking"
        ] = np.arange(
            1,
            len(result) + 1,
        )

        self.result = result

        if save:
            self.save()

        self.print_summary()

        return self.result.copy()

    def save(self) -> None:
        """
        Salva os sinais consolidados.
        """

        if self.result.empty:
            raise ValueError(
                "Não há sinais para salvar."
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

    def get_result(self) -> pd.DataFrame:
        """
        Retorna o resultado completo.
        """

        return self.result.copy()

    def get_approved_signals(
        self,
    ) -> pd.DataFrame:
        """
        Retorna apenas entradas aprovadas.
        """

        if self.result.empty:
            return pd.DataFrame()

        return (
            self.result.loc[
                self.result[
                    "signal_approved"
                ]
            ]
            .copy()
            .reset_index(drop=True)
        )

    def get_watchlist(
        self,
    ) -> pd.DataFrame:
        """
        Retorna ações que ainda aguardam confirmação.
        """

        if self.result.empty:
            return pd.DataFrame()

        watch_status = {
            "AGUARDAR GATILHO",
            "AGUARDAR CONFIRMAÇÃO TÉCNICA",
            "OBSERVAÇÃO PRIORITÁRIA",
        }

        return (
            self.result.loc[
                self.result[
                    "signal_status"
                ].isin(
                    watch_status
                )
            ]
            .copy()
            .reset_index(drop=True)
        )

    def print_summary(self) -> None:
        """
        Exibe o resumo do motor.
        """

        total = len(
            self.result
        )

        approved = (
            int(
                self.result[
                    "signal_approved"
                ].sum()
            )
            if not self.result.empty
            else 0
        )

        print()
        print("=" * 115)
        print("SIGNAL ENGINE — DECISÃO FINAL")
        print("=" * 115)

        print(
            f"Empresas analisadas: {total}"
        )

        print(
            f"Entradas aprovadas: {approved}"
        )

        print(
            f"Institutional Score mínimo: "
            f"{MIN_INSTITUTIONAL_SCORE}"
        )

        print(
            f"Technical Score mínimo: "
            f"{MIN_TECHNICAL_SCORE}"
        )

        print(
            f"Final Score mínimo: "
            f"{MIN_FINAL_SCORE}"
        )

        print(
            f"Arquivo: {self.output_file}"
        )

        if not self.result.empty:

            print()
            print(
                "TOP 15 — SINAIS DO SCANNER"
            )

            columns = [
                "signal_ranking",
                "ticker",
                "setor",
                "signal_status",
                "signal_approved",
                "final_score",
                "signal_strength_score",
                "institutional_score",
                "technical_entry_score",
                "institutional_classification",
                "technical_classification",
                "executive_decision",
            ]

            available_columns = [
                column
                for column in columns
                if column in self.result.columns
            ]

            print(
                self.result[
                    available_columns
                ]
                .head(15)
                .to_string(
                    index=False
                )
            )

        print("=" * 115)


# ============================================================
# FUNÇÃO SIMPLIFICADA
# ============================================================

def generate_signals(
    institutional_data: pd.DataFrame,
    technical_data: pd.DataFrame,
    save: bool = True,
) -> pd.DataFrame:
    """
    Interface simplificada do Signal Engine.
    """

    engine = SignalEngine()

    return engine.calculate(
        institutional_data=institutional_data,
        technical_data=technical_data,
        save=save,
    )


# ============================================================
# EXECUÇÃO DIRETA
# ============================================================

if __name__ == "__main__":

    institutional_file = (
        Path(DATA_PATH)
        / "institutional_score.csv"
    )

    technical_file = (
        Path(DATA_PATH)
        / "technical_score.csv"
    )

    if not institutional_file.exists():
        raise FileNotFoundError(
            "institutional_score.csv não foi encontrado. "
            "Execute primeiro institutional_score.py."
        )

    if not technical_file.exists():
        raise FileNotFoundError(
            "technical_score.csv não foi encontrado. "
            "Execute primeiro technical_score.py."
        )

    institutional_data = pd.read_csv(
        institutional_file
    )

    technical_data = pd.read_csv(
        technical_file
    )

    signal_engine = SignalEngine()

    signals = signal_engine.calculate(
        institutional_data=institutional_data,
        technical_data=technical_data,
        save=True,
    )

    print()
    print("ENTRADAS APROVADAS:")

    approved = (
        signal_engine
        .get_approved_signals()
    )

    if approved.empty:

        print(
            "Nenhuma entrada aprovada neste pregão."
        )

    else:

        columns = [
            "ticker",
            "setor",
            "signal_status",
            "final_score",
            "institutional_score",
            "technical_entry_score",
            "signal_positive_factors",
            "executive_decision",
        ]

        available_columns = [
            column
            for column in columns
            if column in approved.columns
        ]

        print(
            approved[
                available_columns
            ]
            .to_string(
                index=False
            )
        )
