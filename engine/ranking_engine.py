# ============================================================
# AI INFRASTRUCTURE SCANNER
# ranking_engine.py
#
# Motor responsável pelo ranking final do scanner.
#
# Objetivos:
# - ordenar as melhores oportunidades;
# - priorizar entradas aprovadas;
# - separar entrada imediata, pré-entrada, pullback e rompimento;
# - incorporar Entry Timing Score e risco/retorno;
# - penalizar oportunidades com relação risco/retorno insuficiente;
# - corrigir duplicidades de colunas;
# - selecionar a melhor ação de cada setor;
# - gerar ranking executivo;
# - salvar data/ranking.csv.
# ============================================================

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import (
    DATA_PATH,
    RANKING_FILE,
    TOP_N,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURAÇÕES
# ============================================================

SIGNAL_FILE = (
    Path(DATA_PATH)
    / "signals.csv"
)

BEST_BY_SECTOR_FILE = (
    Path(DATA_PATH)
    / "melhor_por_setor.csv"
)

WATCHLIST_FILE = (
    Path(DATA_PATH)
    / "watchlist.csv"
)

APPROVED_ENTRIES_FILE = (
    Path(DATA_PATH)
    / "entradas_aprovadas.csv"
)

PRE_ENTRY_FILE = (
    Path(DATA_PATH)
    / "pre_entradas.csv"
)

PULLBACK_FILE = (
    Path(DATA_PATH)
    / "aguardar_pullback.csv"
)

BREAKOUT_FILE = (
    Path(DATA_PATH)
    / "aguardar_rompimento.csv"
)


REQUIRED_COLUMNS = {
    "ticker",
    "signal_status",
    "signal_approved",
    "final_score",
    "signal_strength_score",
    "institutional_score",
    "technical_entry_score",
    "entry_timing_score",
    "timing_status",
    "pullback_probability",
    "parabolic_risk",
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


def boolean_value(
    value: Any,
) -> bool:
    """
    Converte diferentes formatos em booleano.
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
    Padroniza nomes de colunas sem criar duplicidades.

    Quando a coluna canônica já existe, ela é preservada e a
    equivalente alternativa é removida.
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

    # Remove duplicidades que já possam existir.
    df = df.loc[
        :,
        ~df.columns.duplicated(
            keep="last"
        )
    ].copy()

    aliases = {
        "segmento":
            "setor",

        "empresa":
            "company",

        "nome_empresa":
            "company",

        "ranking_sinal":
            "signal_ranking",

        "score_final":
            "final_score",

        "status_sinal":
            "signal_status",

        "entrada_aprovada":
            "signal_approved",
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

    # Proteção final contra nomes duplicados.
    df = df.loc[
        :,
        ~df.columns.duplicated(
            keep="last"
        )
    ].copy()

    return df


def validate_input(
    signals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Valida o resultado produzido pelo Signal Engine.
    """

    if not isinstance(
        signals,
        pd.DataFrame,
    ):
        raise TypeError(
            "Os sinais devem ser fornecidos "
            "em um pandas DataFrame."
        )

    if signals.empty:
        raise ValueError(
            "O DataFrame de sinais está vazio."
        )

    df = normalize_columns(
        signals
    )

    missing_columns = (
        REQUIRED_COLUMNS
        .difference(
            df.columns
        )
    )

    if missing_columns:
        raise KeyError(
            "Colunas obrigatórias ausentes: "
            f"{sorted(missing_columns)}"
        )

    df["ticker"] = (
        df["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    numeric_columns = [
        "final_score",
        "signal_strength_score",
        "institutional_score",
        "technical_entry_score",
        "entry_timing_score",
        "pullback_probability",
        "timing_confidence",
        "entry_probability",
        "estimated_upside_percent",
        "risk_reward_ratio",
    ]

    optional_numeric_columns = [
        "confluence_score",
        "score_money_flow",
        "score_market_leadership",
        "score_sector_strength",
        "score_growth_momentum",
        "score_discount",
        "score_momentum",
        "score_trend",
        "score_volume_flow",
        "score_risk",
        "technical_penalty",
        "institutional_penalty",
        "close",
        "rsi_14",
        "adx_14",
        "atr_percentual",
        "distancia_maxima_20d",
        "distancia_maxima_52s",
        "distancia_sma_20",
        "retorno_5d",
        "retorno_10d",
        "volume_relativo_5d",
        "volume_relativo_20d",
    ]

    for column in (
        numeric_columns
        +
        optional_numeric_columns
    ):

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    df["signal_approved"] = (
        df["signal_approved"]
        .apply(
            boolean_value
        )
    )

    if "timing_approved" in df.columns:
        df["timing_approved"] = (
            df["timing_approved"]
            .apply(
                boolean_value
            )
        )

    if "timing_veto" in df.columns:
        df["timing_veto"] = (
            df["timing_veto"]
            .apply(
                boolean_value
            )
        )

    df = (
        df.dropna(
            subset=[
                "ticker",
                "final_score",
                "institutional_score",
                "technical_entry_score",
            ]
        )
        .drop_duplicates(
            subset=["ticker"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError(
            "Nenhum sinal válido permaneceu "
            "após a validação."
        )

    return df


# ============================================================
# PRIORIDADE DO STATUS
# ============================================================

def signal_status_priority(
    status: str,
) -> int:
    """
    Define a prioridade operacional de cada status.
    """

    status_map = {
        "ENTRADA APROVADA": 1,
        "PRÉ-ENTRADA — AGUARDAR GATILHO": 2,
        "AGUARDAR PULLBACK": 3,
        "AGUARDAR ROMPIMENTO": 4,
        "OBSERVAÇÃO PRIORITÁRIA": 5,
        "TÉCNICA BOA — FLUXO INSUFICIENTE": 6,
        "OBSERVAÇÃO": 7,
        "NÃO COMPRAR": 8,
        "MUITO ESTICADA — NÃO PERSEGUIR": 9,
    }

    return status_map.get(
        str(status).strip(),
        99,
    )


# ============================================================
# SCORE DE PRIORIDADE
# ============================================================

def calculate_priority_score(
    row: pd.Series,
) -> float:
    """
    Calcula a prioridade operacional com três motores.

    Pesos:
    - Final Score: 25%
    - Signal Strength: 20%
    - Institutional Score: 15%
    - Technical Score: 15%
    - Entry Timing Score: 20%
    - Confluence Score: 5%

    Ajustes:
    - bônus para entrada aprovada;
    - bônus menor para pré-entrada;
    - penalidade para risco/retorno insuficiente;
    - penalidade para risco de pullback e movimento parabólico.
    """

    components = [
        (
            row.get(
                "final_score",
                0,
            ),
            0.25,
        ),
        (
            row.get(
                "signal_strength_score",
                0,
            ),
            0.20,
        ),
        (
            row.get(
                "institutional_score",
                0,
            ),
            0.15,
        ),
        (
            row.get(
                "technical_entry_score",
                0,
            ),
            0.15,
        ),
        (
            row.get(
                "entry_timing_score",
                0,
            ),
            0.20,
        ),
        (
            row.get(
                "confluence_score",
                50,
            ),
            0.05,
        ),
    ]

    weighted_sum = 0.0
    total_weight = 0.0

    for value, weight in components:

        if is_valid_number(
            value
        ):

            weighted_sum += (
                float(value)
                *
                float(weight)
            )

            total_weight += float(
                weight
            )

    if total_weight == 0:
        return 0.0

    score = (
        weighted_sum
        /
        total_weight
    )

    status = str(
        row.get(
            "signal_status",
            "",
        )
    ).strip()

    if boolean_value(
        row.get(
            "signal_approved",
            False,
        )
    ):
        score += 5

    elif status == "PRÉ-ENTRADA — AGUARDAR GATILHO":
        score += 2

    risk_reward = row.get(
        "risk_reward_ratio",
        np.nan,
    )

    if is_valid_number(
        risk_reward
    ):

        risk_reward = float(
            risk_reward
        )

        if risk_reward < 0.75:
            score -= 12

        elif risk_reward < 1.00:
            score -= 8

        elif risk_reward < 1.20:
            score -= 3

        elif risk_reward >= 1.50:
            score += 3

    pullback_probability = row.get(
        "pullback_probability",
        np.nan,
    )

    if is_valid_number(
        pullback_probability
    ):

        pullback_probability = float(
            pullback_probability
        )

        if pullback_probability >= 80:
            score -= 20

        elif pullback_probability >= 65:
            score -= 12

        elif pullback_probability >= 50:
            score -= 5

    parabolic_risk = str(
        row.get(
            "parabolic_risk",
            "",
        )
    ).upper()

    if parabolic_risk == "EXTREMO":
        score -= 20

    elif parabolic_risk == "ALTO":
        score -= 12

    if boolean_value(
        row.get(
            "timing_veto",
            False,
        )
    ):
        score -= 15

    return float(
        np.clip(
            score,
            0,
            100,
        )
    )


# ============================================================
# QUALIDADE DO RANKING
# ============================================================

def classify_ranking_quality(
    row: pd.Series,
) -> str:
    """
    Classifica a qualidade geral da oportunidade.
    """

    priority_score = row.get(
        "priority_score",
        0,
    )

    approved = boolean_value(
        row.get(
            "signal_approved",
            False,
        )
    )

    risk_reward = row.get(
        "risk_reward_ratio",
        np.nan,
    )

    risk_reward_ok = bool(
        not is_valid_number(
            risk_reward
        )
        or float(
            risk_reward
        ) >= 1.0
    )

    if (
        approved
        and priority_score >= 85
        and risk_reward_ok
    ):
        return "OPORTUNIDADE EXCEPCIONAL"

    if (
        approved
        and priority_score >= 75
        and risk_reward_ok
    ):
        return "OPORTUNIDADE FORTE"

    if approved and not risk_reward_ok:
        return "ENTRADA APROVADA — RISCO/RETORNO FRACO"

    if approved:
        return "ENTRADA APROVADA"

    if priority_score >= 75:
        return "ALTA PRIORIDADE"

    if priority_score >= 65:
        return "PRIORIDADE MODERADA"

    if priority_score >= 55:
        return "ACOMPANHAMENTO"

    return "BAIXA PRIORIDADE"


# ============================================================
# PERFIL DA OPORTUNIDADE
# ============================================================

def define_opportunity_profile(
    row: pd.Series,
) -> str:
    """
    Resume a característica predominante da oportunidade.
    """

    institutional_score = row.get(
        "institutional_score",
        0,
    )

    technical_score = row.get(
        "technical_entry_score",
        0,
    )

    money_flow = row.get(
        "score_money_flow",
        np.nan,
    )

    discount_score = row.get(
        "score_discount",
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

    timing_status = str(
        row.get(
            "timing_status",
            "",
        )
    ).upper()

    if timing_status == "AGUARDAR PULLBACK":
        return "EMPRESA FORTE — AGUARDAR PULLBACK"

    if timing_status == "AGUARDAR ROMPIMENTO":
        return "AGUARDAR ROMPIMENTO COM VOLUME"

    if timing_status == "PRÉ-ENTRADA":
        return "PRÉ-ENTRADA COM GATILHO PENDENTE"

    if timing_status == "MUITO ESTICADA":
        return "MOVIMENTO ESTICADO — NÃO PERSEGUIR"

    if (
        is_valid_number(money_flow)
        and float(money_flow) >= 80
        and is_valid_number(technical_score)
        and float(technical_score) >= 70
    ):
        return "FLUXO FORTE COM ENTRADA TÉCNICA"

    if (
        is_valid_number(discount_score)
        and float(discount_score) >= 80
        and is_valid_number(momentum_score)
        and float(momentum_score) < 65
    ):
        return "AÇÃO DESCONTADA AGUARDANDO REVERSÃO"

    if (
        is_valid_number(momentum_score)
        and float(momentum_score) >= 75
        and is_valid_number(trend_score)
        and float(trend_score) >= 75
    ):
        return "LIDERANÇA DE MOMENTUM"

    if (
        is_valid_number(institutional_score)
        and float(institutional_score) >= 70
        and is_valid_number(technical_score)
        and float(technical_score) < 65
    ):
        return "FLUXO INSTITUCIONAL SEM TIMING"

    if (
        is_valid_number(technical_score)
        and float(technical_score) >= 70
        and is_valid_number(institutional_score)
        and float(institutional_score) < 65
    ):
        return "TIMING TÉCNICO SEM FLUXO"

    return "CONFIGURAÇÃO MISTA"


# ============================================================
# COLUNAS EXECUTIVAS
# ============================================================

def build_executive_table(
    ranking: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cria uma tabela enxuta para relatório e Streamlit.
    """

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
        "entry_timing_score",
        "timing_status",
        "timing_approved",
        "pullback_probability",
        "parabolic_risk",
        "timing_confidence",
        "entry_probability",
        "estimated_upside_percent",
        "risk_reward_ratio",
        "institutional_classification",
        "technical_classification",
        "institutional_diagnosis",
        "technical_diagnosis",
        "close",
        "rsi_14",
        "adx_14",
        "atr_percentual",
        "distancia_maxima_52s",
        "volume_relativo_20d",
        "signal_positive_factors",
        "signal_pending_conditions",
        "signal_rejection_reasons",
        "executive_decision",
    ]

    available_columns = [
        column
        for column in desired_columns
        if column in ranking.columns
    ]

    return ranking[
        available_columns
    ].copy()


# ============================================================
# CLASSE PRINCIPAL
# ============================================================

class RankingEngine:
    """
    Motor responsável pelo ranking final.
    """

    def __init__(
        self,
        output_file: str | Path = RANKING_FILE,
        top_n: int = TOP_N,
    ) -> None:
        self.output_file = Path(
            output_file
        )

        self.top_n = int(
            top_n
        )

        self.ranking = pd.DataFrame()

        self.executive_ranking = pd.DataFrame()

        self.best_by_sector = pd.DataFrame()

        self.approved_entries = pd.DataFrame()

        self.watchlist = pd.DataFrame()

        self.pre_entries = pd.DataFrame()

        self.pullback_watchlist = pd.DataFrame()

        self.breakout_watchlist = pd.DataFrame()

    def calculate(
        self,
        signals: pd.DataFrame,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Gera o ranking final.

        Parameters
        ----------
        signals:
            DataFrame produzido por signal_engine.py.

        save:
            Quando True, salva todos os arquivos do ranking.

        Returns
        -------
        pandas.DataFrame
            Ranking executivo final.
        """

        df = validate_input(
            signals
        )

        # ----------------------------------------------------
        # SCORES E CLASSIFICAÇÕES
        # ----------------------------------------------------

        df[
            "signal_status_priority"
        ] = df[
            "signal_status"
        ].apply(
            signal_status_priority
        )

        df[
            "priority_score"
        ] = df.apply(
            calculate_priority_score,
            axis=1,
        ).round(2)

        df[
            "ranking_quality"
        ] = df.apply(
            classify_ranking_quality,
            axis=1,
        )

        df[
            "opportunity_profile"
        ] = df.apply(
            define_opportunity_profile,
            axis=1,
        )

        # ----------------------------------------------------
        # ORDENAÇÃO FINAL
        # ----------------------------------------------------

        df = (
            df
            .sort_values(
                [
                    "signal_approved",
                    "signal_status_priority",
                    "priority_score",
                    "final_score",
                    "entry_timing_score",
                    "risk_reward_ratio",
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
                    False,
                    False,
                ],
            )
            .reset_index(drop=True)
        )

        df[
            "ranking"
        ] = np.arange(
            1,
            len(df) + 1,
        )

        self.ranking = df

        # ----------------------------------------------------
        # TABELA EXECUTIVA
        # ----------------------------------------------------

        self.executive_ranking = (
            build_executive_table(
                self.ranking
            )
        )

        # ----------------------------------------------------
        # ENTRADAS APROVADAS
        # ----------------------------------------------------

        self.approved_entries = (
            self.ranking.loc[
                self.ranking[
                    "signal_approved"
                ]
            ]
            .copy()
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # WATCHLIST GERAL
        # ----------------------------------------------------

        watchlist_status = {
            "PRÉ-ENTRADA — AGUARDAR GATILHO",
            "AGUARDAR PULLBACK",
            "AGUARDAR ROMPIMENTO",
            "OBSERVAÇÃO PRIORITÁRIA",
            "MUITO ESTICADA — NÃO PERSEGUIR",
        }

        self.watchlist = (
            self.ranking.loc[
                self.ranking[
                    "signal_status"
                ].isin(
                    watchlist_status
                )
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        self.pre_entries = (
            self.ranking.loc[
                self.ranking[
                    "signal_status"
                ]
                ==
                "PRÉ-ENTRADA — AGUARDAR GATILHO"
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        self.pullback_watchlist = (
            self.ranking.loc[
                self.ranking[
                    "signal_status"
                ].isin(
                    {
                        "AGUARDAR PULLBACK",
                        "MUITO ESTICADA — NÃO PERSEGUIR",
                    }
                )
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        self.breakout_watchlist = (
            self.ranking.loc[
                self.ranking[
                    "signal_status"
                ]
                ==
                "AGUARDAR ROMPIMENTO"
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        # ----------------------------------------------------
        # MELHOR POR SETOR
        # ----------------------------------------------------

        if "setor" in self.ranking.columns:

            self.best_by_sector = (
                self.ranking
                .sort_values(
                    [
                        "signal_approved",
                        "signal_status_priority",
                        "priority_score",
                        "final_score",
                        "entry_timing_score",
                    ],
                    ascending=[
                        False,
                        True,
                        False,
                        False,
                        False,
                    ],
                )
                .groupby(
                    "setor",
                    as_index=False,
                )
                .first()
                .sort_values(
                    [
                        "signal_approved",
                        "priority_score",
                    ],
                    ascending=[
                        False,
                        False,
                    ],
                )
                .reset_index(drop=True)
            )

        else:

            self.best_by_sector = (
                pd.DataFrame()
            )

        if save:
            self.save()

        self.print_summary()

        return self.executive_ranking.copy()

    def save(self) -> None:
        """
        Salva os arquivos gerados pelo ranking.
        """

        if self.ranking.empty:
            raise ValueError(
                "Não há ranking para salvar."
            )

        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.executive_ranking.to_csv(
            self.output_file,
            index=False,
            encoding="utf-8-sig",
        )

        self.approved_entries.to_csv(
            APPROVED_ENTRIES_FILE,
            index=False,
            encoding="utf-8-sig",
        )

        self.watchlist.to_csv(
            WATCHLIST_FILE,
            index=False,
            encoding="utf-8-sig",
        )

        self.pre_entries.to_csv(
            PRE_ENTRY_FILE,
            index=False,
            encoding="utf-8-sig",
        )

        self.pullback_watchlist.to_csv(
            PULLBACK_FILE,
            index=False,
            encoding="utf-8-sig",
        )

        self.breakout_watchlist.to_csv(
            BREAKOUT_FILE,
            index=False,
            encoding="utf-8-sig",
        )

        self.best_by_sector.to_csv(
            BEST_BY_SECTOR_FILE,
            index=False,
            encoding="utf-8-sig",
        )

    def get_ranking(
        self,
        top_n: int | None = None,
    ) -> pd.DataFrame:
        """
        Retorna o ranking executivo.
        """

        if self.executive_ranking.empty:
            return pd.DataFrame()

        limit = (
            self.top_n
            if top_n is None
            else int(top_n)
        )

        return (
            self.executive_ranking
            .head(limit)
            .copy()
            .reset_index(drop=True)
        )

    def get_full_ranking(
        self,
    ) -> pd.DataFrame:
        """
        Retorna o ranking completo.
        """

        return self.ranking.copy()

    def get_best_by_sector(
        self,
    ) -> pd.DataFrame:
        """
        Retorna a melhor ação de cada setor.
        """

        return self.best_by_sector.copy()

    def get_approved_entries(
        self,
    ) -> pd.DataFrame:
        """
        Retorna apenas entradas aprovadas.
        """

        return self.approved_entries.copy()

    def get_watchlist(
        self,
    ) -> pd.DataFrame:
        """
        Retorna a watchlist geral.
        """

        return self.watchlist.copy()

    def get_pre_entries(
        self,
    ) -> pd.DataFrame:
        """
        Retorna as pré-entradas.
        """

        return self.pre_entries.copy()

    def get_pullback_watchlist(
        self,
    ) -> pd.DataFrame:
        """
        Retorna ações que aguardam pullback.
        """

        return self.pullback_watchlist.copy()

    def get_breakout_watchlist(
        self,
    ) -> pd.DataFrame:
        """
        Retorna ações que aguardam rompimento.
        """

        return self.breakout_watchlist.copy()

    def print_summary(self) -> None:
        """
        Exibe resumo do ranking.
        """

        total = len(
            self.ranking
        )

        approved = len(
            self.approved_entries
        )

        watchlist_count = len(
            self.watchlist
        )

        pre_entry_count = len(
            self.pre_entries
        )

        pullback_count = len(
            self.pullback_watchlist
        )

        breakout_count = len(
            self.breakout_watchlist
        )

        sectors = (
            self.ranking[
                "setor"
            ].nunique()
            if (
                not self.ranking.empty
                and
                "setor"
                in self.ranking.columns
            )
            else 0
        )

        print()
        print("=" * 120)
        print("RANKING ENGINE — RESULTADO FINAL")
        print("=" * 120)

        print(
            f"Empresas classificadas: {total}"
        )

        print(
            f"Entradas aprovadas: {approved}"
        )

        print(
            f"Ações na watchlist: {watchlist_count}"
        )

        print(
            f"Pré-entradas: {pre_entry_count}"
        )

        print(
            f"Aguardar pullback: {pullback_count}"
        )

        print(
            f"Aguardar rompimento: {breakout_count}"
        )

        print(
            f"Setores analisados: {sectors}"
        )

        print(
            f"Arquivo principal: {self.output_file}"
        )

        if not self.executive_ranking.empty:

            print()
            print(
                f"TOP {min(self.top_n, total)} "
                "— MELHORES OPORTUNIDADES"
            )

            print(
                self.get_ranking()
                .to_string(
                    index=False
                )
            )

        print("=" * 120)


# ============================================================
# FUNÇÃO SIMPLIFICADA
# ============================================================

def generate_ranking(
    signals: pd.DataFrame,
    save: bool = True,
    top_n: int = TOP_N,
) -> pd.DataFrame:
    """
    Interface simplificada do Ranking Engine.
    """

    engine = RankingEngine(
        top_n=top_n
    )

    return engine.calculate(
        signals=signals,
        save=save,
    )


# ============================================================
# EXECUÇÃO DIRETA
# ============================================================

if __name__ == "__main__":

    if not SIGNAL_FILE.exists():

        raise FileNotFoundError(
            "O arquivo signals.csv não foi encontrado. "
            "Execute primeiro signal_engine.py."
        )

    signal_data = pd.read_csv(
        SIGNAL_FILE
    )

    ranking_engine = RankingEngine(
        top_n=TOP_N
    )

    ranking = ranking_engine.calculate(
        signals=signal_data,
        save=True,
    )

    print()
    print("MELHOR AÇÃO DE CADA SETOR:")

    best_by_sector = (
        ranking_engine
        .get_best_by_sector()
    )

    if best_by_sector.empty:

        print(
            "Nenhum setor disponível."
        )

    else:

        columns = [
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
            "risk_reward_ratio",
            "executive_decision",
        ]

        available_columns = [
            column
            for column in columns
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
