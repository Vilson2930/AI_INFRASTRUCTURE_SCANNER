# ============================================================
# AI INFRASTRUCTURE SCANNER
# app.py
#
# Interface Streamlit do scanner.
#
# Funções:
# - executar o scanner;
# - visualizar ranking;
# - visualizar entradas aprovadas;
# - visualizar watchlist;
# - visualizar melhor ação por setor;
# - filtrar por setor e status;
# - consultar detalhes de uma empresa;
# - baixar ranking e relatório Excel.
# ============================================================

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from config.settings import (
    PROJECT_NAME,
    VERSION,
    DESCRIPTION,
    DATA_PATH,
    RANKING_FILE,
    REPORT_FILE,
)

from main import run_scanner


# ============================================================
# ARQUIVOS
# ============================================================

SIGNAL_FILE = (
    Path(DATA_PATH)
    / "signals.csv"
)

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

INSTITUTIONAL_FILE = (
    Path(DATA_PATH)
    / "institutional_score.csv"
)

TECHNICAL_FILE = (
    Path(DATA_PATH)
    / "technical_score.csv"
)


# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title=PROJECT_NAME,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# ESTILO
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.25rem;
        font-weight: 800;
        margin-bottom: 0.15rem;
    }

    .subtitle {
        color: #666666;
        font-size: 1rem;
        margin-bottom: 1.4rem;
    }

    .status-approved {
        padding: 0.35rem 0.65rem;
        border-radius: 0.45rem;
        background-color: #DCFCE7;
        color: #166534;
        font-weight: 700;
        display: inline-block;
    }

    .status-wait {
        padding: 0.35rem 0.65rem;
        border-radius: 0.45rem;
        background-color: #FEF3C7;
        color: #92400E;
        font-weight: 700;
        display: inline-block;
    }

    .status-rejected {
        padding: 0.35rem 0.65rem;
        border-radius: 0.45rem;
        background-color: #FEE2E2;
        color: #991B1B;
        font-weight: 700;
        display: inline-block;
    }

    .small-note {
        font-size: 0.85rem;
        color: #6B7280;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

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


def load_csv(
    file_path: str | Path,
) -> pd.DataFrame:
    """
    Carrega um arquivo CSV caso exista.
    """

    path = Path(
        file_path
    )

    if not path.exists():
        return pd.DataFrame()

    try:

        df = pd.read_csv(
            path
        )

        df = normalize_columns(
            df
        )

        if "signal_approved" in df.columns:

            df["signal_approved"] = (
                df["signal_approved"]
                .apply(
                    boolean_value
                )
            )

        return df

    except Exception as error:

        st.error(
            f"Erro ao carregar {path.name}: {error}"
        )

        return pd.DataFrame()


def dataframe_to_csv_bytes(
    data: pd.DataFrame,
) -> bytes:
    """
    Converte DataFrame em CSV para download.
    """

    return data.to_csv(
        index=False,
        encoding="utf-8-sig",
    ).encode(
        "utf-8-sig"
    )


def status_label(
    status: str,
) -> str:
    """
    Retorna um selo visual conforme o status.
    """

    normalized = str(
        status
    ).upper()

    if "ENTRADA APROVADA" in normalized:

        return (
            '<span class="status-approved">'
            f"{status}"
            "</span>"
        )

    if (
        "AGUARDAR" in normalized
        or "OBSERVAÇÃO PRIORITÁRIA" in normalized
    ):

        return (
            '<span class="status-wait">'
            f"{status}"
            "</span>"
        )

    if "NÃO COMPRAR" in normalized:

        return (
            '<span class="status-rejected">'
            f"{status}"
            "</span>"
        )

    return (
        '<span class="status-wait">'
        f"{status}"
        "</span>"
    )


def safe_metric_value(
    data: pd.DataFrame,
    column: str,
    operation: str = "mean",
) -> float:
    """
    Calcula métricas com segurança.
    """

    if (
        data.empty
        or column not in data.columns
    ):
        return 0.0

    values = pd.to_numeric(
        data[column],
        errors="coerce",
    )

    if operation == "max":
        return float(
            values.max()
        )

    if operation == "sum":
        return float(
            values.sum()
        )

    return float(
        values.mean()
    )


def display_table(
    data: pd.DataFrame,
    preferred_columns: list[str],
    height: int = 520,
) -> None:
    """
    Exibe tabela com colunas prioritárias.
    """

    if data.empty:

        st.info(
            "Nenhum registro disponível."
        )

        return

    available_columns = [
        column
        for column in preferred_columns
        if column in data.columns
    ]

    remaining_columns = [
        column
        for column in data.columns
        if column not in available_columns
    ]

    table = data[
        available_columns
        +
        remaining_columns
    ].copy()

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        height=height,
    )


# ============================================================
# CARREGAMENTO DOS RESULTADOS
# ============================================================

@st.cache_data(
    show_spinner=False
)
def load_results() -> dict[str, pd.DataFrame]:
    """
    Carrega os arquivos produzidos pelo scanner.
    """

    return {
        "ranking":
            load_csv(
                RANKING_FILE
            ),

        "signals":
            load_csv(
                SIGNAL_FILE
            ),

        "approved":
            load_csv(
                APPROVED_ENTRIES_FILE
            ),

        "watchlist":
            load_csv(
                WATCHLIST_FILE
            ),

        "best_by_sector":
            load_csv(
                BEST_BY_SECTOR_FILE
            ),

        "institutional":
            load_csv(
                INSTITUTIONAL_FILE
            ),

        "technical":
            load_csv(
                TECHNICAL_FILE
            ),
    }


# ============================================================
# EXECUÇÃO DO SCANNER
# ============================================================

def execute_scanner() -> None:
    """
    Executa o pipeline e atualiza a interface.
    """

    try:

        with st.spinner(
            "Executando o scanner completo..."
        ):

            results = run_scanner()

        st.session_state[
            "scanner_results"
        ] = results

        load_results.clear()

        st.success(
            "Scanner executado com sucesso."
        )

        st.rerun()

    except Exception as error:

        st.error(
            f"Erro durante a execução: {error}"
        )

        st.exception(
            error
        )


# ============================================================
# CABEÇALHO
# ============================================================

st.markdown(
    f"""
    <div class="main-title">
        🤖 {PROJECT_NAME}
    </div>

    <div class="subtitle">
        {DESCRIPTION}<br>
        Versão {VERSION}
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# BARRA LATERAL
# ============================================================

st.sidebar.title(
    "Controle do scanner"
)

st.sidebar.caption(
    "Execute o motor para atualizar preços, indicadores e ranking."
)

run_button = st.sidebar.button(
    "▶ Executar scanner",
    type="primary",
    use_container_width=True,
)

if run_button:
    execute_scanner()


st.sidebar.divider()


page = st.sidebar.radio(
    "Navegação",
    options=[
        "Dashboard",
        "Ranking geral",
        "Entradas aprovadas",
        "Watchlist",
        "Melhor por setor",
        "Detalhes da empresa",
        "Arquivos",
    ],
)


st.sidebar.divider()

st.sidebar.markdown(
    """
    **Metodologia**

    O scanner combina:

    - fluxo institucional;
    - crescimento e momentum;
    - liderança no setor;
    - análise técnica;
    - tendência;
    - volume;
    - risco e volatilidade.
    """
)


# ============================================================
# DADOS
# ============================================================

results = load_results()

ranking = results[
    "ranking"
]

approved_entries = results[
    "approved"
]

watchlist = results[
    "watchlist"
]

best_by_sector = results[
    "best_by_sector"
]

institutional = results[
    "institutional"
]

technical = results[
    "technical"
]


if ranking.empty:

    st.warning(
        "O ranking ainda não foi criado. "
        "Clique em **Executar scanner** na barra lateral."
    )

    st.stop()


# ============================================================
# FILTROS GERAIS
# ============================================================

available_sectors = (
    sorted(
        ranking["setor"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    if "setor" in ranking.columns
    else []
)


available_status = (
    sorted(
        ranking[
            "signal_status"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    if "signal_status" in ranking.columns
    else []
)


selected_sectors = st.sidebar.multiselect(
    "Filtrar setores",
    options=available_sectors,
    default=[],
)


selected_status = st.sidebar.multiselect(
    "Filtrar status",
    options=available_status,
    default=[],
)


filtered_ranking = ranking.copy()


if (
    selected_sectors
    and "setor"
    in filtered_ranking.columns
):

    filtered_ranking = (
        filtered_ranking.loc[
            filtered_ranking[
                "setor"
            ].isin(
                selected_sectors
            )
        ]
    )


if (
    selected_status
    and "signal_status"
    in filtered_ranking.columns
):

    filtered_ranking = (
        filtered_ranking.loc[
            filtered_ranking[
                "signal_status"
            ].isin(
                selected_status
            )
        ]
    )


filtered_ranking = (
    filtered_ranking
    .reset_index(drop=True)
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.subheader(
        "Visão geral"
    )

    total_companies = len(
        ranking
    )

    approved_count = (
        int(
            ranking[
                "signal_approved"
            ].sum()
        )
        if "signal_approved"
        in ranking.columns
        else len(
            approved_entries
        )
    )

    watchlist_count = len(
        watchlist
    )

    sector_count = (
        ranking[
            "setor"
        ].nunique()
        if "setor"
        in ranking.columns
        else 0
    )

    average_final_score = (
        safe_metric_value(
            ranking,
            "final_score",
            "mean",
        )
    )

    columns = st.columns(
        5
    )

    columns[0].metric(
        "Empresas",
        total_companies,
    )

    columns[1].metric(
        "Entradas aprovadas",
        approved_count,
    )

    columns[2].metric(
        "Watchlist",
        watchlist_count,
    )

    columns[3].metric(
        "Setores",
        sector_count,
    )

    columns[4].metric(
        "Final Score médio",
        f"{average_final_score:.2f}",
    )


    st.divider()


    left_column, right_column = (
        st.columns(
            [1.25, 1]
        )
    )


    with left_column:

        st.markdown(
            "### Top oportunidades"
        )

        top_columns = [
            "ranking",
            "ticker",
            "company",
            "setor",
            "signal_status",
            "priority_score",
            "final_score",
            "institutional_score",
            "technical_entry_score",
        ]

        display_table(
            data=filtered_ranking.head(15),
            preferred_columns=top_columns,
            height=520,
        )


    with right_column:

        st.markdown(
            "### Final Score por empresa"
        )

        if (
            "ticker" in filtered_ranking.columns
            and "final_score"
            in filtered_ranking.columns
        ):

            chart_data = (
                filtered_ranking
                .head(15)
                .sort_values(
                    "final_score",
                    ascending=True,
                )
            )

            figure = px.bar(
                chart_data,
                x="final_score",
                y="ticker",
                orientation="h",
                hover_data=[
                    column
                    for column in [
                        "setor",
                        "signal_status",
                        "institutional_score",
                        "technical_entry_score",
                    ]
                    if column
                    in chart_data.columns
                ],
                labels={
                    "final_score":
                        "Final Score",

                    "ticker":
                        "Ticker",
                },
            )

            figure.update_layout(
                height=520,
                margin=dict(
                    l=20,
                    r=20,
                    t=20,
                    b=20,
                ),
            )

            st.plotly_chart(
                figure,
                use_container_width=True,
            )


    st.divider()


    st.markdown(
        "### Força dos setores"
    )

    if (
        not institutional.empty
        and
        "setor"
        in institutional.columns
        and
        "score_sector_strength"
        in institutional.columns
    ):

        sector_chart = (
            institutional
            .groupby(
                "setor",
                as_index=False,
            )
            .agg(
                score_sector_strength=(
                    "score_sector_strength",
                    "mean",
                ),
                institutional_score=(
                    "institutional_score",
                    "mean",
                ),
            )
            .sort_values(
                "score_sector_strength",
                ascending=False,
            )
        )

        figure_sector = px.bar(
            sector_chart,
            x="setor",
            y="score_sector_strength",
            hover_data=[
                "institutional_score"
            ],
            labels={
                "setor": "Setor",
                "score_sector_strength":
                    "Força do setor",
            },
        )

        figure_sector.update_layout(
            height=430,
            xaxis_tickangle=-35,
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20,
            ),
        )

        st.plotly_chart(
            figure_sector,
            use_container_width=True,
        )


# ============================================================
# RANKING GERAL
# ============================================================

elif page == "Ranking geral":

    st.subheader(
        "Ranking geral"
    )

    st.caption(
        "Classificação combinada de fluxo institucional "
        "e ponto de entrada técnico."
    )

    ranking_columns = [
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
        "institutional_score",
        "technical_entry_score",
        "executive_decision",
    ]

    display_table(
        data=filtered_ranking,
        preferred_columns=ranking_columns,
        height=650,
    )

    st.download_button(
        label="Baixar ranking filtrado em CSV",
        data=dataframe_to_csv_bytes(
            filtered_ranking
        ),
        file_name=(
            "ranking_filtrado.csv"
        ),
        mime="text/csv",
    )


# ============================================================
# ENTRADAS APROVADAS
# ============================================================

elif page == "Entradas aprovadas":

    st.subheader(
        "Entradas aprovadas"
    )

    if approved_entries.empty:

        st.info(
            "Nenhuma entrada foi aprovada no último pregão."
        )

    else:

        st.success(
            f"{len(approved_entries)} entrada(s) "
            "aprovada(s) pelo scanner."
        )

        approved_columns = [
            "ticker",
            "company",
            "setor",
            "signal_status",
            "ranking_quality",
            "priority_score",
            "final_score",
            "institutional_score",
            "technical_entry_score",
            "signal_positive_factors",
            "executive_decision",
        ]

        display_table(
            data=approved_entries,
            preferred_columns=approved_columns,
            height=600,
        )

        st.download_button(
            label="Baixar entradas aprovadas",
            data=dataframe_to_csv_bytes(
                approved_entries
            ),
            file_name=(
                "entradas_aprovadas.csv"
            ),
            mime="text/csv",
        )


# ============================================================
# WATCHLIST
# ============================================================

elif page == "Watchlist":

    st.subheader(
        "Watchlist"
    )

    st.caption(
        "Empresas qualificadas que ainda aguardam "
        "gatilho ou confirmação técnica."
    )

    watchlist_columns = [
        "ticker",
        "company",
        "setor",
        "signal_status",
        "priority_score",
        "final_score",
        "institutional_score",
        "technical_entry_score",
        "technical_diagnosis",
        "signal_pending_conditions",
        "executive_decision",
    ]

    display_table(
        data=watchlist,
        preferred_columns=watchlist_columns,
        height=620,
    )

    if not watchlist.empty:

        st.download_button(
            label="Baixar watchlist",
            data=dataframe_to_csv_bytes(
                watchlist
            ),
            file_name="watchlist.csv",
            mime="text/csv",
        )


# ============================================================
# MELHOR POR SETOR
# ============================================================

elif page == "Melhor por setor":

    st.subheader(
        "Melhor ação de cada setor"
    )

    sector_columns = [
        "ticker",
        "company",
        "setor",
        "signal_status",
        "ranking_quality",
        "opportunity_profile",
        "priority_score",
        "final_score",
        "institutional_score",
        "technical_entry_score",
        "executive_decision",
    ]

    display_table(
        data=best_by_sector,
        preferred_columns=sector_columns,
        height=650,
    )

    if (
        not best_by_sector.empty
        and
        "setor"
        in best_by_sector.columns
        and
        "final_score"
        in best_by_sector.columns
    ):

        figure = px.bar(
            best_by_sector.sort_values(
                "final_score",
                ascending=False,
            ),
            x="setor",
            y="final_score",
            hover_data=[
                column
                for column in [
                    "ticker",
                    "signal_status",
                    "institutional_score",
                    "technical_entry_score",
                ]
                if column
                in best_by_sector.columns
            ],
            labels={
                "setor": "Setor",
                "final_score":
                    "Final Score",
            },
        )

        figure.update_layout(
            height=460,
            xaxis_tickangle=-35,
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )


# ============================================================
# DETALHES DA EMPRESA
# ============================================================

elif page == "Detalhes da empresa":

    st.subheader(
        "Detalhes da empresa"
    )

    ticker_options = (
        ranking["ticker"]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

    selected_ticker = st.selectbox(
        "Selecione o ticker",
        options=ticker_options,
    )

    company_data = (
        ranking.loc[
            ranking[
                "ticker"
            ] == selected_ticker
        ]
    )

    if company_data.empty:

        st.warning(
            "Dados não encontrados."
        )

    else:

        row = company_data.iloc[0]

        status = row.get(
            "signal_status",
            "OBSERVAÇÃO",
        )

        st.markdown(
            status_label(
                status
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            f"## {selected_ticker}"
        )

        company_name = row.get(
            "company",
            ""
        )

        sector = row.get(
            "setor",
            ""
        )

        if company_name:

            st.write(
                f"**Empresa:** {company_name}"
            )

        if sector:

            st.write(
                f"**Setor:** {sector}"
            )


        metric_columns = st.columns(
            5
        )

        metric_columns[0].metric(
            "Final Score",
            f"{float(row.get('final_score', 0)):.2f}",
        )

        metric_columns[1].metric(
            "Institutional",
            f"{float(row.get('institutional_score', 0)):.2f}",
        )

        metric_columns[2].metric(
            "Technical",
            f"{float(row.get('technical_entry_score', 0)):.2f}",
        )

        metric_columns[3].metric(
            "Prioridade",
            f"{float(row.get('priority_score', 0)):.2f}",
        )

        metric_columns[4].metric(
            "Preço",
            (
                f"${float(row.get('close', 0)):,.2f}"
                if "close" in row.index
                else "-"
            ),
        )


        st.divider()


        left, right = st.columns(
            2
        )


        with left:

            st.markdown(
                "### Diagnóstico institucional"
            )

            st.write(
                row.get(
                    "institutional_diagnosis",
                    "Não disponível.",
                )
            )

            st.markdown(
                "### Diagnóstico técnico"
            )

            st.write(
                row.get(
                    "technical_diagnosis",
                    "Não disponível.",
                )
            )

            st.markdown(
                "### Decisão executiva"
            )

            st.info(
                row.get(
                    "executive_decision",
                    "Não disponível.",
                )
            )


        with right:

            st.markdown(
                "### Fatores positivos"
            )

            positive = row.get(
                "signal_positive_factors",
                "",
            )

            st.write(
                positive
                if positive
                else "Nenhum fator registrado."
            )

            st.markdown(
                "### Condições pendentes"
            )

            pending = row.get(
                "signal_pending_conditions",
                "",
            )

            st.write(
                pending
                if pending
                else "Nenhuma pendência registrada."
            )

            st.markdown(
                "### Motivos de rejeição"
            )

            rejection = row.get(
                "signal_rejection_reasons",
                "",
            )

            st.write(
                rejection
                if rejection
                else "Nenhum motivo grave registrado."
            )


        score_columns = [
            "score_money_flow",
            "score_market_leadership",
            "score_sector_strength",
            "score_growth_momentum",
            "score_discount",
            "score_momentum",
            "score_trend",
            "score_volume_flow",
            "score_risk",
        ]

        available_scores = [
            column
            for column in score_columns
            if column in row.index
        ]

        if available_scores:

            score_table = pd.DataFrame(
                {
                    "Componente":
                        available_scores,

                    "Score":
                        [
                            row.get(
                                column,
                                0,
                            )
                            for column
                            in available_scores
                        ],
                }
            )

            figure = px.bar(
                score_table,
                x="Componente",
                y="Score",
                labels={
                    "Componente":
                        "Componente",

                    "Score":
                        "Nota",
                },
            )

            figure.update_layout(
                height=440,
                xaxis_tickangle=-35,
            )

            st.plotly_chart(
                figure,
                use_container_width=True,
            )


# ============================================================
# ARQUIVOS
# ============================================================

elif page == "Arquivos":

    st.subheader(
        "Arquivos gerados"
    )

    files = [
        (
            "Ranking",
            Path(RANKING_FILE),
        ),
        (
            "Entradas aprovadas",
            APPROVED_ENTRIES_FILE,
        ),
        (
            "Watchlist",
            WATCHLIST_FILE,
        ),
        (
            "Melhor por setor",
            BEST_BY_SECTOR_FILE,
        ),
        (
            "Institutional Score",
            INSTITUTIONAL_FILE,
        ),
        (
            "Technical Score",
            TECHNICAL_FILE,
        ),
        (
            "Relatório Excel",
            Path(REPORT_FILE),
        ),
    ]

    for label, file_path in files:

        column_name, column_status, column_action = (
            st.columns(
                [2, 1, 1]
            )
        )

        with column_name:

            st.write(
                f"**{label}**"
            )

            st.caption(
                str(file_path)
            )

        with column_status:

            if file_path.exists():

                st.success(
                    "Disponível"
                )

            else:

                st.warning(
                    "Não criado"
                )

        with column_action:

            if file_path.exists():

                file_bytes = (
                    file_path.read_bytes()
                )

                mime = (
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    if file_path.suffix == ".xlsx"
                    else "text/csv"
                )

                st.download_button(
                    label="Baixar",
                    data=file_bytes,
                    file_name=file_path.name,
                    mime=mime,
                    key=(
                        f"download_{file_path.name}"
                    ),
                    use_container_width=True,
                )


# ============================================================
# RODAPÉ
# ============================================================

st.divider()

st.markdown(
    """
    <div class="small-note">
        Este scanner é uma ferramenta quantitativa de apoio.
        Os resultados não representam garantia de retorno nem
        recomendação individual de investimento.
    </div>
    """,
    unsafe_allow_html=True,
)
