# ============================================================
# pdf_report.py
# AI INFRASTRUCTURE SCANNER
# Institutional Opportunity Report
# ============================================================

from pathlib import Path
from datetime import datetime
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
)


# ============================================================
# DIRETÓRIOS
# ============================================================

DATA_DIR = Path("data")
REPORTS_DIR = Path("reports")

REPORTS_DIR.mkdir(exist_ok=True)

PDF_FILE = (
    REPORTS_DIR
    / "relatorio_ai_infrastructure_scanner.pdf"
)


# ============================================================
# UTILIDADES
# ============================================================

def carregar_csv(nome):

    caminho = DATA_DIR / nome

    if not caminho.exists():
        print(f"[AVISO] Arquivo não encontrado: {caminho}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(caminho)
        print(f"[OK] {nome}: {len(df)} registros")
        return df

    except Exception as e:
        print(f"[ERRO] Falha ao carregar {nome}: {e}")
        return pd.DataFrame()


def localizar_coluna(df, candidatos):

    if df is None or df.empty:
        return None

    mapa = {
        str(c).lower().strip(): c
        for c in df.columns
    }

    for candidato in candidatos:

        chave = candidato.lower().strip()

        if chave in mapa:
            return mapa[chave]

    return None


def obter(row, candidatos, padrao="N/A"):

    for coluna in candidatos:

        if coluna in row.index:

            valor = row.get(coluna)

            if pd.notna(valor):
                return valor

    return padrao


def numero(valor, casas=2):

    try:
        return f"{float(valor):.{casas}f}"

    except Exception:
        return "-"


def percentual(valor):

    try:

        v = float(valor)

        # Se vier como decimal
        if abs(v) <= 1:
            v *= 100

        return f"{v:.2f}%"

    except Exception:
        return "-"


def texto_seguro(valor):

    if valor is None:
        return "-"

    try:
        if pd.isna(valor):
            return "-"
    except Exception:
        pass

    return str(valor)


# ============================================================
# ESTILOS
# ============================================================

styles = getSampleStyleSheet()

STYLE_TITLE = ParagraphStyle(
    "TitleInstitutional",
    parent=styles["Title"],
    fontName="Helvetica-Bold",
    fontSize=25,
    leading=30,
    alignment=TA_CENTER,
    spaceAfter=18,
)

STYLE_SUBTITLE = ParagraphStyle(
    "SubtitleInstitutional",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=13,
    leading=18,
    alignment=TA_CENTER,
    textColor=colors.HexColor("#555555"),
)

STYLE_SECTION = ParagraphStyle(
    "Section",
    parent=styles["Heading1"],
    fontName="Helvetica-Bold",
    fontSize=16,
    leading=20,
    alignment=TA_LEFT,
    spaceBefore=10,
    spaceAfter=12,
)

STYLE_SUBSECTION = ParagraphStyle(
    "Subsection",
    parent=styles["Heading2"],
    fontName="Helvetica-Bold",
    fontSize=12,
    leading=16,
    spaceBefore=8,
    spaceAfter=8,
)

STYLE_BODY = ParagraphStyle(
    "BodyInstitutional",
    parent=styles["BodyText"],
    fontName="Helvetica",
    fontSize=9,
    leading=14,
    spaceAfter=7,
)

STYLE_SMALL = ParagraphStyle(
    "SmallInstitutional",
    parent=styles["BodyText"],
    fontName="Helvetica",
    fontSize=7.5,
    leading=10,
)

STYLE_CARD_NUMBER = ParagraphStyle(
    "CardNumber",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=18,
    alignment=TA_CENTER,
)

STYLE_CARD_TEXT = ParagraphStyle(
    "CardText",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=8,
    alignment=TA_CENTER,
)


# ============================================================
# RODAPÉ
# ============================================================

def cabecalho_rodape(canvas, doc):

    canvas.saveState()

    largura, altura = landscape(A4)

    canvas.setFont(
        "Helvetica",
        7,
    )

    canvas.setFillColor(
        colors.HexColor("#666666")
    )

    canvas.drawString(
        1.2 * cm,
        0.7 * cm,
        "AI INFRASTRUCTURE SCANNER | Institutional Opportunity Report",
    )

    canvas.drawRightString(
        largura - 1.2 * cm,
        0.7 * cm,
        f"Página {doc.page}",
    )

    canvas.restoreState()


# ============================================================
# TABELA PADRÃO
# ============================================================

def criar_tabela(dados, larguras=None, fonte=7):

    tabela = Table(
        dados,
        colWidths=larguras,
        repeatRows=1,
    )

    tabela.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#1F2937"),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "FONTNAME",
                    (0, 1),
                    (-1, -1),
                    "Helvetica",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    fonte,
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER",
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.25,
                    colors.HexColor("#CCCCCC"),
                ),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.white,
                        colors.HexColor("#F3F4F6"),
                    ],
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
            ]
        )
    )

    return tabela


# ============================================================
# CARDS
# ============================================================

def criar_cards(metricas):

    dados = []

    numeros = []
    textos = []

    for titulo, valor in metricas:

        numeros.append(
            Paragraph(
                texto_seguro(valor),
                STYLE_CARD_NUMBER,
            )
        )

        textos.append(
            Paragraph(
                titulo,
                STYLE_CARD_TEXT,
            )
        )

    dados.append(numeros)
    dados.append(textos)

    largura_total = 26 * cm

    tabela = Table(
        dados,
        colWidths=[
            largura_total / len(metricas)
            for _ in metricas
        ],
    )

    tabela.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor("#F3F4F6"),
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.6,
                    colors.HexColor("#D1D5DB"),
                ),
                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.HexColor("#D1D5DB"),
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    10,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    10,
                ),
            ]
        )
    )

    return tabela


# ============================================================
# CLASSIFICAÇÃO DOS SINAIS
# ============================================================

def classificar_sinal(row):
    """
    Converte a classificação oficial do Signal Engine
    para os grupos operacionais exibidos no PDF.

    A fonte principal é a coluna `signal_status`, criada
    pelo engine/signal_engine.py.
    """

    candidatos = [
        "signal_status",
        "sinal",
        "signal",
        "classificacao",
        "classification",
        "status",
        "entry_signal",
        "decisao",
    ]

    valor = obter(
        row,
        candidatos,
        "",
    )

    texto = str(valor).upper().strip()

    if not texto:
        return "SEM CLASSIFICAÇÃO"

    if texto == "ENTRADA FORTE":
        return "ENTRADA FORTE"

    if texto == "ENTRADA APROVADA":
        return "ENTRADA APROVADA"

    if "AGUARDAR VOLUME" in texto:
        return "AGUARDAR VOLUME"

    if "AGUARDAR GATILHO" in texto:
        return "AGUARDAR GATILHO"

    if "MELHORAR RISCO/RETORNO" in texto:
        return "PRÉ-ENTRADA"

    if texto.startswith("PRÉ-ENTRADA") or texto.startswith("PRE-ENTRADA"):
        return "PRÉ-ENTRADA"

    if "PULLBACK" in texto:
        return "AGUARDAR PULLBACK"

    if "ROMPIMENTO" in texto:
        return "AGUARDAR ROMPIMENTO"

    if "AGUARDAR MELHOR RISCO/RETORNO" in texto:
        return "AGUARDAR MELHOR RISCO/RETORNO"

    if texto in {
        "OBSERVAÇÃO PRIORITÁRIA",
        "OBSERVAÇÃO",
        "TÉCNICA BOA — FLUXO INSUFICIENTE",
        "NÃO COMPRAR",
        "MUITO ESTICADA — NÃO PERSEGUIR",
    }:
        return texto

    return texto


# ============================================================
# TABELA DE OPORTUNIDADES
# ============================================================

def tabela_oportunidades(df, limite=20):

    if df is None or df.empty:

        return Paragraph(
            "Nenhuma oportunidade disponível.",
            STYLE_BODY,
        )

    linhas = [
        [
            "Rank",
            "Ticker",
            "Sinal",
            "Score Final",
            "Institucional",
            "Técnico",
            "Timing",
            "R/R",
        ]
    ]

    for i, (_, row) in enumerate(
        df.head(limite).iterrows(),
        start=1,
    ):

        ticker = obter(
            row,
            ["ticker", "symbol"],
        )

        sinal = classificar_sinal(row)

        score_final = obter(
            row,
            [
                "score_final",
                "final_score",
                "ranking_score",
                "score",
            ],
        )

        institucional = obter(
            row,
            [
                "institutional_score",
                "score_institucional",
                "institutional_money_flow_score",
            ],
        )

        tecnico = obter(
            row,
            [
                "technical_score",
                "technical_entry_score",
                "score_tecnico",
            ],
        )

        timing = obter(
            row,
            [
                "entry_timing_score",
                "timing_score",
                "score_timing",
            ],
        )

        rr = obter(
            row,
            [
                "risk_reward",
                "risk_reward_ratio",
                "rr",
                "r_r",
            ],
        )

        linhas.append(
            [
                i,
                ticker,
                sinal,
                numero(score_final),
                numero(institucional),
                numero(tecnico),
                numero(timing),
                numero(rr),
            ]
        )

    return criar_tabela(
        linhas,
        larguras=[
            1.2 * cm,
            2 * cm,
            4.3 * cm,
            2.5 * cm,
            2.8 * cm,
            2.5 * cm,
            2.5 * cm,
            2 * cm,
        ],
    )


# ============================================================
# DETALHE DAS EMPRESAS
# ============================================================

def bloco_empresa(row):

    ticker = obter(
        row,
        ["ticker", "symbol"],
    )

    sinal = classificar_sinal(row)

    score = obter(
        row,
        [
            "score_final",
            "final_score",
            "ranking_score",
            "score",
        ],
    )

    institucional = obter(
        row,
        [
            "institutional_score",
            "score_institucional",
            "institutional_money_flow_score",
        ],
    )

    tecnico = obter(
        row,
        [
            "technical_score",
            "technical_entry_score",
            "score_tecnico",
        ],
    )

    timing = obter(
        row,
        [
            "entry_timing_score",
            "timing_score",
            "score_timing",
        ],
    )

    rr = obter(
        row,
        [
            "risk_reward",
            "risk_reward_ratio",
            "rr",
        ],
    )

    tendencia = obter(
        row,
        [
            "trend",
            "tendencia",
            "trend_quality",
            "qualidade_tendencia",
        ],
    )

    extensao = obter(
        row,
        [
            "extension_risk",
            "risco_extensao",
            "overextension",
        ],
    )

    preco = obter(
        row,
        [
            "close",
            "price",
            "preco",
            "preco_atual",
        ],
    )

    dados = [
        ["Indicador", "Resultado"],
        ["Ticker", ticker],
        ["Classificação", sinal],
        ["Preço", numero(preco)],
        ["Score Final", numero(score)],
        ["Institutional Score", numero(institucional)],
        ["Technical Entry Score", numero(tecnico)],
        ["Entry Timing Score", numero(timing)],
        ["Risk / Reward", numero(rr)],
        ["Tendência", texto_seguro(tendencia)],
        ["Risco de Extensão", texto_seguro(extensao)],
    ]

    bloco = [
        Paragraph(
            f"{ticker} — {sinal}",
            STYLE_SUBSECTION,
        ),
        criar_tabela(
            dados,
            larguras=[
                6 * cm,
                8 * cm,
            ],
            fonte=8,
        ),
        Spacer(1, 0.35 * cm),
    ]

    return KeepTogether(bloco)


# ============================================================
# METODOLOGIA
# ============================================================

def bloco_metodologia():

    texto = """
O AI Infrastructure Scanner foi desenvolvido para identificar
empresas ligadas à infraestrutura de Inteligência Artificial
que apresentem combinação favorável entre fluxo institucional,
estrutura técnica e qualidade do momento de entrada.

O modelo não procura prever preços futuros. Seu objetivo é
classificar probabilisticamente as oportunidades disponíveis
com base em critérios quantitativos e regras previamente
definidas.
"""

    texto2 = """
A arquitetura de decisão combina três camadas principais:
Institutional Money Flow Score, Technical Entry Score e
Entry Timing Score. O ranking final deve ser interpretado em
conjunto com a classificação operacional, evitando que uma
empresa estruturalmente forte seja automaticamente considerada
uma boa entrada quando o preço estiver excessivamente esticado.
"""

    texto3 = """
O horizonte operacional do scanner é swing trade de até
aproximadamente seis meses. As classificações representam
condições quantitativas observadas no momento da execução e
não constituem garantia de retorno futuro.
"""

    return [
        Paragraph(
            "Metodologia",
            STYLE_SECTION,
        ),
        Paragraph(
            texto,
            STYLE_BODY,
        ),
        Paragraph(
            texto2,
            STYLE_BODY,
        ),
        Paragraph(
            texto3,
            STYLE_BODY,
        ),
    ]


# ============================================================
# GERAR PDF
# ============================================================

def gerar_pdf_institucional():

    print("=" * 80)
    print("AI INFRASTRUCTURE SCANNER")
    print("GERANDO RELATÓRIO INSTITUCIONAL")
    print("=" * 80)

    # --------------------------------------------------------
    # CARREGAR BASES
    # --------------------------------------------------------

    ranking = carregar_csv(
        "ranking.csv"
    )

    signals = carregar_csv(
        "signals.csv"
    )

    timing = carregar_csv(
        "entry_timing_score.csv"
    )

    tecnico = carregar_csv(
        "technical_score.csv"
    )

    institucional = carregar_csv(
        "institutional_score.csv"
    )

    # --------------------------------------------------------
    # ESCOLHER BASE PRINCIPAL
    # --------------------------------------------------------

    if not signals.empty:

        # signals.csv é a saída final do Signal Engine e deve ser
        # a fonte principal do relatório operacional.
        base = signals.copy()

    elif not ranking.empty:

        base = ranking.copy()

    elif not timing.empty:

        base = timing.copy()

    else:

        raise RuntimeError(
            "Nenhuma base válida encontrada em /data."
        )

    # --------------------------------------------------------
    # MERGE OPCIONAL
    # --------------------------------------------------------

    ticker_base = localizar_coluna(
        base,
        ["ticker", "symbol"],
    )

    if ticker_base:

        bases_extra = [
            ranking,
            timing,
            tecnico,
            institucional,
        ]

        for extra in bases_extra:

            if extra.empty:
                continue

            ticker_extra = localizar_coluna(
                extra,
                ["ticker", "symbol"],
            )

            if ticker_extra is None:
                continue

            temp = extra.copy()

            if ticker_extra != ticker_base:

                temp = temp.rename(
                    columns={
                        ticker_extra: ticker_base
                    }
                )

            novas_colunas = [
                c
                for c in temp.columns
                if c == ticker_base
                or c not in base.columns
            ]

            temp = temp[
                novas_colunas
            ]

            if len(temp.columns) > 1:

                base = base.merge(
                    temp,
                    on=ticker_base,
                    how="left",
                )

    # --------------------------------------------------------
    # CLASSIFICAÇÕES
    # --------------------------------------------------------

    base["_classificacao_pdf"] = base.apply(
        classificar_sinal,
        axis=1,
    )

    fortes = base[
        base["_classificacao_pdf"]
        == "ENTRADA FORTE"
    ]

    aprovadas = base[
        base["_classificacao_pdf"]
        == "ENTRADA APROVADA"
    ]

    pre_entradas = base[
        base["_classificacao_pdf"]
        == "PRÉ-ENTRADA"
    ]

    aguardar_volume = base[
        base["_classificacao_pdf"]
        == "AGUARDAR VOLUME"
    ]

    aguardar_gatilho = base[
        base["_classificacao_pdf"]
        == "AGUARDAR GATILHO"
    ]

    aguardar_pullback = base[
        base["_classificacao_pdf"]
        == "AGUARDAR PULLBACK"
    ]

    # --------------------------------------------------------
    # DOCUMENTO
    # --------------------------------------------------------

    doc = SimpleDocTemplate(
        str(PDF_FILE),
        pagesize=landscape(A4),
        rightMargin=1.2 * cm,
        leftMargin=1.2 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.3 * cm,
    )

    story = []

    # ========================================================
    # CAPA
    # ========================================================

    story.append(
        Spacer(
            1,
            2.5 * cm,
        )
    )

    story.append(
        Paragraph(
            "AI INFRASTRUCTURE SCANNER",
            STYLE_TITLE,
        )
    )

    story.append(
        Paragraph(
            "Institutional Opportunity Report",
            STYLE_SUBTITLE,
        )
    )

    story.append(
        Spacer(
            1,
            0.5 * cm,
        )
    )

    story.append(
        Paragraph(
            "Infraestrutura de Inteligência Artificial",
            STYLE_SUBTITLE,
        )
    )

    story.append(
        Paragraph(
            "Horizonte Operacional: Swing Trade — até 6 meses",
            STYLE_SUBTITLE,
        )
    )

    story.append(
        Spacer(
            1,
            1.5 * cm,
        )
    )

    data_relatorio = datetime.now().strftime(
        "%d/%m/%Y %H:%M"
    )

    story.append(
        Paragraph(
            f"Relatório gerado em {data_relatorio}",
            STYLE_SUBTITLE,
        )
    )

    story.append(
        PageBreak()
    )

    # ========================================================
    # DASHBOARD
    # ========================================================

    story.append(
        Paragraph(
            "Dashboard Executivo",
            STYLE_SECTION,
        )
    )

    metricas = [
        (
            "Empresas analisadas",
            len(base),
        ),
        (
            "Entradas fortes",
            len(fortes),
        ),
        (
            "Entradas aprovadas",
            len(aprovadas),
        ),
        (
            "Aguardar volume",
            len(aguardar_volume),
        ),
        (
            "Aguardar gatilho",
            len(aguardar_gatilho),
        ),
        (
            "Aguardar pullback",
            len(aguardar_pullback),
        ),
    ]

    story.append(
        criar_cards(metricas)
    )

    story.append(
        Spacer(
            1,
            0.6 * cm,
        )
    )

    # ========================================================
    # RESUMO EXECUTIVO
    # ========================================================

    story.append(
        Paragraph(
            "Resumo Executivo",
            STYLE_SECTION,
        )
    )

    resumo = (
        f"O scanner analisou {len(base)} empresas. "
        f"Foram identificadas {len(fortes)} entradas fortes, "
        f"{len(aprovadas)} entradas aprovadas e "
        f"{len(pre_entradas)} pré-entradas. "
        f"{len(aguardar_volume)} empresas aguardam confirmação "
        f"de volume, {len(aguardar_gatilho)} aguardam gatilho e "
        f"{len(aguardar_pullback)} apresentam condição de "
        f"aguardar pullback."
    )

    story.append(
        Paragraph(
            resumo,
            STYLE_BODY,
        )
    )

    # ========================================================
    # ENTRADAS FORTES
    # ========================================================

    story.append(
        Paragraph(
            "Entradas Fortes",
            STYLE_SECTION,
        )
    )

    if fortes.empty:

        story.append(
            Paragraph(
                "Nenhuma entrada forte identificada.",
                STYLE_BODY,
            )
        )

    else:

        for _, row in fortes.iterrows():

            story.append(
                bloco_empresa(row)
            )

    # ========================================================
    # ENTRADAS APROVADAS
    # ========================================================

    story.append(
        Paragraph(
            "Entradas Aprovadas",
            STYLE_SECTION,
        )
    )

    if aprovadas.empty:

        story.append(
            Paragraph(
                "Nenhuma entrada aprovada identificada.",
                STYLE_BODY,
            )
        )

    else:

        for _, row in aprovadas.iterrows():

            story.append(
                bloco_empresa(row)
            )

    story.append(
        PageBreak()
    )

    # ========================================================
    # TOP 20
    # ========================================================

    story.append(
        Paragraph(
            "Top 20 — Ranking Institucional",
            STYLE_SECTION,
        )
    )

    story.append(
        Paragraph(
            "Ranking consolidado das melhores oportunidades "
            "identificadas pelo motor.",
            STYLE_BODY,
        )
    )

    story.append(
        tabela_oportunidades(
            base,
            limite=20,
        )
    )

    # ========================================================
    # PRÉ-ENTRADAS
    # ========================================================

    story.append(
        Spacer(
            1,
            0.5 * cm,
        )
    )

    story.append(
        Paragraph(
            "Pré-Entradas",
            STYLE_SECTION,
        )
    )

    if pre_entradas.empty:

        story.append(
            Paragraph(
                "Nenhuma pré-entrada identificada.",
                STYLE_BODY,
            )
        )

    else:

        story.append(
            tabela_oportunidades(
                pre_entradas,
                limite=20,
            )
        )

    # ========================================================
    # AGUARDAR VOLUME
    # ========================================================

    story.append(
        Paragraph(
            "Aguardar Confirmação de Volume",
            STYLE_SECTION,
        )
    )

    if aguardar_volume.empty:

        story.append(
            Paragraph(
                "Nenhuma empresa nesta classificação.",
                STYLE_BODY,
            )
        )

    else:

        story.append(
            tabela_oportunidades(
                aguardar_volume,
                limite=20,
            )
        )

    # ========================================================
    # AGUARDAR GATILHO
    # ========================================================

    story.append(
        Paragraph(
            "Aguardar Gatilho",
            STYLE_SECTION,
        )
    )

    if aguardar_gatilho.empty:

        story.append(
            Paragraph(
                "Nenhuma empresa nesta classificação.",
                STYLE_BODY,
            )
        )

    else:

        story.append(
            tabela_oportunidades(
                aguardar_gatilho,
                limite=20,
            )
        )

    # ========================================================
    # PULLBACK
    # ========================================================

    story.append(
        Paragraph(
            "Aguardar Pullback",
            STYLE_SECTION,
        )
    )

    if aguardar_pullback.empty:

        story.append(
            Paragraph(
                "Nenhuma empresa nesta classificação.",
                STYLE_BODY,
            )
        )

    else:

        story.append(
            tabela_oportunidades(
                aguardar_pullback,
                limite=20,
            )
        )

    story.append(
        PageBreak()
    )

    # ========================================================
    # METODOLOGIA
    # ========================================================

    story.extend(
        bloco_metodologia()
    )

    story.append(
        Spacer(
            1,
            0.5 * cm,
        )
    )

    # ========================================================
    # CONCLUSÃO
    # ========================================================

    story.append(
        Paragraph(
            "Conclusão Executiva",
            STYLE_SECTION,
        )
    )

    conclusao = """
O relatório deve ser utilizado como instrumento de priorização
e filtragem. Uma posição elevada no ranking não representa,
isoladamente, autorização de entrada. A decisão operacional
deve respeitar a classificação produzida pelo Signal Engine e
pelo Entry Timing Engine.

Assim, empresas com elevada qualidade estrutural podem
permanecer classificadas como AGUARDAR quando o preço,
volume, gatilho ou relação risco/retorno ainda não oferecerem
condição adequada de entrada.
"""

    story.append(
        Paragraph(
            conclusao,
            STYLE_BODY,
        )
    )

    # ========================================================
    # DISCLAIMER
    # ========================================================

    story.append(
        Paragraph(
            "Disclaimer",
            STYLE_SECTION,
        )
    )

    disclaimer = """
Este relatório possui finalidade exclusivamente quantitativa,
educacional e informativa. Não constitui recomendação
individualizada de investimento, promessa de rentabilidade ou
garantia de desempenho futuro. Os resultados dependem dos
dados disponíveis no momento da execução do modelo.
"""

    story.append(
        Paragraph(
            disclaimer,
            STYLE_BODY,
        )
    )

    # ========================================================
    # BUILD
    # ========================================================

    doc.build(
        story,
        onFirstPage=cabecalho_rodape,
        onLaterPages=cabecalho_rodape,
    )

    print("=" * 80)
    print("RELATÓRIO GERADO COM SUCESSO")
    print(f"Arquivo: {PDF_FILE}")
    print("=" * 80)

    return PDF_FILE


# ============================================================
# EXECUÇÃO DIRETA
# ============================================================

if __name__ == "__main__":
    gerar_pdf_institucional()
