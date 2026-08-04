# ============================================================
# AI INFRASTRUCTURE SCANNER
# settings.py
#
# Configurações gerais do projeto
# ============================================================

from pathlib import Path


# ============================================================
# PROJETO
# ============================================================

PROJECT_NAME = "AI Infrastructure Scanner"

VERSION = "1.0.0"

AUTHOR = "Vilson Pinto"

DESCRIPTION = (
    "Scanner institucional para seleção de ações "
    "de infraestrutura de IA com foco em Swing Trade."
)


# ============================================================
# PASTAS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data"

REPORT_PATH = DATA_PATH

DATA_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# UNIVERSO
# ============================================================

PERIOD = "3y"

INTERVAL = "1d"


# ============================================================
# FILTROS DE LIQUIDEZ
# ============================================================

MIN_DOLLAR_VOLUME = 100_000_000

MIN_HISTORY = 250


# ============================================================
# INDICADORES
# ============================================================

RSI_PERIOD = 14

ADX_PERIOD = 14

ATR_PERIOD = 14

MFI_PERIOD = 14

MACD_FAST = 12

MACD_SLOW = 26

MACD_SIGNAL = 9

SMA_SHORT = 20

SMA_MEDIUM = 50

SMA_LONG = 200

VOLUME_WINDOW = 20


# ============================================================
# SCORING
# ============================================================

MAX_SCORE = 100


# ============================================================
# TECHNICAL ENTRY SCORE
# ============================================================

WEIGHT_DISCOUNT = 25

WEIGHT_MOMENTUM = 20

WEIGHT_TREND = 20

WEIGHT_VOLUME = 20

WEIGHT_RISK = 15


# ============================================================
# INSTITUTIONAL SCORE
# ============================================================

WEIGHT_GROWTH = 30

WEIGHT_MARKET_LEADER = 20

WEIGHT_LIQUIDITY = 20

WEIGHT_HYPE = 15

WEIGHT_SECTOR = 15


# ============================================================
# RANKING FINAL
# ============================================================

WEIGHT_INSTITUTIONAL = 50

WEIGHT_TECHNICAL = 50


# ============================================================
# CRITÉRIOS DE COMPRA
# ============================================================

MIN_TECHNICAL_SCORE = 70

MIN_INSTITUTIONAL_SCORE = 65

MIN_FINAL_SCORE = 70


# ============================================================
# RELATÓRIO
# ============================================================

TOP_N = 20


# ============================================================
# ARQUIVOS
# ============================================================

PRICE_FILE = DATA_PATH / "historico_precos.csv"

INDICATOR_FILE = DATA_PATH / "indicadores.csv"

RANKING_FILE = DATA_PATH / "ranking.csv"

REPORT_FILE = DATA_PATH / "relatorio.xlsx"


# ============================================================
# CORES
# ============================================================

COLOR_BUY = "#16A34A"

COLOR_WAIT = "#EAB308"

COLOR_SELL = "#DC2626"


# ============================================================
# LOG
# ============================================================

SHOW_PROGRESS = True

VERBOSE = True
