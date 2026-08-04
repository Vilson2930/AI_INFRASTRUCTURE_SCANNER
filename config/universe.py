# ============================================================
# AI INFRASTRUCTURE SCANNER
# universe.py
#
# Universo de ações da infraestrutura de IA
# ============================================================

# ============================================================
# SEMICONDUTORES E ACELERADORES
# ============================================================

SEMICONDUCTORS = {

    "NVDA": "NVIDIA",

    "AMD": "Advanced Micro Devices",

    "AVGO": "Broadcom",

    "MRVL": "Marvell Technology"

}


# ============================================================
# FABRICAÇÃO DE CHIPS
# ============================================================

CHIP_MANUFACTURERS = {

    "TSM": "Taiwan Semiconductor",

    "INTC": "Intel",

    "GFS": "GlobalFoundries"

}


# ============================================================
# EQUIPAMENTOS PARA SEMICONDUTORES
# ============================================================

SEMICONDUCTOR_EQUIPMENT = {

    "ASML": "ASML",

    "AMAT": "Applied Materials",

    "LRCX": "Lam Research",

    "KLAC": "KLA"

}


# ============================================================
# MEMÓRIAS
# ============================================================

MEMORY = {

    "MU": "Micron",

    "STX": "Seagate",

    "WDC": "Western Digital"

}


# ============================================================
# CLOUD
# ============================================================

CLOUD = {

    "MSFT": "Microsoft",

    "GOOGL": "Alphabet",

    "AMZN": "Amazon",

    "ORCL": "Oracle"

}


# ============================================================
# SOFTWARE IA
# ============================================================

SOFTWARE = {

    "PLTR": "Palantir",

    "SNOW": "Snowflake",

    "NOW": "ServiceNow",

    "DDOG": "Datadog"

}


# ============================================================
# SERVIDORES
# ============================================================

SERVERS = {

    "SMCI": "Super Micro Computer",

    "DELL": "Dell",

    "HPE": "Hewlett Packard Enterprise"

}


# ============================================================
# REDES
# ============================================================

NETWORKING = {

    "ANET": "Arista Networks",

    "CSCO": "Cisco",

    "CIEN": "Ciena"

}


# ============================================================
# CONECTIVIDADE ÓPTICA
# ============================================================

OPTICAL = {

    "LITE": "Lumentum",

    "COHR": "Coherent",

    "FN": "Fabrinet"

}


# ============================================================
# DATA CENTERS
# ============================================================

DATA_CENTERS = {

    "VRT": "Vertiv",

    "DLR": "Digital Realty",

    "EQIX": "Equinix",

    "TT": "Trane",

    "JCI": "Johnson Controls"

}


# ============================================================
# ELETRIFICAÇÃO
# ============================================================

ELECTRIFICATION = {

    "ETN": "Eaton",

    "GEV": "GE Vernova",

    "EMR": "Emerson",

    "PWR": "Quanta Services",

    "HUBB": "Hubbell"

}


# ============================================================
# ENERGIA
# ============================================================

ENERGY = {

    "CEG": "Constellation Energy",

    "VST": "Vistra",

    "NRG": "NRG Energy",

    "TLN": "Talen Energy"

}


# ============================================================
# CIBERSEGURANÇA
# ============================================================

CYBERSECURITY = {

    "PANW": "Palo Alto Networks",

    "CRWD": "CrowdStrike",

    "FTNT": "Fortinet",

    "ZS": "Zscaler"

}


# ============================================================
# DICIONÁRIO DE SETORES
# ============================================================

SECTORS = {

    "Semicondutores": SEMICONDUCTORS,

    "Fabricação de Chips": CHIP_MANUFACTURERS,

    "Equipamentos": SEMICONDUCTOR_EQUIPMENT,

    "Memórias": MEMORY,

    "Cloud": CLOUD,

    "Software": SOFTWARE,

    "Servidores": SERVERS,

    "Redes": NETWORKING,

    "Conectividade": OPTICAL,

    "Data Centers": DATA_CENTERS,

    "Eletrificação": ELECTRIFICATION,

    "Energia": ENERGY,

    "Cibersegurança": CYBERSECURITY

}


# ============================================================
# LISTA ÚNICA DE TICKERS
# ============================================================

ALL_TICKERS = []

for companies in SECTORS.values():

    ALL_TICKERS.extend(companies.keys())

ALL_TICKERS = sorted(list(set(ALL_TICKERS)))


# ============================================================
# MAPA TICKER -> SETOR
# ============================================================

TICKER_TO_SECTOR = {}

for sector, companies in SECTORS.items():

    for ticker in companies:

        TICKER_TO_SECTOR[ticker] = sector


# ============================================================
# MAPA TICKER -> EMPRESA
# ============================================================

TICKER_TO_NAME = {}

for companies in SECTORS.values():

    TICKER_TO_NAME.update(companies)


# ============================================================
# ESTATÍSTICAS
# ============================================================

TOTAL_COMPANIES = len(ALL_TICKERS)

TOTAL_SECTORS = len(SECTORS)
