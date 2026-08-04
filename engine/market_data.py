# ============================================================
# AI INFRASTRUCTURE SCANNER
# market_data.py
#
# Download e validação dos preços
# ============================================================

from __future__ import annotations

import warnings
from typing import Dict

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from config.settings import (
    PERIOD,
    INTERVAL,
    MIN_HISTORY,
    MIN_DOLLAR_VOLUME,
    PRICE_FILE,
)

from config.universe import (
    ALL_TICKERS,
    TICKER_TO_NAME,
    TICKER_TO_SECTOR,
)

warnings.filterwarnings("ignore")


# ============================================================
# CLASSE
# ============================================================

class MarketData:

    def __init__(self):

        self.data = {}

        self.summary = pd.DataFrame()

    # ========================================================
    # DOWNLOAD
    # ========================================================

    def download_prices(self):

        print("=" * 80)
        print("DOWNLOAD DOS PREÇOS")
        print("=" * 80)

        resumo = []

        historico = []

        for ticker in tqdm(ALL_TICKERS):

            try:

                df = yf.download(
                    ticker,
                    period=PERIOD,
                    interval=INTERVAL,
                    auto_adjust=True,
                    progress=False,
                    threads=False
                )

                if df.empty:

                    resumo.append({

                        "ticker": ticker,

                        "empresa": TICKER_TO_NAME[ticker],

                        "setor": TICKER_TO_SECTOR[ticker],

                        "status": "SEM DADOS"

                    })

                    continue

                df.reset_index(inplace=True)

                df.columns = [
                    c.lower().replace(" ", "_")
                    for c in df.columns
                ]

                df["ticker"] = ticker

                df["empresa"] = TICKER_TO_NAME[ticker]

                df["setor"] = TICKER_TO_SECTOR[ticker]

                volume_medio = (
                    df["volume"]
                    .tail(20)
                    .mean()
                )

                preco = df.iloc[-1]["close"]

                volume_financeiro = (
                    volume_medio * preco
                )

                status = "APROVADO"

                motivo = ""

                if len(df) < MIN_HISTORY:

                    status = "REPROVADO"

                    motivo = "Histórico insuficiente"

                if volume_financeiro < MIN_DOLLAR_VOLUME:

                    status = "REPROVADO"

                    motivo = "Baixa liquidez"

                resumo.append({

                    "ticker": ticker,

                    "empresa": TICKER_TO_NAME[ticker],

                    "setor": TICKER_TO_SECTOR[ticker],

                    "status": status,

                    "pregoes": len(df),

                    "preco": round(preco,2),

                    "volume_medio": int(volume_medio),

                    "volume_financeiro": round(volume_financeiro,2),

                    "motivo": motivo

                })

                if status == "APROVADO":

                    historico.append(df)

                    self.data[ticker] = df.copy()

            except Exception as erro:

                resumo.append({

                    "ticker": ticker,

                    "empresa": TICKER_TO_NAME[ticker],

                    "setor": TICKER_TO_SECTOR[ticker],

                    "status": "ERRO",

                    "motivo": str(erro)

                })

        self.summary = pd.DataFrame(resumo)

        historico = pd.concat(
            historico,
            ignore_index=True
        )

        historico.to_csv(
            PRICE_FILE,
            index=False,
            encoding="utf-8-sig"
        )

        print()

        print("="*80)

        print("DOWNLOAD FINALIZADO")

        print("="*80)

        print()

        print("Aprovadas:",
              (self.summary.status=="APROVADO").sum())

        print("Reprovadas:",
              (self.summary.status=="REPROVADO").sum())

        print("Erros:",
              (self.summary.status=="ERRO").sum())

        print()

        return historico

    # ========================================================
    # RESUMO
    # ========================================================

    def get_summary(self):

        return self.summary

    # ========================================================
    # DADOS
    # ========================================================

    def get_data(self) -> Dict:

        return self.data
