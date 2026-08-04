# ============================================================
# AI INFRASTRUCTURE SCANNER
# market_data.py
#
# Coleta, normalização e validação dos preços.
# Compatível com colunas MultiIndex do yfinance.
# ============================================================

from __future__ import annotations

import time
import warnings
from typing import Dict

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm.auto import tqdm

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
# COLUNAS OBRIGATÓRIAS
# ============================================================

REQUIRED_PRICE_COLUMNS = {
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
}


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def normalize_yfinance_columns(
    data: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """
    Normaliza as colunas retornadas pelo yfinance.

    Trata:
    - colunas simples;
    - colunas MultiIndex;
    - ticker no primeiro ou no segundo nível;
    - nomes Date, Datetime e index.
    """

    df = data.copy()

    if df.empty:
        return df

    # --------------------------------------------------------
    # TRATAMENTO DE MULTIINDEX
    # --------------------------------------------------------

    if isinstance(df.columns, pd.MultiIndex):

        level_zero = [
            str(value).strip()
            for value in df.columns.get_level_values(0)
        ]

        level_one = [
            str(value).strip()
            for value in df.columns.get_level_values(1)
        ]

        price_names = {
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
        }

        # Normalmente o nível 0 contém Open, High, Low etc.
        if any(
            value in price_names
            for value in level_zero
        ):
            df.columns = level_zero

        # Algumas versões podem retornar o ticker no nível 0.
        elif any(
            value in price_names
            for value in level_one
        ):
            df.columns = level_one

        else:
            df.columns = [
                "_".join(
                    str(part).strip()
                    for part in column
                    if str(part).strip()
                )
                for column in df.columns
            ]

    # --------------------------------------------------------
    # ÍNDICE PARA COLUNA
    # --------------------------------------------------------

    df = df.reset_index()

    # --------------------------------------------------------
    # PADRONIZAÇÃO DOS NOMES
    # --------------------------------------------------------

    normalized_columns = {}

    for column in df.columns:

        normalized = (
            str(column)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        aliases = {
            "datetime": "date",
            "index": "date",
            "adj_close": "adjusted_close",
            "adjclose": "adjusted_close",
        }

        normalized_columns[column] = aliases.get(
            normalized,
            normalized,
        )

    df = df.rename(
        columns=normalized_columns
    )

    # Caso não exista close, mas exista adjusted_close.
    if (
        "close" not in df.columns
        and "adjusted_close" in df.columns
    ):
        df["close"] = df["adjusted_close"]

    # Remove possíveis colunas duplicadas.
    df = df.loc[
        :,
        ~df.columns.duplicated()
    ].copy()

    missing_columns = (
        REQUIRED_PRICE_COLUMNS
        .difference(df.columns)
    )

    if missing_columns:
        raise KeyError(
            f"{ticker}: colunas ausentes após normalização: "
            f"{sorted(missing_columns)}. "
            f"Colunas recebidas: {list(df.columns)}"
        )

    # --------------------------------------------------------
    # CONVERSÃO DE TIPOS
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
        utc=True,
    ).dt.tz_localize(None)

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # LIMPEZA
    # --------------------------------------------------------

    df = (
        df.dropna(
            subset=[
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )
        .loc[df["close"] > 0]
        .loc[df["high"] >= df["low"]]
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    df["ticker"] = ticker
    df["empresa"] = TICKER_TO_NAME.get(
        ticker,
        ticker,
    )
    df["setor"] = TICKER_TO_SECTOR.get(
        ticker,
        "Não classificado",
    )

    return df


def download_single_ticker(
    ticker: str,
    attempts: int = 3,
) -> pd.DataFrame:
    """
    Baixa um ticker com tentativas adicionais.
    """

    last_error: Exception | None = None

    for attempt in range(
        1,
        attempts + 1,
    ):

        try:

            raw_data = yf.download(
                tickers=ticker,
                period=PERIOD,
                interval=INTERVAL,
                auto_adjust=True,
                progress=False,
                threads=False,
                group_by="column",
                multi_level_index=False,
                timeout=20,
            )

            if raw_data is None or raw_data.empty:
                raise ValueError(
                    "Nenhum dado retornado pelo Yahoo Finance."
                )

            normalized = normalize_yfinance_columns(
                data=raw_data,
                ticker=ticker,
            )

            if normalized.empty:
                raise ValueError(
                    "Dados vazios após normalização."
                )

            return normalized

        except Exception as error:

            last_error = error

            if attempt < attempts:
                time.sleep(
                    attempt * 1.5
                )

    raise RuntimeError(
        f"{ticker}: falha após {attempts} tentativas. "
        f"Último erro: {last_error}"
    )


# ============================================================
# CLASSE PRINCIPAL
# ============================================================

class MarketData:
    """
    Motor de coleta e validação dos dados de mercado.
    """

    def __init__(self) -> None:

        self.data: Dict[
            str,
            pd.DataFrame,
        ] = {}

        self.summary = pd.DataFrame()

        self.failures = pd.DataFrame()

    # ========================================================
    # DOWNLOAD
    # ========================================================

    def download_prices(
        self,
    ) -> pd.DataFrame:
        """
        Baixa e valida todos os tickers do universo.
        """

        print("=" * 90)
        print("DOWNLOAD DOS PREÇOS")
        print("=" * 90)

        print(
            f"Quantidade de tickers: {len(ALL_TICKERS)}"
        )

        print(
            f"Período: {PERIOD}"
        )

        print(
            f"Intervalo: {INTERVAL}"
        )

        print("=" * 90)

        summary_records: list[dict] = []

        historical_frames: list[
            pd.DataFrame
        ] = []

        failure_records: list[dict] = []

        for ticker in tqdm(
            ALL_TICKERS,
            desc="Baixando preços",
        ):

            try:

                df = download_single_ticker(
                    ticker=ticker,
                    attempts=3,
                )

                trading_days = len(df)

                current_price = float(
                    df["close"].iloc[-1]
                )

                average_volume_20d = float(
                    df["volume"]
                    .tail(20)
                    .mean()
                )

                average_dollar_volume_20d = float(
                    (
                        df["close"]
                        *
                        df["volume"]
                    )
                    .tail(20)
                    .mean()
                )

                status = "APROVADO"
                reason = "Dados válidos."

                if trading_days < MIN_HISTORY:

                    status = "REPROVADO"

                    reason = (
                        "Histórico insuficiente: "
                        f"{trading_days} pregões; "
                        f"mínimo exigido: {MIN_HISTORY}."
                    )

                elif (
                    not np.isfinite(
                        average_dollar_volume_20d
                    )
                    or average_dollar_volume_20d
                    < MIN_DOLLAR_VOLUME
                ):

                    status = "REPROVADO"

                    reason = (
                        "Liquidez inferior ao mínimo: "
                        f"US$ {average_dollar_volume_20d:,.2f}; "
                        f"mínimo: US$ {MIN_DOLLAR_VOLUME:,.2f}."
                    )

                summary_records.append(
                    {
                        "ticker": ticker,
                        "empresa": TICKER_TO_NAME.get(
                            ticker,
                            ticker,
                        ),
                        "setor": TICKER_TO_SECTOR.get(
                            ticker,
                            "Não classificado",
                        ),
                        "status": status,
                        "pregoes": trading_days,
                        "preco_atual": round(
                            current_price,
                            4,
                        ),
                        "volume_medio_20d": round(
                            average_volume_20d,
                            2,
                        ),
                        "volume_financeiro_medio_20d": round(
                            average_dollar_volume_20d,
                            2,
                        ),
                        "motivo": reason,
                    }
                )

                if status == "APROVADO":

                    historical_frames.append(
                        df
                    )

                    self.data[ticker] = (
                        df.copy()
                    )

            except Exception as error:

                error_message = str(
                    error
                )

                summary_records.append(
                    {
                        "ticker": ticker,
                        "empresa": TICKER_TO_NAME.get(
                            ticker,
                            ticker,
                        ),
                        "setor": TICKER_TO_SECTOR.get(
                            ticker,
                            "Não classificado",
                        ),
                        "status": "ERRO",
                        "pregoes": 0,
                        "preco_atual": np.nan,
                        "volume_medio_20d": np.nan,
                        "volume_financeiro_medio_20d": np.nan,
                        "motivo": error_message,
                    }
                )

                failure_records.append(
                    {
                        "ticker": ticker,
                        "erro": error_message,
                    }
                )

        self.summary = pd.DataFrame(
            summary_records
        )

        self.failures = pd.DataFrame(
            failure_records
        )

        # ----------------------------------------------------
        # PROTEÇÃO CONTRA LISTA VAZIA
        # ----------------------------------------------------

        if not historical_frames:

            print()
            print("=" * 90)
            print("FALHA NA COLETA")
            print("=" * 90)

            print(
                "Nenhum ticker foi aprovado."
            )

            if not self.failures.empty:

                print()
                print("ERROS IDENTIFICADOS:")

                print(
                    self.failures
                    .head(15)
                    .to_string(
                        index=False
                    )
                )

            summary_file = (
                PRICE_FILE.parent
                /
                "resumo_coleta_precos.csv"
            )

            PRICE_FILE.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            self.summary.to_csv(
                summary_file,
                index=False,
                encoding="utf-8-sig",
            )

            raise RuntimeError(
                "Nenhum histórico válido foi coletado. "
                "Consulte data/resumo_coleta_precos.csv "
                "para identificar os erros de cada ticker."
            )

        historical_data = (
            pd.concat(
                historical_frames,
                ignore_index=True,
            )
            .sort_values(
                [
                    "ticker",
                    "date",
                ]
            )
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # SALVAMENTO
        # ----------------------------------------------------

        PRICE_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        historical_data.to_csv(
            PRICE_FILE,
            index=False,
            encoding="utf-8-sig",
        )

        summary_file = (
            PRICE_FILE.parent
            /
            "resumo_coleta_precos.csv"
        )

        self.summary.to_csv(
            summary_file,
            index=False,
            encoding="utf-8-sig",
        )

        if not self.failures.empty:

            failure_file = (
                PRICE_FILE.parent
                /
                "falhas_coleta_precos.csv"
            )

            self.failures.to_csv(
                failure_file,
                index=False,
                encoding="utf-8-sig",
            )

        # ----------------------------------------------------
        # RELATÓRIO
        # ----------------------------------------------------

        approved_count = int(
            (
                self.summary["status"]
                == "APROVADO"
            ).sum()
        )

        rejected_count = int(
            (
                self.summary["status"]
                == "REPROVADO"
            ).sum()
        )

        error_count = int(
            (
                self.summary["status"]
                == "ERRO"
            ).sum()
        )

        print()
        print("=" * 90)
        print("RESULTADO DA COLETA")
        print("=" * 90)

        print(
            f"✅ Aprovadas: {approved_count}"
        )

        print(
            f"⚠️ Reprovadas: {rejected_count}"
        )

        print(
            f"❌ Erros: {error_count}"
        )

        print(
            f"📊 Total analisado: "
            f"{len(self.summary)}"
        )

        print()
        print("Arquivos criados:")

        print(
            f"- {PRICE_FILE}"
        )

        print(
            f"- {summary_file}"
        )

        print("=" * 90)

        return historical_data

    # ========================================================
    # CONSULTAS
    # ========================================================

    def get_summary(
        self,
    ) -> pd.DataFrame:
        """
        Retorna o resumo da coleta.
        """

        return self.summary.copy()

    def get_data(
        self,
    ) -> Dict[str, pd.DataFrame]:
        """
        Retorna os históricos por ticker.
        """

        return {
            ticker: data.copy()
            for ticker, data
            in self.data.items()
        }

    def get_failures(
        self,
    ) -> pd.DataFrame:
        """
        Retorna as falhas da coleta.
        """

        return self.failures.copy()


# ============================================================
# EXECUÇÃO DIRETA
# ============================================================

if __name__ == "__main__":

    engine = MarketData()

    prices = engine.download_prices()

    print()
    print("Últimos registros:")

    print(
        prices
        .groupby(
            "ticker",
            as_index=False,
        )
        .tail(1)
        .head(20)
        .to_string(
            index=False
        )
    )
