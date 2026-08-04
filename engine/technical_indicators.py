# ============================================================
# AI INFRASTRUCTURE SCANNER
# technical_indicators.py
#
# Cálculo dos indicadores técnicos utilizados pelo scanner.
#
# Indicadores:
# - SMA 20, 50 e 200
# - RSI 14
# - MACD 12, 26 e 9
# - ATR 14
# - ADX 14, +DI e -DI
# - MFI 14
# - OBV
# - CMF 20
# - VWAP móvel de 20 dias
# - Volume relativo
# - Máxima e mínima de 52 semanas
# - Retornos de 1, 3, 6 e 12 meses
# - Distâncias das médias e da máxima anual
# ============================================================

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from config.settings import (
    RSI_PERIOD,
    ADX_PERIOD,
    ATR_PERIOD,
    MFI_PERIOD,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    SMA_SHORT,
    SMA_MEDIUM,
    SMA_LONG,
    VOLUME_WINDOW,
    INDICATOR_FILE,
    SHOW_PROGRESS,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONSTANTES
# ============================================================

TRADING_DAYS_1_MONTH = 21
TRADING_DAYS_3_MONTHS = 63
TRADING_DAYS_6_MONTHS = 126
TRADING_DAYS_12_MONTHS = 252

REQUIRED_COLUMNS = {
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "volume",
}


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """
    Realiza divisão segura entre duas séries.

    Valores infinitos são convertidos para NaN.
    """

    safe_denominator = denominator.replace(
        0,
        np.nan,
    )

    result = numerator / safe_denominator

    return result.replace(
        [np.inf, -np.inf],
        np.nan,
    )


def normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Padroniza os nomes das colunas.

    Também trata DataFrames com colunas MultiIndex.
    """

    df = data.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(
                str(part)
                for part in column
                if str(part) not in {"", "None"}
            )
            for column in df.columns
        ]

    df.columns = [
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        for column in df.columns
    ]

    aliases = {
        "datetime": "date",
        "index": "date",
        "adj_close": "close",
        "adjclose": "close",
    }

    df = df.rename(
        columns={
            column: aliases.get(column, column)
            for column in df.columns
        }
    )

    return df


def validate_price_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Valida e prepara o histórico recebido do market_data.py.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "O histórico de preços deve ser um pandas DataFrame."
        )

    if data.empty:
        raise ValueError(
            "O histórico de preços está vazio."
        )

    df = normalize_columns(data)

    missing_columns = REQUIRED_COLUMNS.difference(
        df.columns
    )

    if missing_columns:
        raise KeyError(
            "Colunas obrigatórias ausentes no histórico: "
            f"{sorted(missing_columns)}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

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

    df["ticker"] = (
        df["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df = (
        df.dropna(
            subset=[
                "date",
                "ticker",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )
        .loc[df["close"] > 0]
        .loc[df["high"] >= df["low"]]
        .sort_values(
            ["ticker", "date"]
        )
        .drop_duplicates(
            subset=[
                "ticker",
                "date",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError(
            "Nenhuma linha válida permaneceu após a validação."
        )

    return df


# ============================================================
# RSI
# ============================================================

def calculate_rsi(
    close: pd.Series,
    period: int = RSI_PERIOD,
) -> pd.Series:
    """
    Calcula o RSI utilizando suavização de Wilder.
    """

    delta = close.diff()

    gains = delta.clip(
        lower=0,
    )

    losses = -delta.clip(
        upper=0,
    )

    average_gain = gains.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    average_loss = losses.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    relative_strength = safe_divide(
        average_gain,
        average_loss,
    )

    rsi = 100 - (
        100 / (1 + relative_strength)
    )

    rsi = rsi.where(
        average_loss != 0,
        100,
    )

    rsi = rsi.where(
        average_gain != 0,
        0,
    )

    return rsi


# ============================================================
# MACD
# ============================================================

def calculate_macd(
    close: pd.Series,
    fast_period: int = MACD_FAST,
    slow_period: int = MACD_SLOW,
    signal_period: int = MACD_SIGNAL,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calcula MACD, linha de sinal e histograma.
    """

    ema_fast = close.ewm(
        span=fast_period,
        adjust=False,
    ).mean()

    ema_slow = close.ewm(
        span=slow_period,
        adjust=False,
    ).mean()

    macd = ema_fast - ema_slow

    macd_signal = macd.ewm(
        span=signal_period,
        adjust=False,
    ).mean()

    macd_histogram = (
        macd - macd_signal
    )

    return (
        macd,
        macd_signal,
        macd_histogram,
    )


# ============================================================
# ATR
# ============================================================

def calculate_true_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """
    Calcula o True Range.
    """

    previous_close = close.shift(1)

    ranges = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    )

    return ranges.max(axis=1)


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = ATR_PERIOD,
) -> pd.Series:
    """
    Calcula o ATR utilizando suavização de Wilder.
    """

    true_range = calculate_true_range(
        high=high,
        low=low,
        close=close,
    )

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


# ============================================================
# ADX
# ============================================================

def calculate_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = ADX_PERIOD,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calcula ADX, +DI e -DI.
    """

    upward_movement = high.diff()
    downward_movement = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (upward_movement > downward_movement)
            & (upward_movement > 0),
            upward_movement,
            0.0,
        ),
        index=high.index,
        dtype=float,
    )

    minus_dm = pd.Series(
        np.where(
            (downward_movement > upward_movement)
            & (downward_movement > 0),
            downward_movement,
            0.0,
        ),
        index=high.index,
        dtype=float,
    )

    atr = calculate_atr(
        high=high,
        low=low,
        close=close,
        period=period,
    )

    smoothed_plus_dm = plus_dm.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    smoothed_minus_dm = minus_dm.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    plus_di = 100 * safe_divide(
        smoothed_plus_dm,
        atr,
    )

    minus_di = 100 * safe_divide(
        smoothed_minus_dm,
        atr,
    )

    directional_index = 100 * safe_divide(
        (plus_di - minus_di).abs(),
        plus_di + minus_di,
    )

    adx = directional_index.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    return (
        adx,
        plus_di,
        minus_di,
    )


# ============================================================
# MFI
# ============================================================

def calculate_mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = MFI_PERIOD,
) -> pd.Series:
    """
    Calcula o Money Flow Index.
    """

    typical_price = (
        high + low + close
    ) / 3

    raw_money_flow = (
        typical_price * volume
    )

    price_change = typical_price.diff()

    positive_flow = pd.Series(
        np.where(
            price_change > 0,
            raw_money_flow,
            0.0,
        ),
        index=close.index,
        dtype=float,
    )

    negative_flow = pd.Series(
        np.where(
            price_change < 0,
            raw_money_flow,
            0.0,
        ),
        index=close.index,
        dtype=float,
    )

    positive_sum = positive_flow.rolling(
        period
    ).sum()

    negative_sum = negative_flow.rolling(
        period
    ).sum()

    money_flow_ratio = safe_divide(
        positive_sum,
        negative_sum,
    )

    mfi = 100 - (
        100 / (1 + money_flow_ratio)
    )

    mfi = mfi.where(
        negative_sum != 0,
        100,
    )

    return mfi


# ============================================================
# OBV
# ============================================================

def calculate_obv(
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """
    Calcula o On-Balance Volume.
    """

    direction = np.sign(
        close.diff()
    ).fillna(0)

    signed_volume = (
        direction * volume
    )

    return signed_volume.cumsum()


# ============================================================
# CMF
# ============================================================

def calculate_cmf(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    Calcula o Chaikin Money Flow.
    """

    price_range = (
        high - low
    ).replace(
        0,
        np.nan,
    )

    money_flow_multiplier = (
        (
            close - low
        )
        -
        (
            high - close
        )
    ) / price_range

    money_flow_volume = (
        money_flow_multiplier.fillna(0)
        * volume
    )

    return safe_divide(
        money_flow_volume.rolling(
            period
        ).sum(),
        volume.rolling(
            period
        ).sum(),
    )


# ============================================================
# VWAP MÓVEL
# ============================================================

def calculate_rolling_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = VOLUME_WINDOW,
) -> pd.Series:
    """
    Calcula o VWAP móvel.
    """

    typical_price = (
        high + low + close
    ) / 3

    numerator = (
        typical_price * volume
    ).rolling(
        period
    ).sum()

    denominator = volume.rolling(
        period
    ).sum()

    return safe_divide(
        numerator,
        denominator,
    )


# ============================================================
# INDICADORES POR TICKER
# ============================================================

def calculate_ticker_indicators(
    ticker_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula todos os indicadores de um único ticker.
    """

    df = (
        ticker_data.copy()
        .sort_values("date")
        .reset_index(drop=True)
    )

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # --------------------------------------------------------
    # MÉDIAS MÓVEIS
    # --------------------------------------------------------

    df["sma_20"] = close.rolling(
        SMA_SHORT
    ).mean()

    df["sma_50"] = close.rolling(
        SMA_MEDIUM
    ).mean()

    df["sma_200"] = close.rolling(
        SMA_LONG
    ).mean()

    df["sma_20_slope_5d"] = (
        df["sma_20"]
        / df["sma_20"].shift(5)
        - 1
    ) * 100

    df["sma_50_slope_20d"] = (
        df["sma_50"]
        / df["sma_50"].shift(20)
        - 1
    ) * 100

    df["sma_200_slope_20d"] = (
        df["sma_200"]
        / df["sma_200"].shift(20)
        - 1
    ) * 100

    # --------------------------------------------------------
    # DISTÂNCIA DAS MÉDIAS
    # --------------------------------------------------------

    df["distance_sma_20"] = (
        close / df["sma_20"] - 1
    ) * 100

    df["distance_sma_50"] = (
        close / df["sma_50"] - 1
    ) * 100

    df["distance_sma_200"] = (
        close / df["sma_200"] - 1
    ) * 100

    # Nomes em português mantidos para compatibilidade
    df["distancia_sma_20"] = (
        df["distance_sma_20"]
    )

    df["distancia_sma_50"] = (
        df["distance_sma_50"]
    )

    df["distancia_sma_200"] = (
        df["distance_sma_200"]
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    df["rsi_14"] = calculate_rsi(
        close=close,
        period=RSI_PERIOD,
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    (
        df["macd"],
        df["macd_signal"],
        df["macd_hist"],
    ) = calculate_macd(
        close=close,
    )

    df["macd_hist_change"] = (
        df["macd_hist"].diff()
    )

    df["macd_hist_variacao"] = (
        df["macd_hist_change"]
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    df["atr_14"] = calculate_atr(
        high=high,
        low=low,
        close=close,
        period=ATR_PERIOD,
    )

    df["atr_percent"] = (
        df["atr_14"] / close * 100
    )

    df["atr_percentual"] = (
        df["atr_percent"]
    )

    # --------------------------------------------------------
    # ADX
    # --------------------------------------------------------

    (
        df["adx_14"],
        df["plus_di_14"],
        df["minus_di_14"],
    ) = calculate_adx(
        high=high,
        low=low,
        close=close,
        period=ADX_PERIOD,
    )

    # --------------------------------------------------------
    # MFI
    # --------------------------------------------------------

    df["mfi_14"] = calculate_mfi(
        high=high,
        low=low,
        close=close,
        volume=volume,
        period=MFI_PERIOD,
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    df["volume_average_20d"] = (
        volume.rolling(
            VOLUME_WINDOW
        ).mean()
    )

    df["volume_medio_20d"] = (
        df["volume_average_20d"]
    )

    df["relative_volume_20d"] = (
        safe_divide(
            volume,
            df["volume_average_20d"],
        )
    )

    df["volume_relativo_20d"] = (
        df["relative_volume_20d"]
    )

    df["average_dollar_volume_20d"] = (
        (
            close * volume
        )
        .rolling(
            VOLUME_WINDOW
        )
        .mean()
    )

    # --------------------------------------------------------
    # OBV
    # --------------------------------------------------------

    df["obv"] = calculate_obv(
        close=close,
        volume=volume,
    )

    df["obv_sma_20"] = (
        df["obv"]
        .rolling(20)
        .mean()
    )

    df["obv_above_average"] = (
        df["obv"] > df["obv_sma_20"]
    )

    df["obv_acima_media"] = (
        df["obv_above_average"]
    )

    # --------------------------------------------------------
    # CMF
    # --------------------------------------------------------

    df["cmf_20"] = calculate_cmf(
        high=high,
        low=low,
        close=close,
        volume=volume,
        period=20,
    )

    # --------------------------------------------------------
    # VWAP
    # --------------------------------------------------------

    df["vwap_20"] = calculate_rolling_vwap(
        high=high,
        low=low,
        close=close,
        volume=volume,
        period=VOLUME_WINDOW,
    )

    df["above_vwap_20"] = (
        close > df["vwap_20"]
    )

    df["acima_vwap_20"] = (
        df["above_vwap_20"]
    )

    # --------------------------------------------------------
    # MÁXIMA E MÍNIMA DE 52 SEMANAS
    # --------------------------------------------------------

    df["high_52w"] = (
        high.rolling(
            TRADING_DAYS_12_MONTHS,
            min_periods=126,
        ).max()
    )

    df["low_52w"] = (
        low.rolling(
            TRADING_DAYS_12_MONTHS,
            min_periods=126,
        ).min()
    )

    df["maxima_52s"] = (
        df["high_52w"]
    )

    df["minima_52s"] = (
        df["low_52w"]
    )

    df["distance_from_52w_high"] = (
        close / df["high_52w"] - 1
    ) * 100

    df["distancia_maxima_52s"] = (
        df["distance_from_52w_high"]
    )

    annual_range = (
        df["high_52w"]
        - df["low_52w"]
    )

    df["position_in_52w_range"] = (
        safe_divide(
            close - df["low_52w"],
            annual_range,
        )
        * 100
    )

    df["posicao_intervalo_52s"] = (
        df["position_in_52w_range"]
    )

    # --------------------------------------------------------
    # RETORNOS
    # --------------------------------------------------------

    df["return_1m"] = (
        close.pct_change(
            TRADING_DAYS_1_MONTH,
            fill_method=None,
        )
        * 100
    )

    df["return_3m"] = (
        close.pct_change(
            TRADING_DAYS_3_MONTHS,
            fill_method=None,
        )
        * 100
    )

    df["return_6m"] = (
        close.pct_change(
            TRADING_DAYS_6_MONTHS,
            fill_method=None,
        )
        * 100
    )

    df["return_12m"] = (
        close.pct_change(
            TRADING_DAYS_12_MONTHS,
            fill_method=None,
        )
        * 100
    )

    df["retorno_1m"] = (
        df["return_1m"]
    )

    df["retorno_3m"] = (
        df["return_3m"]
    )

    df["retorno_6m"] = (
        df["return_6m"]
    )

    df["retorno_12m"] = (
        df["return_12m"]
    )

    # --------------------------------------------------------
    # SINAIS AUXILIARES
    # --------------------------------------------------------

    df["above_sma_20"] = (
        close > df["sma_20"]
    )

    df["above_sma_50"] = (
        close > df["sma_50"]
    )

    df["above_sma_200"] = (
        close > df["sma_200"]
    )

    df["acima_sma_200"] = (
        df["above_sma_200"]
    )

    df["bullish_macd_cross"] = (
        (df["macd"] > df["macd_signal"])
        &
        (
            df["macd"].shift(1)
            <= df["macd_signal"].shift(1)
        )
    )

    df["cruzamento_macd_alta"] = (
        df["bullish_macd_cross"]
    )

    df["rsi_leaving_oversold"] = (
        (df["rsi_14"] > 30)
        &
        (
            df["rsi_14"].shift(1)
            <= 30
        )
    )

    df["rsi_saindo_sobrevenda"] = (
        df["rsi_leaving_oversold"]
    )

    df["positive_directional_trend"] = (
        df["plus_di_14"]
        >
        df["minus_di_14"]
    )

    df["macd_hist_rising"] = (
        df["macd_hist_change"] > 0
    )

    # --------------------------------------------------------
    # QUALIDADE DOS INDICADORES
    # --------------------------------------------------------

    critical_indicator_columns = [
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "adx_14",
        "atr_14",
        "mfi_14",
        "sma_20",
        "sma_50",
        "sma_200",
        "volume_relativo_20d",
        "distancia_maxima_52s",
    ]

    df["indicators_complete"] = (
        df[
            critical_indicator_columns
        ]
        .notna()
        .all(axis=1)
    )

    df["indicadores_completos"] = (
        df["indicators_complete"]
    )

    return df


# ============================================================
# CLASSE PRINCIPAL
# ============================================================

class TechnicalIndicators:
    """
    Motor responsável pelo cálculo dos indicadores técnicos.
    """

    def __init__(
        self,
        output_file: str | Path = INDICATOR_FILE,
    ) -> None:
        self.output_file = Path(
            output_file
        )

        self.history = pd.DataFrame()
        self.latest = pd.DataFrame()
        self.failures = pd.DataFrame()

    def calculate(
        self,
        price_history: pd.DataFrame,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Calcula os indicadores para todos os tickers.

        Parameters
        ----------
        price_history:
            Histórico produzido pelo market_data.py.

        save:
            Quando True, salva o histórico completo em
            data/indicadores.csv.

        Returns
        -------
        pandas.DataFrame
            Histórico completo com os indicadores.
        """

        validated_data = validate_price_data(
            price_history
        )

        processed_tickers: list[
            pd.DataFrame
        ] = []

        failures: list[dict] = []

        ticker_groups = list(
            validated_data.groupby(
                "ticker",
                sort=True,
            )
        )

        iterator: Iterable = ticker_groups

        if SHOW_PROGRESS:
            iterator = tqdm(
                ticker_groups,
                desc="Calculando indicadores",
            )

        for ticker, ticker_data in iterator:
            try:
                result = calculate_ticker_indicators(
                    ticker_data
                )

                processed_tickers.append(
                    result
                )

            except Exception as error:
                failures.append(
                    {
                        "ticker": ticker,
                        "error": str(error),
                    }
                )

        if not processed_tickers:
            raise RuntimeError(
                "Nenhum ticker teve os indicadores calculados."
            )

        self.history = (
            pd.concat(
                processed_tickers,
                ignore_index=True,
            )
            .sort_values(
                ["ticker", "date"]
            )
            .reset_index(drop=True)
        )

        self.failures = pd.DataFrame(
            failures
        )

        self.latest = (
            self.history
            .sort_values(
                ["ticker", "date"]
            )
            .groupby(
                "ticker",
                as_index=False,
            )
            .tail(1)
            .sort_values(
                "ticker"
            )
            .reset_index(drop=True)
        )

        if save:
            self.save()

        self.print_summary()

        return self.history

    def save(self) -> None:
        """
        Salva o histórico completo dos indicadores.
        """

        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.history.to_csv(
            self.output_file,
            index=False,
            encoding="utf-8-sig",
        )

    def get_history(self) -> pd.DataFrame:
        """
        Retorna o histórico completo.
        """

        return self.history.copy()

    def get_latest(self) -> pd.DataFrame:
        """
        Retorna o registro mais recente de cada ticker.
        """

        return self.latest.copy()

    def get_failures(self) -> pd.DataFrame:
        """
        Retorna os tickers que apresentaram erro.
        """

        return self.failures.copy()

    def print_summary(self) -> None:
        """
        Exibe um resumo da execução.
        """

        calculated = (
            self.latest["ticker"].nunique()
            if not self.latest.empty
            else 0
        )

        complete = (
            int(
                self.latest[
                    "indicadores_completos"
                ].sum()
            )
            if (
                not self.latest.empty
                and
                "indicadores_completos"
                in self.latest.columns
            )
            else 0
        )

        failures = len(
            self.failures
        )

        print()
        print("=" * 90)
        print("INDICADORES TÉCNICOS")
        print("=" * 90)
        print(
            f"Tickers calculados: {calculated}"
        )
        print(
            f"Dados completos: {complete}"
        )
        print(
            f"Falhas: {failures}"
        )
        print(
            f"Arquivo: {self.output_file}"
        )
        print("=" * 90)


# ============================================================
# FUNÇÃO SIMPLIFICADA
# ============================================================

def calculate_technical_indicators(
    price_history: pd.DataFrame,
    save: bool = True,
) -> pd.DataFrame:
    """
    Interface simplificada para calcular os indicadores.

    Exemplo
    -------
    indicators = calculate_technical_indicators(
        price_history
    )
    """

    engine = TechnicalIndicators()

    return engine.calculate(
        price_history=price_history,
        save=save,
    )


# ============================================================
# EXECUÇÃO DIRETA
# ============================================================

if __name__ == "__main__":

    from config.settings import PRICE_FILE

    if not PRICE_FILE.exists():
        raise FileNotFoundError(
            "O arquivo historico_precos.csv não foi encontrado. "
            "Execute primeiro o market_data.py."
        )

    historical_prices = pd.read_csv(
        PRICE_FILE
    )

    indicator_engine = TechnicalIndicators()

    indicator_engine.calculate(
        historical_prices
    )

    print("\nÚltimo registro por ticker:")

    columns_to_display = [
        "ticker",
        "date",
        "close",
        "rsi_14",
        "macd_hist",
        "adx_14",
        "mfi_14",
        "atr_percentual",
        "distancia_maxima_52s",
        "volume_relativo_20d",
        "indicadores_completos",
    ]

    available_columns = [
        column
        for column in columns_to_display
        if column
        in indicator_engine.get_latest().columns
    ]

    print(
        indicator_engine
        .get_latest()[
            available_columns
        ]
        .to_string(
            index=False
        )
    )
