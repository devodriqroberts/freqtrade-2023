# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IntParameter, IStrategy, merge_informative_pair)

# --------------------------------
# Add your lib to import here
import random
from datetime import datetime
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from functools import reduce
from itertools import permutations

random.seed(datetime.now().timestamp())

# Training AWS Instance: 
# # Compute Optimized: c7g.2xlarge - 8 vCPU, 16 GB Mem
# # Accelerated Computing: g5.xlarge - 4 vCPU, 16 GB Mem, 1 GPU, 24 GB GPU Mem

def indicator_permutations(profiles, max_indicators=1, include_none=True):
    profile_permutations = set()
    if include_none:
        profile_permutations.add("NONE")

    if max_indicators == 1:
        profile_permutations.update(profiles)
        return profile_permutations

    for i in range(1, len(profiles)+1):
        for perm in permutations(profiles, i):
            if len(perm) <= max_indicators:
                profile_permutations.add(", ".join(sorted(list(perm))))
    return profile_permutations


class AwesomeStrategy(IStrategy):
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = '1h'

    # Can this strategy go short?
    can_short: bool = False

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }

    @property
    def plot_config(self):
        return {
            # Main plot indicators (Moving averages, ...)
            'main_plot': {
                'tema': {},
                'sar': {'color': 'white'},
            },
            'subplots': {
                # Subplots - each dict defines one additional plot
                "MACD": {
                    'macd': {'color': 'blue'},
                    'macdsignal': {'color': 'orange'},
                },
                "RSI": {
                    'rsi': {'color': 'red'},
                }
            }
        }

    # EMA
    ema_profiles = {
        "quick_trend": {
            "fast": 5,
            "slow": 10
        },
        "short_term": {
            "fast": 10,
            "slow": 20
        },
        "intraday": {
            "fast": 15,
            "slow": 30
        },
        "medium_term": {
            "fast": 20,
            "slow": 50
        },
        "reversal_continuation": {
            "fast": 9,
            "slow": 21
        },
        "ultra_fast": {
            "fast": 3,
            "slow": 8
        },
        "longer_term": {
            "fast": 50,
            "slow": 200
        }
    }

    ema_periods_slow    = set([ma_profile_setting["slow"] for ma_profile_setting in ema_profiles.values()])
    ema_periods_fast    = set([ma_profile_setting["fast"] for ma_profile_setting in ema_profiles.values()])
    ma_periods          = ema_periods_slow.union(ema_periods_fast)

    # Exponential Moving Average (EMA) -- Leading
    ema_profile_names   = list(ema_profiles.keys())
    buy_ema             = CategoricalParameter(ema_profile_names, default=random.choice(ema_profile_names), space="buy", optimize=True)
    sell_ema            = CategoricalParameter(ema_profile_names, default=random.choice(ema_profile_names), space="sell", optimize=True)


    # Average Directional Index (ADX) -- Lagging
    buy_adx_strength    = IntParameter(low=20, high=50, default=random.randint(20, 50), space="buy", optimize=True)
    sell_adx_strength   = IntParameter(low=20, high=50, default=random.randint(20, 50), space="sell", optimize=True)

    # Moving Average Bounce -- Lagging
    ma_bounce_periods   = [f"ema{period}" for period in ma_periods]
    buy_enable_mab      = CategoricalParameter([True, False], default=random.choice([True, False]), space="buy", optimize=True)
    buy_mab             = CategoricalParameter(ma_bounce_periods, default=random.choice(ma_bounce_periods), space="buy", optimize=True)


    # Relative Strength Index (RSI) -- Leading
    rsi_periods         = list(range(8, 15))
    buy_rsi_period      = IntParameter(min(rsi_periods), max(rsi_periods), default=random.randint(8, 14), space="buy", optimize=True)
    sell_rsi_period     = IntParameter(min(rsi_periods), max(rsi_periods), default=random.randint(8, 14), space="sell", optimize=True)
    buy_rsi_oversold    = IntParameter(low=10, high=50, default=random.randint(10, 50), space="buy", optimize=True)
    sell_rsi_overbought = IntParameter(low=50, high=90, default=random.randint(50, 90), space="sell", optimize=True)
    buy_rsi_cross       = CategoricalParameter(["ABOVE", "BELOW"], default=random.choice(["ABOVE", "BELOW"]), space="buy", optimize=True)
    sell_rsi_cross      = CategoricalParameter(["ABOVE", "BELOW"], default=random.choice(["ABOVE", "BELOW"]), space="sell", optimize=True)
    

    # Moving Average Convergence Divergence (MACD) -- Lagging
    macd_profiles = {
        "balanced_default": {
            "fast": 12,
            "slow": 26,
            "signal": 9
        },
        "quick_intraday": {
            "fast": 5,
            "slow": 13,
            "signal": 5
        },
        "short_term_trend": {
            "fast": 8,
            "slow": 21,
            "signal": 8
        },
        "ultra_short_term": {
            "fast": 3,
            "slow": 10,
            "signal": 3
        },
        "volatile_markets": {
            "fast": 9,
            "slow": 18,
            "signal": 9
        },
    }

    macd_profile_names  = list(macd_profiles.keys())
    buy_macd_profile    = CategoricalParameter(macd_profile_names, default=random.choice(macd_profile_names), space="buy", optimize=True)
    sell_macd_profile   = CategoricalParameter(macd_profile_names, default=random.choice(macd_profile_names), space="sell", optimize=True)
    

    # Commodity Channel Index (CCI) -- Lagging
    buy_cci_oversold    = IntParameter(low=-200, high=-100, default=random.randint(-200, -100), space="buy", optimize=True)
    sell_cci_overbought = IntParameter(low=100, high=200, default=random.randint(100, 200), space="sell", optimize=True)
    

    # Bollinger Bands
    bollinger_profiles = {
        "standard_default": {
            "window": 21,
            "stds": 2
        },
        "narrow_range_focus": {
            "window": 10,
            "stds": 1
        },
        "volatility_expansion": {
            "window": 30,
            "stds": 2.5
        },
        "trend_confirmation": {
            "window": 50,
            "stds": 2
        }
    }

    bollinger_profile_names = list(bollinger_profiles.keys())
    buy_bollinger_profile   = CategoricalParameter(bollinger_profile_names, default=random.choice(bollinger_profile_names), space="buy", optimize=True)
    sell_bollinger_profile  = CategoricalParameter(bollinger_profile_names, default=random.choice(bollinger_profile_names), space="sell", optimize=True)
    buy_bollinger_cross     = CategoricalParameter(["ABOVE", "BELOW"], default=random.choice(["ABOVE", "BELOW"]), space="buy", optimize=True)
    sell_bollinger_cross    = CategoricalParameter(["ABOVE", "BELOW"], default=random.choice(["ABOVE", "BELOW"]), space="sell", optimize=True)

    # Chaikin Volatility Indicator
    chaikin_profiles = {
        "standard_default": {
            "atr_period": 14,
            "sma_period": 10,
            "threshold": {
                "low": 0.10,
                "high": 0.20
            }
        },
        "higher_atr": {
            "atr_period": 20,
            "sma_period": 10,
            "threshold": {
                "low": 0.10,
                "high": 0.20
            }
        },
        "short_term": {
            "atr_period": 7,
            "sma_period": 5,
            "threshold": {
                "low": 0.05,
                "high": 0.10
            }
        },
        "trend_confirmation": {
            "atr_period": 14,
            "sma_period": 20,
            "threshold": {
                "low": 0.15,
                "high": 0.30
            }
        },
        "volatility_breakout": {
            "atr_period": 5,
            "sma_period": 10,
            "threshold": {
                "low": 0.30,
                "high": 0.50
            }
        },
        "intraday_scalping": {
            "atr_period": 5,
            "sma_period": 3,
            "threshold": {
                "low": 0.05,
                "high": 0.10
            }
        },
        "conservative": {
            "atr_period": 14,
            "sma_period": 20,
            "threshold": {
                "low": 0.15,
                "high": 0.25
            }
        },
        "aggressive": {
            "atr_period": 7,
            "sma_period": 5,
            "threshold": {
                "low": 0.25,
                "high": 0.35
            }
        },
        "market_opening": {
            "atr_period": 10,
            "sma_period": 5,
            "threshold": {
                "low": 0.20,
                "high": 0.30
            }
        },
        "market_opening": {
            "atr_period": 14,
            "sma_period": 10,
            "threshold": {
                "low": 0.15,
                "high": 0.25
            }
        },
    }

    chaikin_profile_names   = list(chaikin_profiles.keys())
    buy_chaikin_profile     = CategoricalParameter(chaikin_profile_names, default=random.choice(chaikin_profile_names), space="buy", optimize=True)
    buy_chaikin_limit       = DecimalParameter(low=chaikin_profiles[buy_chaikin_profile.value]["threshold"]["low"], high=chaikin_profiles[buy_chaikin_profile.value]
                                         ["threshold"]["high"], default=chaikin_profiles[buy_chaikin_profile.value]["threshold"]["high"], space="buy", optimize=True)
    sell_chaikin_profile    = CategoricalParameter(chaikin_profile_names, default=random.choice(chaikin_profile_names), space="sell", optimize=True)
    sell_chaikin_limit      = DecimalParameter(low=chaikin_profiles[sell_chaikin_profile.value]["threshold"]["low"], high=chaikin_profiles[sell_chaikin_profile.value]
                                          ["threshold"]["high"], default=chaikin_profiles[sell_chaikin_profile.value]["threshold"]["high"], space="sell", optimize=True)

    # Volume Indicators
    # On-Balance Volume (OBV)
    obv_periods     = list(range(7, 51))
    buy_obv_window  = IntParameter(min(obv_periods), max(obv_periods), default=random.randint(7, 50), space="buy", optimize=True)
    sell_obv_window = IntParameter(min(obv_periods), max(obv_periods), default=random.randint(7, 50), space="sell", optimize=True)
    # buy_enable_obv  = CategoricalParameter([True, False], default=random.choice([True, False]), space="buy", optimize=True)
    # sell_enable_obv = CategoricalParameter([True, False], default=random.choice([True, False]), space="sell", optimize=True)


    # indicator_combinations = [
    #     "ADX and ICHIMOKU",
    #     "EMA CROSSOVER",
    #     "CHIAKIN and CCI",
    #     "CHIAKIN and RSI",
    #     "CHIAKIN and STOCH",
    #     "BOLLINGER and CCI",
    #     "BOLLINGER and RSI",
    #     "BOLLINGER and STOCH",
    #     "MACD and RSI"
    # ]

    # buy_indicator_combination  = CategoricalParameter(indicator_combinations, default=random.choice(indicator_combinations), space="buy", optimize=True)
    # sell_indicator_combination = CategoricalParameter(indicator_combinations, default=random.choice(indicator_combinations), space="sell", optimize=True)

    buy_enable_adx          = CategoricalParameter([True, False], default=random.choice([True, False]), space="buy", optimize=True)
    # buy_enable_bollinger    = CategoricalParameter([True, False], default=random.choice([True, False]), space="buy", optimize=True)
    buy_enable_cci          = CategoricalParameter([True, False], default=random.choice([True, False]), space="buy", optimize=True)
    buy_enable_ema          = CategoricalParameter([True, False], default=random.choice([True, False]), space="buy", optimize=True)
    buy_enable_ichimoku     = CategoricalParameter([True, False], default=random.choice([True, False]), space="buy", optimize=True)
    buy_enable_macd         = CategoricalParameter([True, False], default=random.choice([True, False]), space="buy", optimize=True)
    buy_enable_mab          = CategoricalParameter([True, False], default=random.choice([True, False]), space="buy", optimize=True)
    # buy_enable_rsi          = CategoricalParameter([True, False], default=random.choice([True, False]), space="buy", optimize=True)
    buy_enable_stoch        = CategoricalParameter([True, False], default=random.choice([True, False]), space="buy", optimize=True)
    buy_enable_obv          = CategoricalParameter([True, False], default=random.choice([True, False]), space="buy", optimize=True)
    buy_enable_sar          = CategoricalParameter([True, False], default=random.choice([True, False]), space="buy", optimize=True)
    buy_enable_hist_cross   = CategoricalParameter([True, False], default=random.choice([True, False]), space="buy", optimize=True)
    
    sell_enable_adx         = CategoricalParameter([True, False], default=random.choice([True, False]), space="sell", optimize=True)
    # sell_enable_bollinger    = CategoricalParameter([True, False], default=random.choice([True, False]), space="sell", optimize=True)
    sell_enable_cci         = CategoricalParameter([True, False], default=random.choice([True, False]), space="sell", optimize=True)
    sell_enable_ema         = CategoricalParameter([True, False], default=random.choice([True, False]), space="sell", optimize=True)
    sell_enable_ichimoku    = CategoricalParameter([True, False], default=random.choice([True, False]), space="sell", optimize=True)
    sell_enable_macd        = CategoricalParameter([True, False], default=random.choice([True, False]), space="sell", optimize=True)
    # sell_enable_rsi          = CategoricalParameter([True, False], default=random.choice([True, False]), space="sell", optimize=True)
    sell_enable_stoch       = CategoricalParameter([True, False], default=random.choice([True, False]), space="sell", optimize=True)
    sell_enable_obv         = CategoricalParameter([True, False], default=random.choice([True, False]), space="sell", optimize=True)
    sell_enable_sar         = CategoricalParameter([True, False], default=random.choice([True, False]), space="sell", optimize=True)




    # Position sizing parameters
    risk_per_trade = DecimalParameter(
        low=0.02, high=0.10, default=0.05, space='buy', optimize=True)

    # Confirmation signal parameters
    macd_histogram_threshold = DecimalParameter(
        low=0, high=0.2, default=0.05, space='buy', optimize=True)

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    # Define the parameter spaces
    max_epa = CategoricalParameter(
        [-1, 0, 1, 3, 5, 10], default=3, space="buy", optimize=True)

    @property
    def max_entry_position_adjustment(self):
        return self.max_epa.value

    # Define the parameter spaces
    cooldown_lookback           = IntParameter(low=2, high=48, default=random.randint(2, 48), space="protection", optimize=True)
    stop_duration               = IntParameter(low=12, high=200, default=random.randint(12, 200), space="protection", optimize=True)
    use_stop_protection         = BooleanParameter(default=True, space="protection", optimize=True)
    use_drawdown_protection     = BooleanParameter(default=True, space="protection", optimize=True)
    use_low_profit_protection   = BooleanParameter(default=True, space="protection", optimize=True)

    @property
    def protections(self):
        prot = []

        prot.append({
            "method": "CooldownPeriod",
            "stop_duration_candles": self.cooldown_lookback.value
        })
        if self.use_stop_protection.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period_candles": 24 * 3,
                "trade_limit": 4,
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": False
            })
        if self.use_drawdown_protection.value:
            prot.append({
                "method": "MaxDrawdown",
                "lookback_period_candles": 48,
                "trade_limit": 20,
                "stop_duration_candles": 12,
                "max_allowed_drawdown": 0.15
            })
        if self.use_low_profit_protection.value:
            prot.append({
                "method": "LowProfitPairs",
                "lookback_period_candles": 6,
                "trade_limit": 2,
                "stop_duration": 60,
                "required_profit": 0.02,
                "only_per_pair": False,
            })

        return prot

    def position_size(self, **kwargs):
        """
        Returns the maximum position size for the current pair and price.
        """
        risk_factor = self.risk_per_trade.value
        portfolio_value = self.wallets[self.base_currency]
        stop_loss_pct = abs(self.stoploss)
        entry_price = kwargs.get('price')

        if entry_price:
            position_size = (portfolio_value * risk_factor) / \
                (entry_price * stop_loss_pct)
            return position_size
        return 0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Trend-Following Indicators
        # Exponential Moving Average (EMA)
        for ma_period in self.ma_periods:
            dataframe[f'ema{ma_period}'] = ta.EMA(dataframe, timeperiod=ma_period)

        # Parabolic SAR
        dataframe['sar'] = ta.SAR(dataframe)

        # Ichimoku Cloud
        dataframe['tenkan_sen']     = ta.EMA(dataframe, timeperiod=9)
        dataframe['kijun_sen']      = ta.EMA(dataframe, timeperiod=26)
        dataframe['senkou_span_a']  = 0.5 * (dataframe['tenkan_sen'] + dataframe['kijun_sen'])
        dataframe['senkou_span_b']  = ta.EMA(dataframe, timeperiod=52)

        # Average Directional Index (ADX)
        dataframe['adx'] = ta.ADX(dataframe)

        # Momentum Indicators
        # Relative Strength Index (RSI)
        for rsi_period in self.rsi_periods:
            dataframe[f'rsi{rsi_period}'] = ta.RSI(dataframe, timeperiod=rsi_period)

        # Moving Average Convergence Divergence (MACD)
        for macd_profile_name, macd_profile_settings in self.macd_profiles.items():
            macd = ta.MACD(dataframe, fastperiod=macd_profile_settings["fast"],slowperiod=macd_profile_settings["slow"], signalperiod=macd_profile_settings["signal"])
            dataframe[f'macd_{macd_profile_name}'] = macd["macd"]
            dataframe[f'macdsignal_{macd_profile_name}'] = macd["macdsignal"]
            dataframe[f'macd_histogram_{macd_profile_name}'] = macd["macdhist"]

        # Stochastic Oscillator
        stoch_rsi = ta.STOCHRSI(dataframe)
        dataframe['fastd'] = stoch_rsi['fastd']
        dataframe['fastk'] = stoch_rsi['fastk']

        # # Commodity Channel Index (CCI)
        dataframe['cci'] = ta.CCI(dataframe)

        # Volatility Indicators

        # Bollinger Bands
        for bollinger_profile_name, bollinger_profile_settings in self.bollinger_profiles.items():
            bollinger_bands = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=bollinger_profile_settings["window"], stds=bollinger_profile_settings["stds"])
            dataframe[f'bb_upperband_{bollinger_profile_name}'] = bollinger_bands['upper']
            dataframe[f'bb_middleband_{bollinger_profile_name}'] = bollinger_bands['mid']
            dataframe[f'bb_lowerband_{bollinger_profile_name}'] = bollinger_bands['lower']

        # Chaikin Volatility Indicator
        for chaikin_profile_name, chaikin_profile_settings in self.chaikin_profiles.items():
            dataframe[f'chaikin_{chaikin_profile_name}'] = ta.ATR(dataframe, timeperiod=chaikin_profile_settings["atr_period"]) / ta.SMA(dataframe, timeperiod=chaikin_profile_settings["sma_period"])

        # On-Balance Volume (OBV)
        dataframe['obv'] = ta.OBV(dataframe)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Define the buy conditions
        conditions = []

        ### Trend-Following Indicators ###
        # EMA
        EMA = (qtpylib.crossed_above(dataframe[f"ema{self.ema_profiles[self.buy_ema.value]['fast']}"], dataframe[f"ema{self.ema_profiles[self.buy_ema.value]['slow']}"]))
        # Ichimoku Cloud
        ICHIMOKU = (dataframe['tenkan_sen'] > dataframe['kijun_sen']) & (dataframe['tenkan_sen'].shift(1) <= dataframe['kijun_sen'].shift(1))
        # Parabolic SAR
        SAR = (dataframe['close'] >= dataframe['sar'])
        # ADX
        ADX = (dataframe['adx'] > self.buy_adx_strength.value)
        # Moving Average Bounce
        MAB = (dataframe['close'].shift(1) <= dataframe[self.buy_mab.value].shift(1)) & (dataframe['close'] > dataframe[self.buy_mab.value])

        ### Momentum Indicators ###
        # RSI
        RSI_CROSS_ABOVE = (qtpylib.crossed_above(dataframe[f'rsi{self.buy_rsi_period.value}'], self.buy_rsi_oversold.value))
        RSI_CROSS_BELOW = (qtpylib.crossed_above(dataframe[f'rsi{self.buy_rsi_period.value}'], self.buy_rsi_oversold.value))
        # Stochastic Oscillator
        STOCH_CROSS_ABOVE = (dataframe['fastk'] > dataframe['fastd'])
        # MACD
        BULLISH_MACD = (qtpylib.crossed_above(dataframe[f"macd_{self.buy_macd_profile.value}"], dataframe[f"macdsignal_{self.buy_macd_profile.value}"])) | (dataframe[f"macd_{self.buy_macd_profile.value}"] > 0)
        # CCI
        CCI = (qtpylib.crossed_above(dataframe['cci'], self.buy_cci_oversold.value))

        ### Volatility Indicators ###
        # Bollinger Bands
        dataframe['bb_lowerband'] = dataframe[f'bb_lowerband_{self.buy_bollinger_profile.value}']
        BOLLINGER_CROSS_ABOVE = (qtpylib.crossed_above(dataframe['close'], dataframe['bb_lowerband']))
        BOLLINGER_CROSS_BELOW = (qtpylib.crossed_below(dataframe['close'], dataframe['bb_lowerband']))

        # Chaikin
        CHAIKIN = (dataframe[f'chaikin_{self.buy_chaikin_profile.value}'] > self.buy_chaikin_limit.value)

        # OBV
        OBV = (dataframe['obv'] > dataframe['obv'].rolling(window=self.buy_obv_window.value).mean())


        if self.buy_enable_adx.value:
            conditions.append(ADX)
        if self.buy_enable_cci.value:
            conditions.append(CCI)
        if self.buy_enable_ema.value:
            conditions.append(EMA)
        if self.buy_enable_ichimoku.value:
            conditions.append(ICHIMOKU)
        if self.buy_enable_macd.value:
            conditions.append(BULLISH_MACD)
        if self.buy_enable_mab.value:
            conditions.append(MAB)
        if self.buy_enable_stoch.value:
            conditions.append(STOCH_CROSS_ABOVE)
        if self.buy_enable_obv.value:
            conditions.append(OBV)
        if self.buy_enable_sar.value:
            conditions.append(SAR)

        if self.buy_bollinger_cross.value == "ABOVE":
            conditions.append(BOLLINGER_CROSS_ABOVE)
        if self.buy_bollinger_cross.value == "BELOW":
            conditions.append(BOLLINGER_CROSS_BELOW)

        if self.buy_rsi_cross.value == "ABOVE":
            conditions.append(RSI_CROSS_ABOVE)
        if self.buy_rsi_cross.value == "BELOW":
            conditions.append(RSI_CROSS_BELOW)

        # indicator_combination = self.buy_indicator_combination.value
        # if "EMA" in indicator_combination:
        #     conditions.append(EMA)
        # if "ICHIMOKU" in indicator_combination:
        #     conditions.append(ICHIMOKU)
        # if "SAR" in indicator_combination:
        #     conditions.append(SAR)
        # if "ADX" in indicator_combination:
        #     conditions.append(ADX)
        # # if "MAB" in indicator_combination:
        # #     conditions.append(MAB)
        # # if "RSI" in indicator_combination:
        #     # conditions.append(RSI_CROSS_BELOW)
        # if "STOCH" in indicator_combination:
        #     conditions.append(STOCH_CROSS_ABOVE)
        # if "MACD" in indicator_combination:
        #     conditions.append(BULLISH_MACD)
        # if "CCI" in indicator_combination:
        #     conditions.append(CCI)
        # # if "BOLLINGER" in indicator_combination:
        #     # conditions.append(BOLLINGER_CROSS_ABOVE)
        # if "CHAIKIN" in indicator_combination:
        #     conditions.append(CHAIKIN)

            
        # if "OBV" in indicator_combination:
        # # if self.buy_enable_obv.value:
        #     conditions.append(OBV)


        # Add confirmation signal: Check MACD histogram
        if self.buy_enable_hist_cross.value:
            conditions.append(qtpylib.crossed_above(dataframe[f'macd_histogram_{self.buy_macd_profile.value}'], self.macd_histogram_threshold.value))

        # Check that volume is not 0
        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe


    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Define the sell conditions
        conditions = []

        ### Trend-Following Indicators ###
        # EMA
        EMA = (qtpylib.crossed_below(dataframe[f"ema{self.ema_profiles[self.sell_ema.value]['fast']}"],  dataframe[f"ema{self.ema_profiles[self.sell_ema.value]['slow']}"]))
        # Ichimoku Cloud
        ICHIMOKU = (dataframe['close'] < dataframe['senkou_span_a']) & (dataframe['close'] < dataframe['senkou_span_b'])
        # Parabolic SAR
        SAR = (dataframe['close'] < dataframe['sar'])
        # ADX
        ADX = (dataframe['adx'] > self.sell_adx_strength.value)

        ### Momentum Indicators ###
        # MACD
        BEARISH_MACD = (qtpylib.crossed_below(dataframe[f"macd_{self.sell_macd_profile.value}"], dataframe[f"macdsignal_{self.sell_macd_profile.value}"])) | (dataframe[f"macd_{self.sell_macd_profile.value}"] < 0)
        # RSI
        RSI_CROSS_ABOVE = (qtpylib.crossed_above(dataframe[f'rsi{self.sell_rsi_period.value}'], self.sell_rsi_overbought.value))
        RSI_CROSS_BELOW = (qtpylib.crossed_below(dataframe[f'rsi{self.sell_rsi_period.value}'], self.sell_rsi_overbought.value))
        # Stochastic Oscillator
        STOCH_CROSS_BELOW = (dataframe['fastk'] < dataframe['fastd'])
        # CCI
        CCI = (qtpylib.crossed_below(dataframe['cci'], self.sell_cci_overbought.value))


        ### Volatility Indicators ###
        # Bollinger Bands
        dataframe['bb_upperband'] = dataframe[f'bb_upperband_{self.sell_bollinger_profile.value}']
        BOLLINGER_CROSS_ABOVE = (qtpylib.crossed_above(dataframe['close'], dataframe['bb_upperband']))
        BOLLINGER_CROSS_BELOW = (qtpylib.crossed_below(dataframe['close'], dataframe['bb_upperband']))

        # Chaikin
        CHAIKIN = (dataframe[f'chaikin_{self.sell_chaikin_profile.value}'] < self.sell_chaikin_limit.value)

        # OBV
        OBV = (dataframe['obv'] < dataframe['obv'].rolling(window=self.sell_obv_window.value).mean()) & (dataframe['close'] < dataframe['ema200'])

        if self.sell_enable_adx.value:
            conditions.append(ADX)
        if self.sell_enable_cci.value:
            conditions.append(CCI)
        if self.sell_enable_ema.value:
            conditions.append(EMA)
        if self.sell_enable_ichimoku.value:
            conditions.append(ICHIMOKU)
        if self.sell_enable_macd.value:
            conditions.append(BEARISH_MACD)
        if self.sell_enable_stoch.value:
            conditions.append(STOCH_CROSS_BELOW)
        if self.sell_enable_obv.value:
            conditions.append(OBV)
        if self.sell_enable_sar.value:
            conditions.append(SAR)

        if self.sell_bollinger_cross.value == "ABOVE":
            conditions.append(BOLLINGER_CROSS_ABOVE)
        if self.sell_bollinger_cross.value == "BELOW":
            conditions.append(BOLLINGER_CROSS_BELOW)

        if self.sell_rsi_cross.value == "ABOVE":
            conditions.append(RSI_CROSS_ABOVE)
        if self.sell_rsi_cross.value == "BELOW":
            conditions.append(RSI_CROSS_BELOW)

        # indicator_combination = self.sell_indicator_combination.value
        # if "EMA" in indicator_combination:
        #     conditions.append(EMA)
        # if "ICHIMOKU" in indicator_combination:
        #     conditions.append(ICHIMOKU)
        # if "SAR" in indicator_combination:
        #     conditions.append(SAR)
        # if "ADX" in indicator_combination:
        #     conditions.append(ADX)
        # if "RSI" in indicator_combination:
        #     conditions.append(RSI_CROSS_ABOVE)
        # if "STOCH" in indicator_combination:
        #     conditions.append(STOCH_CROSS_BELOW)
        # if "MACD" in indicator_combination:
        #     conditions.append(BEARISH_MACD)
        # if "CCI" in indicator_combination:
        #     conditions.append(CCI)
        # if "BOLLINGER" in indicator_combination:
        #     conditions.append(BOLLINGER_CROSS_BELOW)
        # if "CHAIKIN" in indicator_combination:
        #     conditions.append(CHAIKIN)
        # if "OBV" in indicator_combination:
        #     conditions.append(OBV)


        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1

        return dataframe

