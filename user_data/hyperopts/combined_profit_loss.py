from freqtrade.optimize.hyperopt import IHyperOptLoss
from datetime import datetime
import math
from pandas import DataFrame, date_range

class CombinedProfitLoss(IHyperOptLoss):
    """
    Defines a loss function for hyperopt that combines Sharpe, Sortino, and profit loss functions.
    Aims to maximize profits and minimize losses.
    """

    @staticmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int,
                               min_date: datetime, max_date: datetime,
                               *args, **kwargs) -> float:
        """
        Objective function, returns a combined score based on Sharpe, Sortino, and profit loss functions.

        A higher combined score is better, indicating higher profits and lower risk.
        """
        resample_freq = '1D'
        slippage_per_trade_ratio = 0.0005
        days_in_year = 365
        annual_risk_free_rate = 0.0
        risk_free_rate = annual_risk_free_rate / days_in_year
        minimum_acceptable_return = 0.0

        # Calculate Sharpe Ratio
        results.loc[:, 'profit_ratio_after_slippage'] = \
            results['profit_ratio'] - slippage_per_trade_ratio

        t_index = date_range(start=min_date, end=max_date, freq=resample_freq,
                             normalize=True)

        sum_daily = (
            results.resample(resample_freq, on='close_date').agg(
                {"profit_ratio_after_slippage": 'sum'}).reindex(t_index).fillna(0)
        )

        total_profit = sum_daily["profit_ratio_after_slippage"] - risk_free_rate
        expected_returns_mean = total_profit.mean()
        up_stdev = total_profit.std()

        if up_stdev != 0:
            sharpe_ratio = expected_returns_mean / up_stdev * math.sqrt(days_in_year)
        else:
            sharpe_ratio = -20.

        # Calculate Sortino Ratio
        results.loc[:, 'profit_ratio_after_slippage'] = \
            results['profit_ratio'] - slippage_per_trade_ratio

        t_index = date_range(start=min_date, end=max_date, freq=resample_freq,
                             normalize=True)

        sum_daily = (
            results.resample(resample_freq, on='close_date').agg(
                {"profit_ratio_after_slippage": 'sum'}).reindex(t_index).fillna(0)
        )

        total_profit = sum_daily["profit_ratio_after_slippage"] - minimum_acceptable_return
        expected_returns_mean = total_profit.mean()

        sum_daily['downside_returns'] = 0
        sum_daily.loc[total_profit < 0, 'downside_returns'] = total_profit
        total_downside = sum_daily['downside_returns']
        down_stdev = math.sqrt((total_downside**2).sum() / len(total_downside))

        if down_stdev != 0:
            sortino_ratio = expected_returns_mean / down_stdev * math.sqrt(days_in_year)
        else:
            sortino_ratio = -20.

        # # Calculate Profit Loss
        profit_loss = results['profit_abs'].sum()

        # Combine Sharpe, Sortino, and Profit loss functions
        combined_score = -sharpe_ratio \
                        - profit_loss \
                        # - sortino_ratio \

        return combined_score
