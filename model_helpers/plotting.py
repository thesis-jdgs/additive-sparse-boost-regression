"""Implement a functions to plot 1D trees."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from model_helpers.od_tree import ListTreeRegressor


def plot_continuous(
    regressor: ListTreeRegressor,
    x: np.ndarray,
    x_processed: np.ndarray,
) -> go.Figure:
    """Plot the score of the estimator, assuming a continuous feature.

    Parameters
    ----------
    regressor : ListTreeRegressor
        The estimator to plot.
    x : np.ndarray of shape (n_samples,)
        The original data, before preprocessing. It's used for plotting.
    x_processed : np.ndarray of shape (n_samples,)
        The processed data. It's used for prediction.

    Returns
    -------
    go.Figure
        The figure for the plot.
    """
    y = regressor.predict(x_processed)
    fig = make_subplots(
        rows=2,
        cols=2,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=(
            f"{regressor.output_name} prediction",
            "Contribution distribution",
        ),
        vertical_spacing=0.02,
        horizontal_spacing=0.02,
        column_widths=[0.8, 0.2],
    )

    fig.add_trace(px.line(x=x, y=y, line_shape="hv").data[0], row=1, col=1)
    fig.add_trace(
        go.Histogram(x=x, name=f"{regressor.feature_name} distribution"), row=2, col=1
    )
    fig.add_trace(go.Box(y=y, name=""), row=1, col=2)

    fig.update_layout(
        showlegend=False,
        template="plotly",
        title=f"Contributions are on the same scale as {regressor.output_name},"
        "<br>and centered vertically on the train set.",
    )
    fig.update_yaxes(title_text="Contribution", row=1, col=1)
    fig.update_xaxes(title_text=regressor.feature_name, row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    return fig


def plot_categorical(
    regressor: ListTreeRegressor,
    x: np.ndarray,
    x_processed: np.ndarray,
) -> go.Figure:
    """Plot the score of the estimator, assuming a categorical feature.

    Parameters
    ----------
    regressor : ListTreeRegressor
        The estimator to plot.
    x : np.ndarray of shape (n_samples,)
        The original data, before preprocessing. It's used for plotting.
    x_processed : np.ndarray of shape (n_samples,)
        The processed data. It's used for prediction.

    Returns
    -------
    go.Figure
        The figure for the plot.
    """
    y = regressor.predict(x_processed)
    fig = make_subplots(
        cols=2,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=(f"{regressor.output_name} prediction",),
        horizontal_spacing=0.02,
    )
    as_series = pd.Series(data=y, index=x)
    mean_grouped = as_series.groupby(level=0).mean()
    fig.add_trace(mean_grouped.plot.barh().data[0], row=1, col=1)
    fig.add_trace(
        go.Histogram(y=x, name=f"{regressor.feature_name} distribution"), row=1, col=2
    )
    fig.update_layout(
        showlegend=False,
        template="plotly",
        title=f"Contributions are on the same scale as {regressor.output_name},"
        "<br>and centered vertically on the train set.",
    )
    fig.update_xaxes(title_text="Contribution", row=1, col=1)
    fig.update_yaxes(title_text=regressor.feature_name, row=1, col=1)
    fig.update_xaxes(title_text="Count", row=1, col=2)

    return fig
