# mason.ai

Streamlit dashboard for monitoring equities with synthetic data fallbacks.  The app can
fetch live information from `yfinance` when the dependency is installed, yet it also
works fully offline thanks to the lightweight service layer implemented under `src/`.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # or install streamlit, pandas, numpy, plotly, yfinance
streamlit run app.py
```

The Streamlit UI provides tabs for live monitoring, order book exploration, alerts,
news, and historical analytics.  Environment variables such as `SENDGRID_API_KEY` and
`TWILIO_ACCOUNT_SID`/`TWILIO_AUTH_TOKEN` enable email or SMS alert toggles.

## Project layout

```
app.py              # Streamlit user interface
src/
  alerts.py         # simple threshold based alert helper
  anomaly_detection.py
  data_sources.py   # wraps yfinance with synthetic fallbacks
  database.py       # in-memory storage used by the dashboard
  news_scraper.py
  notifications.py
  order_book.py
  visualization.py  # chart helpers built on Plotly
```

The service classes keep the original prototype structure intact while ensuring the
repository runs as-is for new contributors.
