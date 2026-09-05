---
title: auto-experimenter
emoji: 🔁
colorFrom: blue
colorTo: yellow
sdk: streamlit
sdk_version: 1.63.0
app_file: app.py
pinned: false
license: mit
short_description: Autonomous experiment loop for churn models
---

# auto-experimenter

An autonomous loop for churn models: a planner reads the ledger of everything tried so far and proposes the next experiments as testable hypotheses, the runner scores them under one fixed cross-validation protocol, a judge marks each hypothesis confirmed or refuted against the incumbent, and a narrator turns the numbers into a conclusion with next steps and retention actions.

- Dataset: IBM Telco Customer Churn (Kaggle), 7,043 customers, churn rate 26.5 %.
- Code and design notes: https://github.com/aimanelasad/auto-experimenter
- The saved runs shown on load were produced by the CLI and committed with the code. A live run with the rules planner takes about 20 seconds on the free CPU tier; the Claude planner is available when the Space has an `ANTHROPIC_API_KEY` secret.
