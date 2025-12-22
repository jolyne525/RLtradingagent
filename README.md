# 🤖 RL-Quant-Trader: Reinforcement Learning for Algorithmic Trading

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📖 Project Overview

This project demonstrates a **Reinforcement Learning (RL)** agent designed for automated stock trading. Built with **Streamlit** for interactivity, the agent utilizes **Q-Learning** with linear function approximation to make trading decisions (Buy, Sell, Hold) based on market state representations.

The system simulates a Markov Decision Process (MDP) on real historical data (via `yfinance`), aiming to maximize the **Sharpe Ratio** and **Total Returns** while managing transaction costs.

> **Note**: This project serves as a demonstration of Quantitative Analysis and AI application skills for academic applications (HKU/HKUST/CUHK).

## 🚀 Key Features

* **Interactive Training**: Visualize the Q-Learning process in real-time as the agent iterates through episodes.
* **Real-World Data**: Fetches real-time stock data (e.g., NVDA, AAPL) using the Yahoo Finance API.
* **Performance Metrics**: Automatically calculates key financial metrics for CVs:
    * **Alpha** (Excess Return vs. Benchmark)
    * **Sharpe Ratio** (Risk-adjusted Return)
    * **Cumulative Return**
* **Strategy Comparison**: Benchmarks the RL agent against a standard "Buy & Hold" strategy.
* **Interpretability**: Uses a linear approximation for Q-values to maintain transparency in decision logic.

## 🧮 Methodology (MDP Formulation)

The trading problem is modeled as an MDP $(S, A, P, R)$:

* **State Space ($S$)**: A vector representing market conditions:
    $$S_t = [\text{Daily Return}, \text{Holding Status}, \text{Bias}]$$
* **Action Space ($A$)**: Discrete actions $\{0: \text{Hold}, 1: \text{Buy}, 2: \text{Sell}\}$.
* **Reward Function ($R$)**:
    $$R_t = \Delta \text{NetWorth} - \text{Transaction Cost}$$
    The agent is penalized for excessive trading to simulate realistic friction costs.

## 💻 Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/YourUsername/RL-Quant-Trader.git](https://github.com/YourUsername/RL-Quant-Trader.git)
cd RL-Quant-Trader