# Agri-Demand Optimizer
### Supply Chain Intelligence for Kenyan Fresh Produce

A predictive analytics system designed to coordinate supply and demand in Kenyan fresh produce networks. The system predicts buyer orders weeks in advance, simulating up to a 28% reduction in post-harvest surplus waste.

This repository contains the delta-modeling ML pipeline (CatBoost + LightGBM ensemble), the local presentation server, and a Model Context Protocol (MCP) server that allows supply chain managers to interact with forecast data via natural language.

---

## getting Started

### Prerequisites
- Python 3.10+
- `pip`

### 1. Installation
Clone this repository and navigate to the application folder to install the required dependencies:

```bash
git clone https://github.com/okech-christopher/farm-to-feed-quant.git
cd farm-to-feed-quant
pip install -r requirements.txt
```

*(Note: Ensure you have `catboost`, `lightgbm`, `pandas`, `scikit-learn`, `flask` or `fastapi` depending on your wrapper, and `mcp` installed).*

### 2. Running the System
If you want to run the core predictive models and data transformations locally:
```bash
python main.py
```
*(Check the source scripts for CLI arguments on toggling between model training and inference modes).*

### 3. Running the MCP Server (Conversational Analytics)
The system includes an MCP (Model Context Protocol) server. Instead of forcing stakeholders to look at static SQL dashboards, this server exposes the forecast data to LLM agents, allowing users to ask questions like *"Which leafy greens have the highest demand next week?"*

To start the MCP server:
```bash
python mcp_farm.py --port 8100
```
This server provides 6 tools for conversational querying: `list_products`, `list_buyers`, `query_demand`, `top_products`, `waste_estimate`, and `compare_weeks`.

## Live Demo & Stakeholder Report
- **Live Predictive Dashboard:** [Hugging Face Space](https://huggingface.co/spaces/okechobonyo/farm-to-feed-elite)
- **Stakeholder Report:** [View the interactive report](https://okech-christopher.github.io/farm_report.html)
