"""
Agri-Demand Optimizer — MCP Server
Model Context Protocol server exposing demand forecasting tools.

Connects any LLM-based agent to the Agri-Demand Optimizer's data layer,
enabling conversational queries like:
  "Which products have the highest predicted demand next week?"
  "Compare week-1 vs week-2 forecasts for leafy greens."

Usage:
  python mcp_farm.py                  # Start the MCP server
  python mcp_farm.py --port 8100      # Custom port

Architecture:
  Agent (LLM)  →  MCP Client  →  This Server  →  Model / Data Layer
"""

import json
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# Simulated data layer — replace with live model calls in production
# ---------------------------------------------------------------------------

PRODUCTS = [
    {"id": "P001", "name": "Sukuma Wiki (Kale)",    "category": "Leafy Greens"},
    {"id": "P002", "name": "Spinach",               "category": "Leafy Greens"},
    {"id": "P003", "name": "Tomatoes",               "category": "Fruit Vegetables"},
    {"id": "P004", "name": "Onions",                 "category": "Root Vegetables"},
    {"id": "P005", "name": "Carrots",                "category": "Root Vegetables"},
    {"id": "P006", "name": "Coriander (Dhania)",     "category": "Herbs"},
    {"id": "P007", "name": "Capsicum",               "category": "Fruit Vegetables"},
    {"id": "P008", "name": "Potatoes",               "category": "Root Vegetables"},
    {"id": "P009", "name": "Cabbage",                "category": "Leafy Greens"},
    {"id": "P010", "name": "Green Beans",            "category": "Legumes"},
]

BUYERS = [
    {"id": "B001", "name": "Nairobi Hospital Kitchen",   "category": "Institutional"},
    {"id": "B002", "name": "Java House Procurement",     "category": "Restaurant Chain"},
    {"id": "B003", "name": "Tuskys Supermarket",         "category": "Retail"},
    {"id": "B004", "name": "Kenyatta University Mess",   "category": "Institutional"},
    {"id": "B005", "name": "Mama Oliech Restaurant",     "category": "Restaurant"},
]

# Simulated demand forecasts (buyer_id, product_id) → predictions
FORECASTS = {}
import random
random.seed(42)
for b in BUYERS:
    for p in PRODUCTS:
        buy_prob_1w = round(random.uniform(0.15, 0.95), 3)
        qty_1w = round(random.uniform(2, 80), 1) if buy_prob_1w > 0.4 else 0
        buy_prob_2w = round(min(buy_prob_1w + random.uniform(-0.1, 0.15), 0.99), 3)
        qty_2w = round(qty_1w * random.uniform(0.85, 1.25), 1) if qty_1w > 0 else 0
        FORECASTS[(b["id"], p["id"])] = {
            "buyer": b["name"],
            "product": p["name"],
            "category": p["category"],
            "week_1": {"purchase_probability": buy_prob_1w, "predicted_qty_kg": qty_1w},
            "week_2": {"purchase_probability": buy_prob_2w, "predicted_qty_kg": qty_2w},
            "surplus_reduction_kg": round(qty_1w * 0.28, 1) if qty_1w > 0 else 0,
        }


# ---------------------------------------------------------------------------
# MCP Tool definitions
# ---------------------------------------------------------------------------

TOOLS = {
    "list_products": {
        "description": "List all tracked product categories and items.",
        "parameters": {},
    },
    "list_buyers": {
        "description": "List all registered buyers and their categories.",
        "parameters": {},
    },
    "query_demand": {
        "description": "Query predicted demand for a specific buyer and product. "
                       "Returns 1-week and 2-week purchase probabilities and quantities.",
        "parameters": {
            "buyer_id": {"type": "string", "description": "Buyer ID (e.g. B001)"},
            "product_id": {"type": "string", "description": "Product ID (e.g. P001)"},
        },
    },
    "top_products": {
        "description": "Rank products by predicted order volume for a given week horizon.",
        "parameters": {
            "week": {"type": "integer", "description": "1 or 2 (forecast week)"},
            "limit": {"type": "integer", "description": "Number of results (default 5)"},
        },
    },
    "waste_estimate": {
        "description": "Estimate surplus food reduction (kg) if model-guided ordering "
                       "is adopted for a specific buyer or across all buyers.",
        "parameters": {
            "buyer_id": {"type": "string", "description": "Buyer ID, or 'all' for portfolio-wide"},
        },
    },
    "compare_weeks": {
        "description": "Compare week-1 vs week-2 demand forecasts for a product category.",
        "parameters": {
            "category": {"type": "string", "description": "Product category name"},
        },
    },
}


def execute_tool(name, params):
    if name == "list_products":
        return PRODUCTS

    elif name == "list_buyers":
        return BUYERS

    elif name == "query_demand":
        key = (params.get("buyer_id", ""), params.get("product_id", ""))
        forecast = FORECASTS.get(key)
        if not forecast:
            return {"error": f"No forecast found for buyer={key[0]}, product={key[1]}"}
        return forecast

    elif name == "top_products":
        week = params.get("week", 1)
        limit = params.get("limit", 5)
        week_key = "week_1" if week == 1 else "week_2"
        ranked = {}
        for (_, pid), fc in FORECASTS.items():
            pname = fc["product"]
            ranked[pname] = ranked.get(pname, 0) + fc[week_key]["predicted_qty_kg"]
        sorted_products = sorted(ranked.items(), key=lambda x: x[1], reverse=True)[:limit]
        return [{"product": p, "total_predicted_kg": round(q, 1)} for p, q in sorted_products]

    elif name == "waste_estimate":
        bid = params.get("buyer_id", "all")
        total = 0
        count = 0
        for (b, _), fc in FORECASTS.items():
            if bid == "all" or b == bid:
                total += fc["surplus_reduction_kg"]
                count += 1
        return {
            "buyer": bid,
            "estimated_weekly_surplus_reduction_kg": round(total, 1),
            "pairs_covered": count,
        }

    elif name == "compare_weeks":
        cat = params.get("category", "")
        results = []
        for (_, _), fc in FORECASTS.items():
            if fc["category"].lower() == cat.lower():
                results.append({
                    "buyer": fc["buyer"],
                    "product": fc["product"],
                    "week_1_qty": fc["week_1"]["predicted_qty_kg"],
                    "week_2_qty": fc["week_2"]["predicted_qty_kg"],
                    "delta": round(fc["week_2"]["predicted_qty_kg"] - fc["week_1"]["predicted_qty_kg"], 1),
                })
        return results if results else {"error": f"No products found in category '{cat}'"}

    return {"error": f"Unknown tool: {name}"}


# ---------------------------------------------------------------------------
# HTTP handler — MCP-style JSON-RPC endpoint
# ---------------------------------------------------------------------------

class MCPHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if self.path == "/tools":
            self._respond({"tools": TOOLS})
        elif self.path == "/execute":
            tool_name = body.get("tool")
            tool_params = body.get("parameters", {})
            result = execute_tool(tool_name, tool_params)
            self._respond({"result": result})
        else:
            self._respond({"error": "Unknown endpoint"}, 404)

    def do_GET(self):
        if self.path == "/health":
            self._respond({"status": "ok", "server": "agri-demand-mcp", "tools": list(TOOLS.keys())})
        else:
            self._respond({"error": "Use POST /tools or POST /execute"}, 405)

    def _respond(self, data, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())

    def log_message(self, format, *args):
        print(f"[MCP-Farm] {args[0]}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agri-Demand MCP Server")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()

    server = HTTPServer(("0.0.0.0", args.port), MCPHandler)
    print(f"[MCP-Farm] Agri-Demand MCP Server running on http://localhost:{args.port}")
    print(f"[MCP-Farm] Tools: {list(TOOLS.keys())}")
    server.serve_forever()
