/**
 * Client for the Agri-Demand Optimizer (Kenya Fresh Produce).
 * Facilitates predictive procurement for commercial kitchens.
 */
class AgriDemandClient {
    constructor(apiUrl = "https://okechobonyo-farm-to-feed-elite.hf.space/predict_demand") {
        this.apiUrl = apiUrl;
    }

    async getForecast(customerCategory, skuUnit, lagQty, freq4w, peakQty) {
        const payload = {
            customer_category: customerCategory,
            unit_name: skuUnit,
            grade_name: "GRADE_01",
            qty_lag_1w: lagQty,
            freq_4w: freq4w,
            freq_8w: freq4w * 2,
            freq_12w: freq4w * 3,
            pair_max_qty: peakQty,
            pair_avg_qty: (lagQty + peakQty) / 2
        };

        try {
            const response = await fetch(this.apiUrl, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!response.ok) throw new Error(`HTTP Error: ${response.status}`);
            return await response.json();
        } catch (error) {
            return { status: "error", message: error.message };
        }
    }
}

// --- Usage Example ---
// const client = new AgriDemandClient();
// const report = await client.getForecast("CUST_CAT_001", "UNIT_004", 12.5, 3, 25.0);
// console.log(`14-Day Outlook: ${report.predictions.week_2.predicted_quantity} units`);
