import os
import numpy as np
import pandas as pd

def generate_sample_sales_data(output_path: str = "data/sample_sales.csv") -> None:
    """ Generate a synthetic retail sales dataset for KPI analysis.
    
    Columns:
    - date
    - product_id
    - category
    - region
    - channel
    - sales
    - units_sold
    - promotion_flag
    - discount_rate
    - campaign_name
    - promotion_cost
    """

    np.random.seed(42)

    dates = pd.date_range(start="2025-01-01", end="2025-03-31", freq='D')
    categories = ['Skincare', 'Makeup', 'Haircare', 'Fragrance']
    regions = ['North', 'South', 'East', 'West']
    channels = ['Shopee', 'Amazon', 'Retail', 'Brand.com']

    rows = []

    for date in dates:
        for category in categories:
            for region in regions:
                product_id = f"{category[:3].upper()}-{region[:1]}"

                base_units = {
                    "Skincare": 120,
                    "Makeup": 95,
                    "Haircare": 80,
                    "Fragrance": 60,
                }[category]

                region_multiplier = {
                    "North": 1.10,
                    "South": 0.95,
                    "East": 1.05,
                    "West": 0.90,
                }[region]

                promotion_flag = np.random.choice([0, 1], p=[0.72, 0.28])

                discount_rate = 0.0
                campaign_name = "No Campaign"
                promotion_cost = 0.0         

                if promotion_flag == 1:
                    discount_rate = np.random.choice([0.10, 0.15, 0.20, 0.25])
                    campaign_name = np.random.choice(
                        ["Spring Promo", "Flash Sale", "Member Day", "Bundle Deal"]
                    )
                    promotion_cost = np.random.uniform(200, 1200)

                noise = np.random.normal(loc=0, scale=12)

                uplift_multiplier = 1.0
                if promotion_flag == 1:
                    uplift_multiplier += discount_rate * np.random.uniform(1.5, 3.5)

                units_sold = max(
                    0,
                    int(base_units * region_multiplier * uplift_multiplier + noise),
                )

                unit_price = {
                    "Skincare": 28,
                    "Makeup": 22,
                    "Haircare": 18,
                    "Fragrance": 45,
                }[category]

                sales = units_sold * unit_price * (1 - discount_rate)

                rows.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "product_id": product_id,
                        "category": category,
                        "region": region,
                        "channel": np.random.choice(channels),
                        "sales": round(sales, 2),
                        "units_sold": units_sold,
                        "promotion_flag": promotion_flag,
                        "discount_rate": discount_rate,
                        "campaign_name": campaign_name,
                        "promotion_cost": round(promotion_cost, 2),
                    }
                )

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Sample data saved to: {output_path}")
    print(df.head())


if __name__ == "__main__":
    generate_sample_sales_data()      