from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from docx import Document
from docx.shared import Inches

from model_pipeline import DATA_PATH, MODEL_PATH, build_model, load_clean_data, train_and_save


BASE_DIR = Path(__file__).resolve().parent
REPORT_PATH = BASE_DIR / "SudhanshuRanjan_ProjectReport.docx"
CHART_PATH = BASE_DIR / "model_performance.png"
ASSET_DIR = BASE_DIR / "report_assets"
DATASET_URL = "https://www.kaggle.com/datasets/medhekarabhinav5/indian-cars-dataset?select=cars_ds_final_2021.csv"


def save_chart(name):
    path = ASSET_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def add_chart(document, title, path):
    document.add_heading(title, level=2)
    document.add_picture(str(path), width=Inches(6.3))


def main():
    ASSET_DIR.mkdir(exist_ok=True)
    data = load_clean_data(DATA_PATH)
    if not MODEL_PATH.exists():
        train_and_save()
    pipeline, metrics, split, features = build_model(data)
    _, y_test, predictions = split

    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, predictions, alpha=0.55, color="#176b87")
    lower, upper = min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())
    plt.plot([lower, upper], [lower, upper], "--", color="#d95f02")
    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")
    plt.title("Random Forest: actual vs predicted price")
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=160, bbox_inches="tight")
    plt.close()

    numeric = data.select_dtypes(include="number")
    charts = {}

    plt.figure(figsize=(8, 5))
    sns.histplot(data=data, x="Ex-Showroom_Price", bins=30, color="#176b87")
    plt.title("Distribution of ex-showroom prices")
    plt.xlabel("Ex-showroom price")
    charts["price_distribution"] = save_chart("price_distribution.png")

    missing = data.isna().sum().sort_values(ascending=False).head(12)
    missing = missing[missing > 0]
    plt.figure(figsize=(8, 5))
    if len(missing):
        sns.barplot(x=missing.values, y=missing.index, color="#d95f02")
        plt.xlabel("Missing values")
        plt.ylabel("")
        plt.title("Columns with the most missing values")
    else:
        plt.text(0.5, 0.5, "No missing values after cleaning", ha="center", va="center")
        plt.axis("off")
    charts["missing_values"] = save_chart("missing_values.png")

    correlation = numeric.corr(numeric_only=True)
    top_correlation = correlation["Ex-Showroom_Price"].abs().sort_values(ascending=False).head(12).index
    plt.figure(figsize=(9, 7))
    sns.heatmap(numeric[top_correlation].corr(), cmap="RdBu_r", center=0, annot=False)
    plt.title("Correlation heatmap of numeric features")
    charts["correlation"] = save_chart("correlation_heatmap.png")

    box_columns = [column for column in ["Ex-Showroom_Price", "Displacement", "Power", "Torque", "Kerb_Weight"] if column in numeric]
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=numeric[box_columns], orient="h", color="#4c9f70")
    plt.title("Outlier review for selected numeric columns")
    charts["outliers"] = save_chart("outlier_boxplots.png")

    comparison = pd.DataFrame({
        "Model": ["Random Forest", "Linear Regression"],
        "R2": [0.979115, 0.968772],
        "MAPE (%)": [8.474366, 12.392255],
    })
    plt.figure(figsize=(8, 5))
    sns.barplot(data=comparison, x="Model", y="R2", color="#176b87")
    plt.ylim(0, 1.05)
    plt.title("Notebook model comparison by R2")
    charts["model_comparison"] = save_chart("model_comparison.png")

    residuals = y_test - predictions
    plt.figure(figsize=(8, 5))
    plt.scatter(predictions, residuals, alpha=0.55, color="#7b3294")
    plt.axhline(0, color="#333333", linestyle="--")
    plt.xlabel("Predicted price")
    plt.ylabel("Residual")
    plt.title("Residual plot for the saved API pipeline")
    charts["residuals"] = save_chart("residuals.png")

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importance = pd.Series(pipeline.named_steps["model"].feature_importances_, index=feature_names).sort_values(ascending=False).head(15)
    plt.figure(figsize=(9, 7))
    sns.barplot(x=importance.values, y=importance.index, color="#d95f02")
    plt.xlabel("Importance")
    plt.ylabel("")
    plt.title("Top feature importances from the saved API pipeline")
    charts["feature_importance"] = save_chart("feature_importance.png")

    document = Document()
    document.add_heading("Sudhanshu Ranjan - Car Price Prediction Report", 0)
    document.add_paragraph("Analysis of the 2021 car specification dataset and a Random Forest price predictor.")
    document.add_heading("Dataset and preparation", level=1)
    document.add_paragraph(f"Source file: {DATA_PATH.name}. After selecting the modeling fields and converting units embedded in text, the analysis contains {len(data):,} usable rows and {len(features)} input features.")
    document.add_paragraph(f"Dataset source: Indian Cars Dataset on Kaggle - {DATASET_URL}")
    document.add_paragraph("Missing numeric values are imputed with medians and categorical values with the most frequent category. Categorical specifications are one-hot encoded and unknown categories are accepted at prediction time.")
    document.add_heading("Model results", level=1)
    document.add_paragraph("The original exploratory notebook compared Random Forest with a Linear Regression baseline using the same train/test split and processed features. Its reported results are shown below.")
    table = document.add_table(rows=1, cols=2)
    table.style = "Light Shading Accent 1"
    table.rows[0].cells[0].text = "Metric"
    table.rows[0].cells[1].text = "Notebook comparison"
    notebook_metrics = {
        "Random Forest MAE": 149663.323727,
        "Random Forest RMSE": 320467.424072,
        "Random Forest MAPE (%)": 8.486981,
        "Random Forest R2": 0.978769,
        "Linear Regression MAE": 238985.060384,
        "Linear Regression RMSE": 388509.884028,
        "Linear Regression MAPE (%)": 12.390477,
        "Linear Regression R2": 0.968796,
    }
    for name, value in notebook_metrics.items():
        cells = table.add_row().cells
        cells[0].text = name
        cells[1].text = f"{value:,.4f}"
    document.add_paragraph("The notebook results round to R2 = 0.97 for Random Forest and R2 = 0.96 for Linear Regression. The saved API model is trained by the reusable pipeline and may produce different metrics because its preprocessing configuration is separate from the original notebook experiment.")
    document.add_picture(str(CHART_PATH), width=Inches(6.3))
    document.add_heading("Interpretation", level=1)
    document.add_paragraph("The model is intended as an estimate, not a valuation guarantee. Errors can be larger for rare luxury vehicles because the target price spans a wide range. A future version could compare gradient boosting, logarithmic target transformation, and cross-validation.")
    document.add_heading("Application dashboard", level=1)
    document.add_paragraph("The project includes a Streamlit dashboard named Indian Cars Intelligence. It presents the analysis as an interactive application instead of requiring the user to inspect notebook cells manually.")
    document.add_heading("Application screenshots", level=2)
    document.add_paragraph("The following screenshots show the live dashboard interface used for the project.")
    screenshot_paths = [
        ("Live dashboard overview", ASSET_DIR / "application_overview.png"),
        ("Live market explorer", ASSET_DIR / "application_market_explorer.png"),
        ("Live price predictor", ASSET_DIR / "application_predictor.png"),
    ]
    for title, screenshot_path in screenshot_paths:
        if screenshot_path.exists():
            add_chart(document, title, screenshot_path)
    document.add_heading("Overview screen", level=2)
    document.add_paragraph("The Overview screen gives a quick market snapshot: 1,267 cars analyzed, brand coverage, median ex-showroom price, the official notebook Random Forest R2 score, price distribution, leading brands by number of listings, and model-quality metrics.")
    add_chart(document, "Dashboard overview: price distribution", charts["price_distribution"])
    add_chart(document, "Dashboard overview: model comparison", charts["model_comparison"])
    document.add_heading("Market Explorer screen", level=2)
    document.add_paragraph("The Market Explorer screen provides filters for brand, body type, and fuel type. It shows median price by body type, the relationship between engine displacement and price, and a table of the highest-priced listings in the selected segment. Missing brand values are shown as Not specified rather than None.")
    add_chart(document, "Market Explorer evidence: feature relationships", charts["correlation"])
    document.add_heading("Predictor screen", level=2)
    document.add_paragraph("The Predictor screen accepts vehicle identity and specification values such as make, model, variant, displacement, seating capacity, kerb weight, fuel information, power, and torque. When the user submits the form, the saved Random Forest deployment pipeline returns an estimated ex-showroom price.")
    add_chart(document, "Predictor evidence: actual versus predicted price", CHART_PATH)
    document.add_heading("Application services", level=2)
    document.add_paragraph("Streamlit dashboard: http://localhost:8503")
    document.add_paragraph("Flask backend health check: http://127.0.0.1:5000/health")
    document.add_paragraph("Flask prediction endpoint: POST http://127.0.0.1:5000/predict")
    document.add_heading("Analysis diagrams", level=1)
    add_chart(document, "Price distribution", charts["price_distribution"])
    add_chart(document, "Missing-value review", charts["missing_values"])
    add_chart(document, "Correlation heatmap", charts["correlation"])
    add_chart(document, "Outlier review", charts["outliers"])
    add_chart(document, "Notebook model comparison", charts["model_comparison"])
    add_chart(document, "Actual versus predicted price", CHART_PATH)
    add_chart(document, "Residual analysis", charts["residuals"])
    add_chart(document, "Feature importance", charts["feature_importance"])
    document.add_heading("How to use the project", level=1)
    document.add_paragraph("Run `py model_pipeline.py` to train and save the model, `py app.py` to start the Flask API, or `streamlit run streamlit_app.py` for the interactive interface.")
    document.save(REPORT_PATH)
    print(f"Created {REPORT_PATH}")


if __name__ == "__main__":
    main()