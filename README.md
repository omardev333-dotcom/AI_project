# AI Smart Study Advisor 🎓

An educational machine-learning prototype that estimates a student's possible final grade from study-related inputs and provides practical study suggestions.

## Features

- Grade estimation using a `RandomForestRegressor`.
- Model evaluation using MAE, RMSE, and R² on a held-out test split.
- Simple performance categories.
- Rule-based, explainable study recommendations.
- Interactive Streamlit interface.
- Reproducible synthetic demonstration dataset.

## Important data and ethics note

The included dataset is **synthetic** and was generated for software-learning purposes. It does not contain real student records. The model learns a simulated relationship created by `generate_data.py`; therefore, its metrics and predictions must not be presented as evidence of real-world academic accuracy.

Before any real-world use, replace the demo data with an appropriately licensed, representative, anonymized dataset, document its source and limitations, check for bias, and obtain appropriate consent and oversight. Do not use predictions to make high-stakes decisions about students.

## Project structure

```text
AI-Smart-Study-Advisor/
├── app.py
├── advisor.py
├── generate_data.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── student_performance_demo.csv
└── models/
    ├── metrics.json
    └── student_grade_model.joblib
```

## Run locally

Python 3.10 or newer is recommended.

1. Clone or download this repository.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Generate the demo data (optional; the CSV is already included):

   ```bash
   python generate_data.py
   ```

4. Train and evaluate the model:

   ```bash
   python train_model.py
   ```

5. Start the app:

   ```bash
   streamlit run app.py
   ```

The app opens in your browser, usually at `http://localhost:8501`.

## Machine-learning workflow

1. Generate/load the dataset.
2. Separate input features from the target (`final_grade`).
3. Split records into training and test sets.
4. Train a Random Forest regression model.
5. Evaluate predictions with MAE, RMSE, and R².
6. Use the trained model for an estimate and a separate recommendation layer.

## Inputs

- Daily study hours
- Attendance percentage
- Previous grade
- Assignment average
- Sleep hours
- Number of previous failed subjects

## Future improvements

- Replace synthetic data with a properly sourced real dataset.
- Add cross-validation and compare multiple regression models.
- Add feature-importance analysis and fairness checks.
- Support multiple subjects and time-based progress tracking.
- Add multilingual support.

## License

No license has been selected yet. Add one only after deciding how you want others to reuse the project.
