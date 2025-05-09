# Final Report: Predicting and Reducing Equipment Energy Consumption

---

### 1. Approach to the Problem

This project aimed to build a predictive machine learning model to estimate equipment energy consumption based on environmental and temporal data. The process follows a standard ML pipeline:

#### a. Data Cleaning

* Executed in the `Data_cleaning.ipynb` notebook.
* Likely operations include:
  * Removing duplicates  
  * Checking for inconsistent types  
  * Flagging or removing outliers  
  * Initial assessment of missing data  
* This step ensured high-quality input before modeling.

#### b. Data Ingestion

* Source: `data/final_data.csv`
* Key Feature Engineering:
  * `hour_sin`, `hour_cos`: To capture daily patterns.  
  * `temp_diff`: Calculated as zone temperature minus outdoor temperature, a strong predictor of heating/cooling load.
* Split: 80% training, 20% testing. All splits are saved locally for reproducibility.

#### c. Data Transformation

* Strategy based on feature characteristics:
  * Symmetric nulls: Mean imputation  
  * Skewed nulls: Median imputation  
  * Clean features: Standard scaling only  

#### d. Model Training

* Trained Models:
  * Random Forest Regressor (with GridSearch)  
  * Lasso Regression (with alpha tuning)  
* Evaluation Metrics:
  * R²  
  * MAE  
  * MSE  

---

### 2. Key Insights from the Data

* Time of day is a critical factor in energy use — hence the conversion of hour into sine and cosine components.  
* The indoor-outdoor temperature difference (`temp_diff`) has significant predictive power.  
* Feature skewness requires tailored imputation to preserve statistical integrity.  
* Data standardization helps all models perform on equal footing, especially Lasso.  

---

### 3. Model Performance Evaluation

| Model         | R² Score | MAE   | MSE    |
| ------------- | -------- | ----- | ------ |
| Random Forest | 0.443    | 13.33 | 370.23 |
| Lasso         | 0.219    | —     | —      |

* The Random Forest model had the highest R² score of 0.443, meaning it could explain ~44.3% of the variance in equipment energy consumption.  
* MAE of 13.33 suggests that, on average, the model’s predictions deviate by 13.33 energy units (likely kWh or similar).  
* MSE of 370.23 highlights the squared average error, more sensitive to large deviations.  

**Best Model: Random Forest**

---

### 4. Recommendations for Reducing Equipment Energy Consumption

1. **Time-Based Scheduling**  
   Leverage the hour-based patterns captured in the model. Run high-consumption operations during hours predicted to require less HVAC support.

2. **Optimize Temperature Difference (`temp_diff`)**  
   * Improve insulation to reduce indoor-outdoor gradients.  
   * Implement pre-conditioning strategies using outdoor temperature forecasts.  

3. **Sensor Deployment for Feature Monitoring**  
   Use sensors to track real-time values for key features (zone humidity, outdoor temperature, etc.) to trigger proactive energy-saving interventions.

4. **Model-Informed Operational Policies**  
   * Use model predictions as input for control systems.  
   * Predict and limit consumption spikes by adjusting HVAC loads.  

---

### 5. Limitations of the Model

While we attempted to build a robust solution, it is important to acknowledge that our model is not ideal and its predictive performance is modest. The following are key limitations:

* **Moderate Predictive Power**: With an R² score of 0.443, the best model (Random Forest) explains less than half of the variance in energy consumption. External factors not captured in the dataset may have significant influence.  
* **Lack of Real-Time Inputs**: The model is trained on historical static data and may not perform well for dynamic, real-time decision making without integration into a live system.  
* **Feature Limitations**: Critical variables such as occupancy, appliance usage, or HVAC control settings were not included in the dataset and may limit prediction accuracy.  
* **Scalability Concerns**: The current model is trained on a static dataset and does not incorporate online learning or periodic retraining to adapt to system changes over time.  

---

### 6. Deployment and Usage

 Try the live prediction app:  
**[Streamlit Deployment](https://ds-intern-assignment-suraj-kumar-energy-consumption-prediction.streamlit.app/)**

To run this project locally:

```bash
# Clone the repository
git clone https://github.com/your-username/energy-consumption-prediction.git
cd energy-consumption-prediction

# (Optional) Create virtual environment
python -m venv venv


# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
