# 🌦️ Aussie Rain Predictor

A machine learning web application built with Streamlit that predicts whether it will rain tomorrow in Australia based on current weather observations.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://app-hw-v1.streamlit.app/)

## 📋 Overview

This application uses a pre-trained machine learning model to predict the likelihood of rain in Australia based on daily weather data from the Australian Bureau of Meteorology. The model analyzes 22 different weather features including temperature, humidity, wind speed, pressure, and more to make accurate predictions.

## ✨ Features

- 🎯 **Real-time Predictions** - Get instant rain predictions with probability scores
- 🎲 **Random Input Generation** - Quickly test the model with randomized weather data
- 🔄 **Reset Functionality** - Easily reset inputs to median/default values
- 📊 **Interactive UI** - User-friendly sliders and dropdowns for all weather parameters
- 📈 **Probability Display** - See the exact probability percentage of rain tomorrow
- 📖 **Feature Descriptions** - Built-in documentation explaining all weather parameters

## 🎬 Live Demo

Try the live application here: **[https://app-hw-v1.streamlit.app/](https://app-hw-v1.streamlit.app/)**

## 🏗️ Project Structure

```
streamlit-hw/
├── .devcontainer/
│   └── devcontainer.json     # VS Code Dev Container configuration
├── data/
│   └── weatherAUS.csv        # Australian weather dataset
├── models/
│   └── aussie_rain.joblib    # Pre-trained ML model bundle
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## 🔧 Model Components

The `aussie_rain.joblib` file contains a complete model pipeline with:
- **Trained classifier** - Machine learning model for rain prediction
- **Imputer** - Handles missing values in numeric features
- **Scaler** - Normalizes numeric features
- **OneHotEncoder** - Encodes categorical features
- **Feature lists** - Column names for numeric and categorical features

## 📊 Input Features

### Numeric Features (16)
- **MinTemp / MaxTemp** - Minimum and maximum temperature (°C)
- **Rainfall** - Total rainfall measured (mm)
- **Evaporation** - Amount of water evaporated (mm)
- **Sunshine** - Hours of bright sunshine
- **WindGustSpeed** - Maximum wind gust speed (km/h)
- **WindSpeed9am / WindSpeed3pm** - Wind speeds at 9am/3pm (km/h)
- **Humidity9am / Humidity3pm** - Relative humidity at 9am/3pm (%)
- **Pressure9am / Pressure3pm** - Atmospheric pressure at 9am/3pm (hPa)
- **Cloud9am / Cloud3pm** - Cloud coverage at 9am/3pm (0-9 scale)
- **Temp9am / Temp3pm** - Temperature at 9am/3pm (°C)

### Categorical Features (5)
- **Location** - Weather station location in Australia
- **WindGustDir** - Direction of the strongest wind gust
- **WindDir9am / WindDir3pm** - Wind direction at 9am/3pm
- **RainToday** - Whether it rained today (Yes/No)

Wind directions include: N, NNE, NE, ENE, E, ESE, SE, SSE, S, SSW, SW, WSW, W, WNW, NW, NNW

## 🚀 Installation & Usage

### Prerequisites
- Python 3.11 or higher
- pip package manager

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/Mytros/streamlit-hw.git
cd streamlit-hw
```

2. **Create virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

### Using Dev Container

This project includes a Dev Container configuration for easy setup in VS Code or GitHub Codespaces:

1. Open the project in VS Code
2. Install the "Dev Containers" extension
3. Press `F1` and select "Dev Containers: Reopen in Container"
4. The app will start automatically on port 8501

## 📦 Dependencies

```
streamlit>=1.38      # Web application framework
pandas>=2.0          # Data manipulation
numpy>=1.24          # Numerical computing
scikit-learn>=1.3    # Machine learning library
joblib>=1.3          # Model serialization
```

## 🎮 How to Use

1. **Adjust Weather Parameters**
   - Use sliders to set numeric values (temperature, humidity, wind speed, etc.)
   - Use dropdowns to select categorical values (location, wind direction, etc.)

2. **Quick Actions**
   - Click "🎲 Randomize Inputs" to generate random weather data
   - Click "🔄 Reset Inputs" to restore default values

3. **Get Prediction**
   - Click "🔮 Predict RainTomorrow" to get the prediction
   - View the result (Yes/No) and probability percentage
   - Expand "Input details" to see the exact values used

## 🧠 Model Information

The model was trained on the **WeatherAUS** dataset from the Australian Bureau of Meteorology, containing daily weather observations from numerous Australian weather stations. The dataset includes historical weather data with features used to predict rainfall on the following day.

### Preprocessing Pipeline
1. **Imputation** - Missing numeric values are filled using training statistics
2. **Scaling** - Numeric features are standardized
3. **Encoding** - Categorical features are one-hot encoded
4. **Prediction** - The processed data is fed to the classifier

### Prediction Threshold
- Probability ≥ 50% → "Yes" (Rain expected)
- Probability < 50% → "No" (No rain expected)

## 🌐 Deployment

The application is deployed on **Streamlit Community Cloud** and is accessible at:
**[https://app-hw-v1.streamlit.app/](https://app-hw-v1.streamlit.app/)**

To deploy your own version:
1. Fork this repository
2. Sign up at [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Deploy from your forked repository

## 📝 Data Source

The WeatherAUS dataset is sourced from the Australian Bureau of Meteorology and contains approximately 10 years of daily weather observations from various locations across Australia.

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is created as an educational assignment. Please refer to your course guidelines for usage and distribution terms.

## 👤 Author

**Mytros**
- GitHub: [@Mytros](https://github.com/Mytros)
- Project: [streamlit-hw](https://github.com/Mytros/streamlit-hw)

## 🙏 Acknowledgments

- Australian Bureau of Meteorology for the weather data
- Streamlit team for the amazing framework
- scikit-learn for machine learning tools

