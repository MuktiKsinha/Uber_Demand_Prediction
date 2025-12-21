<h1 align="center">🚖 Uber Demand Prediction System</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python">
  <img src="https://img.shields.io/badge/Streamlit-App-red?logo=streamlit">
  <img src="https://img.shields.io/badge/Docker-Containerized-blue?logo=docker">
  <img src="https://img.shields.io/badge/AWS-EC2%20%7C%20ECR-orange?logo=amazonaws">
  <img src="https://img.shields.io/badge/GitHub-Actions-black?logo=githubactions">
  <img src="https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue?logo=mlflow">
</p>

<p align="center">
A <strong>production-ready Uber Demand Prediction System</strong> that forecasts
<strong>ride demand</strong> using <strong>spatio-temporal features</strong> and
<strong>machine learning models</strong>, deployed using <strong>Streamlit</strong>,
<strong>Docker</strong>, <strong>AWS</strong>, and a fully automated
<strong>CI/CD pipeline</strong>.
</p>

<hr/>

<h2>🎥 Project Demo</h2>

<p>
Watch the end-to-end demo of the <strong>Uber Demand Prediction System</strong>
showcasing interactive demand maps and real-time predictions:
</p>

<p align="center">
  <a href="https://youtu.be/YOUR_DEMO_LINK">
    <img src="assets/demo_thumbnail.png" width="700"/>
  </a>
</p>

<p align="center">▶️ Click the image to watch the demo on YouTube</p>

<hr/>

<h2>🚀 Project Overview</h2>

<p>
This project builds an <strong>end-to-end demand forecasting system</strong> for
Uber-like ride-hailing platforms. It predicts
<strong>ride demand across locations and time windows</strong> using historical trip data.
</p>

<p>
The system follows modern <strong>MLOps & deployment best practices</strong>,
including experiment tracking, reproducible pipelines, CI/CD automation,
and cloud-native deployment.
</p>

<hr/>

<h2>🧠 Demand Prediction Techniques</h2>

<h3>📍 Spatio-Temporal Feature Engineering</h3>
<ul>
  <li>📌 Pickup latitude & longitude clustering</li>
  <li>⏰ Hour, day-of-week, weekend indicators</li>
  <li>📅 Seasonal & holiday-based features</li>
  <li>📊 Lag features & rolling demand statistics</li>
</ul>

<h3>📈 Supervised Machine Learning</h3>
<ul>
  <li>🌳 Regression-based forecasting models</li>
  <li>⚖️ Feature scaling & encoding pipelines</li>
  <li>🧪 Hyperparameter tuning using Optuna</li>
  <li>📉 Error-based optimization using MAPE</li>
</ul>

<h3>🔀 Final Demand Forecast</h3>
<p>
The final model predicts demand per region and time window, enabling
<strong>better fleet allocation</strong>, <strong>reduced passenger wait time</strong>,
and <strong>optimized surge pricing</strong>.
</p>

<hr/>

<h2>🗂️ Project Structure</h2>

<pre>
├── data/
│   ├── raw/
│   ├── processed/
│   └── interim/
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── Feature_Engineering.ipynb
│   └── Model_Training.ipynb
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── visualization/
│
├── models/
├── reports/
│
├── app.py
├── Dockerfile
├── requirements.txt
├── .github/workflows/ci-cd.yml
└── README.md
</pre>

<hr/>

<h2>⚙️ Tech Stack</h2>

<p>
  <img src="https://skillicons.dev/icons?i=python,docker,aws,github,sklearn" />
</p>

<ul>
  <li>Python, Pandas, NumPy</li>
  <li>Scikit-learn, Dask</li>
  <li>Streamlit</li>
  <li>Docker & GitHub Actions</li>
  <li>AWS (EC2, ECR, IAM)</li>
</ul>

<hr/>

<h2>🧹 Data Processing</h2>

<pre>python src/data/data_ingestion.py</pre>

<ul>
  <li>🧼 Removes duplicate & invalid records</li>
  <li>❓ Handles missing timestamps</li>
  <li>📍 Filters invalid geo-coordinates</li>
</ul>

<hr/>

<h2>📍 Feature Engineering</h2>

<pre>python src/features/extract_features.py</pre>
<pre>python src/features/feature_processing.py</pre>

<ul>
  <li>⏰ Temporal features (hour, weekday, weekend)</li>
  <li>📊 Lag & rolling demand features</li>
  <li>🗺️ Region-wise spatial clustering</li>
</ul>

<hr/>

<h2>🤖 Model Training</h2>

<pre>python src/models/train.py</pre>

<ul>
  <li>⚙️ Pipeline-based training</li>
  <li>🧪 Hyperparameter optimization using Optuna</li>
  <li>📈 MLflow experiment tracking</li>
</ul>

<hr/>

<h2>📊 Model Evaluation</h2>

<p>
The model is evaluated using
<strong>MAPE (Mean Absolute Percentage Error)</strong>,
which measures prediction accuracy in percentage terms and
is well-suited for demand forecasting problems.
</p>

<h3>📘 MAPE Formula</h3>

<pre>
MAPE = (1 / n) × Σ | (Actual − Predicted) / Actual | × 100
</pre>

<h3>🧪 Validation Strategy</h3>

<ul>
  <li>⏳ Time-based train–validation split</li>
  <li>🚫 Prevents future data leakage</li>
  <li>📊 Evaluated across multiple temporal windows</li>
</ul>

<h3>📈 MLflow Experiment Tracking</h3>

<ul>
  <li>🔍 MAPE logged as the primary evaluation metric</li>
  <li>📦 Parameters and artifacts tracked</li>
  <li>🏆 Best model selected based on minimum MAPE</li>
</ul>

<h3>📊 Best Model Performance</h3>

<pre>
Best MAPE : 7.93%
Model     : Linear Regression
</pre>

<hr/>

<h2>🎛️ Streamlit Application</h2>

<h3>▶️ Run Locally</h3>

<pre>streamlit run app.py --server.port 8000</pre>

<p><strong>Access:</strong> http://localhost:8000</p>

<hr/>

<h2>🐳 Dockerized Application</h2>

<pre>
docker build -t uber-demand-prediction .
docker run -p 8000:8000 uber-demand-prediction
</pre>

<hr/>

<h2>🔄 CI/CD Pipeline</h2>

<ul>
  <li>Triggered on every git push</li>
  <li>Automated testing & model validation</li>
  <li>Docker image pushed to Amazon ECR</li>
  <li>Auto deployment to AWS EC2</li>
</ul>

<hr/>

<h2>🚧 Future Improvements</h2>

<ul>
  <li>Real-time demand prediction</li>
  <li>Deep learning-based forecasting</li>
  <li>Monitoring & alerting</li>
  <li>Kubernetes deployment</li>
</ul>

<hr/>

<h2>⭐ Support & Feedback</h2>

<p>
🌟 If you enjoyed this project, give it a star!<br/>
💡 Suggestions, feedback, or ideas are highly appreciated.
</p>
