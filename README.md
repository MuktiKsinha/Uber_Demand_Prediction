<h1 align="center">🚖 Uber Demand Prediction System</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python">
  <img src="https://img.shields.io/badge/Streamlit-App-red?logo=streamlit">
  <img src="https://img.shields.io/badge/Docker-Containerized-blue?logo=docker">
  <img src="https://img.shields.io/badge/AWS-EC2%20%7C%20ECR-orange?logo=amazonaws">
  <img src="https://img.shields.io/badge/GitHub-Actions-black?logo=githubactions">
  <img src="https://img.shields.io/badge/MLflow-Tracking-blue?logo=mlflow">
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
  <li>🌳 Tree-based regression models</li>
  <li>⚖️ Feature scaling & encoding pipelines</li>
  <li>🧪 Hyperparameter tuning using Optuna</li>
  <li>📉 Error-based optimization (RMSE, MAE)</li>
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

<pre>python src/data/make_dataset.py</pre>

<ul>
  <li>🧼 Removes duplicate & invalid records</li>
  <li>❓ Handles missing timestamps</li>
  <li>📍 Filters invalid geo-coordinates</li>
</ul>

<hr/>

<h2>📍 Feature Engineering</h2>

<pre>python src/features/build_features.py</pre>

<ul>
  <li>⏰ Temporal features (hour, weekday, weekend)</li>
  <li>📊 Lag & rolling demand features</li>
  <li>🗺️ Region-wise spatial clustering</li>
</ul>

<hr/>

<h2>🤖 Model Training</h2>

<pre>python src/models/train_model.py</pre>

<ul>
  <li>⚙️ Pipeline-based training</li>
  <li>🧪 Hyperparameter optimization</li>
  <li>📈 MLflow experiment tracking</li>
</ul>

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

<ul>
  <li>Same container runs locally & on AWS</li>
  <li>Ensures environment consistency</li>
</ul>

<hr/>

<h2>🔄 CI/CD Pipeline</h2>

<p>
🚀 Fully automated CI/CD using <strong>GitHub Actions</strong> and <strong>AWS</strong>
</p>

<ul>
  <li>Triggered on every git push</li>
  <li>Automated tests & model validation</li>
  <li>Docker image pushed to Amazon ECR</li>
  <li>Auto deployment to AWS EC2</li>
</ul>

<hr/>

<h2>🧾 CI/CD Architecture</h2>

<pre class="mermaid">
flowchart LR
    A[Developer Push] --> B[GitHub Repo]
    B --> C[GitHub Actions]
    C --> D[Tests & Build]
    D --> E[Docker Image]
    E --> F[AWS ECR]
    F --> G[AWS EC2]
    G --> H[Streamlit App Live]
</pre>

<hr/>

<h2>☁️ AWS Services</h2>

<table>
<tr><th>Service</th><th>Purpose</th></tr>
<tr><td>EC2</td><td>Streamlit hosting</td></tr>
<tr><td>ECR</td><td>Docker image registry</td></tr>
<tr><td>IAM</td><td>Secure access</td></tr>
<tr><td>AWS CLI</td><td>Deployment automation</td></tr>
</table>

<hr/>

<h2>🔐 Security Best Practices</h2>

<ul>
  <li>🔒 No hardcoded credentials</li>
  <li>🛡️ IAM-based access control</li>
  <li>🔑 GitHub Secrets</li>
  <li>📦 Isolated Docker containers</li>
</ul>

<hr/>

<h2>🌟 Key Highlights</h2>

<ul>
  <li>End-to-end demand forecasting system</li>
  <li>Production-grade MLOps workflow</li>
  <li>Cloud-deployed Streamlit dashboard</li>
  <li>Automated CI/CD pipeline</li>
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
🌟 If you enjoyed this project, give it a star!<br>
💡 Suggestions, feedback, or ideas are highly appreciated.
</p>
