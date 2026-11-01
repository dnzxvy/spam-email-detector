# Spam Email Detector

A machine learning web application that classifies email messages as **Spam** or **Ham (legitimate email)**. Built with Python, scikit-learn and Flask, this project compares multiple text classification models and provides a simple browser interface for testing new messages.

## Project overview

The aim of this project is to explore how machine learning can identify spam emails using the text of a message. I carried out exploratory data analysis, converted messages into numerical features using TF-IDF, trained and evaluated three classification algorithms, and integrated the best-performing model into a Flask web application.

## Features

* Paste an email into a web form and receive a Spam or Ham prediction.
* TF-IDF text vectorisation with up to **5,000 features**.
* Comparison of **Multinomial Naive Bayes, Logistic Regression and Linear Support Vector Machine (LinearSVC)**.
* SVM hyperparameter selection using **5-fold cross-validation**.
* Evaluation with accuracy, precision, recall, F1-score and confusion matrices.
* Additional testing on **30 manually written emails** to examine performance on new examples.

## Technologies used

* **Python** — data processing, machine learning and application backend
* **Pandas and NumPy** — dataset handling and analysis
* **scikit-learn** — TF-IDF, model training, tuning and evaluation
* **Matplotlib / Seaborn** — exploratory analysis and visualisation
* **Flask** — web application and prediction API
* **HTML, CSS and JavaScript** — user interface
* **Joblib** — saving and loading the trained model and vectoriser

## Dataset and pre-processing

The dataset contains **5,572 labelled messages**:

|Category|Messages|
|-|-:|
|Ham|4,825|
|Spam|747|
|**Total**|**5,572**|

Messages were converted into TF-IDF features, with a maximum vocabulary size of 5,000. The data was split into **80% training and 20% testing**, using stratification to preserve the class distribution. The target labels were encoded as `0` for Ham and `1` for Spam.

Exploratory analysis also examined message length, word count, punctuation, numbers, currency symbols and URLs to understand differences between spam and legitimate messages.

## Model comparison

|Model|Test accuracy|
|-|-:|
|Multinomial Naive Bayes|97.22%|
|Logistic Regression|96.14%|
|**Linear SVM (LinearSVC)**|**98.21%**|

**Linear SVM** achieved the highest test accuracy and was selected for the web application. On the held-out test set of **1,115 messages**, it achieved approximately **0.99 precision, 0.87 recall and 0.93 F1-score for the Spam class**.

I also used `GridSearchCV` with **5-fold cross-validation** to evaluate different values of the SVM regularisation parameter `C` (`0.01`, `0.1`, `1`, `10` and `100`). The best value was **`C=1`**, with a cross-validation F1-score of approximately **0.936**. Tuning did not improve the held-out test accuracy beyond 98.21%.

### Additional testing

To test the model beyond the original dataset, I evaluated it on **30 manually written emails** (15 Spam and 15 Ham).

|Metric|Result|
|-|-:|
|Accuracy|86.67%|
|Spam precision|1.00|
|Spam recall|0.73|
|Spam F1-score|0.85|

The model correctly classified all 15 Ham messages but missed 4 of the 15 Spam messages. This smaller test highlights a limitation: performance on newly written messages can be lower than on the original held-out dataset. Because the additional sample contains only 30 emails, these results should not be treated as a definitive measure of real-world performance.

## How the application works

1. The user pastes an email into the browser interface.
2. JavaScript sends the text to Flask's `POST /predict` endpoint as JSON.
3. The application loads the saved TF-IDF vectoriser and SVM model using Joblib.
4. The vectoriser transforms the new message into the same feature space used during training.
5. The model predicts `Spam` or `Ham`, and Flask returns the result to the webpage.

### Example API request

```bash
curl -X POST http://127.0.0.1:5000/predict \\\\
  -H "Content-Type: application/json" \\\\
  -d '{"email":"Congratulations! You have won a prize. Claim it now!"}'
```

Example response format:

```json
{"prediction": "Spam"}
```

*The example illustrates the API response format; predictions depend on the trained model.*

## Running the project locally

### 1\. Clone the repository

```bash
git clone https://github.com/dnzxvy/SpamEmailDetector.git
cd SpamEmailDetector
```

### 2\. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\\\\Scripts\\\\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3\. Install dependencies

If a `requirements.txt` file is included:

```bash
pip install -r requirements.txt
```

Otherwise, install the main packages:

```bash
pip install flask flask-cors joblib scikit-learn pandas numpy matplotlib seaborn
```

### 4\. Check the saved model files

The Flask app expects these files in the project root:

* `spam\\\_model.pkl`
* `tfidf\\\_vectorizer.pkl`

If these files are not included in the repository, run the training code to generate them before starting the app. Only load Joblib files from trusted sources.

### 5\. Start Flask

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser, paste an email and click **Check Email**.

## Project structure

```text
SpamEmailDetector/
├── app.py                    # Flask routes and prediction logic
├── spam\\\_model.pkl            # Saved Linear SVM model (generated during training)
├── tfidf\\\_vectorizer.pkl      # Saved TF-IDF vectoriser (generated during training)
├── templates/
│   └── index.html            # Web interface
├── static/
│   └── css/
│       └── style.css         # Webpage styling
└── README.md
```

*The structure above shows the main application files. The training script, dataset and `requirements.txt` may also be included depending on how the repository is organised.*

## Limitations and future improvements

* **Generalisation:** Expand evaluation to more independently sourced email datasets and more diverse spam styles.
* **False negatives:** Improve detection of spam messages that resemble ordinary emails.
* **Class imbalance:** Explore approaches that improve Spam recall without introducing too many false positives.
* **Explainability:** Investigate which words and features contribute most to predictions.
* **Deployment:** Package and deploy the application so it can be used without running Flask locally.

## Author

Developed by [dnzxvy](https://github.com/dnzxvy).



