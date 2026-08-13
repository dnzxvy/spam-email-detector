import kagglehub
import nltk
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression



# Downloading spam email classification
path = kagglehub.dataset_download("ashfakyeafi/spam-email-classification")

print("Path to dataset files:", path)

# loading spam email dataset
csv_path = path + "/SpamEmail Dataset.csv"


spam_dataset = pd.read_csv(csv_path)
print(spam_dataset.head())

spam_dataset = spam_dataset[
    spam_dataset["Category"].isin(["ham", "spam"])
]
spam_dataset = spam_dataset.reset_index(drop=True)

print(spam_dataset["Category"].value_counts())

# Bar chart to showcase value count of category
spam_dataset["Category"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90,
    ylabel="")

plt.title("Distribution of Email categories")
#plt.xlabel("Email Type")
#plt.ylabel("Number of Emails")

plt.show()

# Calculating the average email length

spam_dataset["email_length"] = spam_dataset["Message"].apply(len)
averageEmail_length = spam_dataset.groupby("Category")["email_length"].mean()
print(spam_dataset.columns)
print(averageEmail_length)

averageEmail_length.plot(kind="bar")

plt.title("Average Email length for Spam & Ham emails")
plt.xlabel("Email Type")
plt.ylabel("Email Length (Measured in characters)")

plt.show()

# Calculating frequent spam and ham words

spam_emails = spam_dataset[spam_dataset["Category"] == "spam"]
ham_emails = spam_dataset[spam_dataset["Category"] == "ham"]

spam_text = " ".join(spam_emails["Message"])
ham_text = " ".join(ham_emails["Message"]) # combining all spam and ham messages into
# a single string

spam_text = spam_text.lower()
ham_text = ham_text.lower()

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

spam_words = spam_text.split()
ham_words = ham_text.split()

spam_words = [word for word in spam_words if word not in stop_words]
ham_words = [word for word in ham_words if word not in stop_words]

spam_word_counts = Counter(spam_words)
ham_word_counts = Counter(ham_words)

print(spam_word_counts.most_common(20))
print(ham_word_counts.most_common(20))

# Plotting the most common ham and spam words

spam_df = pd.DataFrame(
    spam_word_counts.most_common(20),
    columns=["Word", "Count"]
)

ham_df = pd.DataFrame(
    ham_word_counts.most_common(20),
    columns=["Word","Count"]
)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
# spam words plot
axes[0].bar(
    spam_df["Word"],
    spam_df["Count"]
)

axes[0].set_title("Most frequent words in spam emails")
axes[0].set_xlabel("Words")
axes[0].set_ylabel("Frequency")
axes[0].tick_params(axis="x", rotation=45)

#plt.xticks(rotation=45)
#plt.show()

# ham words plot

axes[1].bar(
    ham_df["Word"],
    ham_df["Count"]
)

axes[1].set_title("Most frequent words in ham emails")
axes[1].set_xlabel("Words")
axes[1].set_ylabel("Frequency")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

spam_dataset["word_count"] = spam_dataset["Message"].apply(
    lambda x: len(x.split())
)

average_words = spam_dataset.groupby("Category")["word_count"].mean()
print(average_words)

# Punctuation Analysis

# Amount of exclamation marks within emails
cnt_spam = spam_text.count("!")
cnt_ham = ham_text.count("!")
print(f"the amount of times the character ! appeared within spam emails is: {cnt_spam}, and for ham emails it appeared: {cnt_ham} times")

spam_dataset["exclamation_count"] = spam_dataset["Message"].apply(lambda x: x.count("!"))

print(spam_dataset.groupby("Category")["exclamation_count"].mean())

spam_dataset.groupby("Category")["exclamation_count"].mean().plot(
    kind="bar"
)

plt.title("Average Number of Punctuation Marks per Email")
plt.xlabel("Email Category")
plt.ylabel("Average Number of Punctuation Marks")
plt.xticks(rotation=0)

plt.show()

# Amount of question marks within emails
cnt_qm_spam = spam_text.count("?")
cnt_qm_ham = ham_text.count("?")
print(f"the amount of times the character ? appeared within spam emails is: {cnt_qm_spam}, and for ham emails it appeared: {cnt_qm_ham} times")

spam_dataset["question_mark_count"] = spam_dataset["Message"].apply(lambda x: x.count("?"))

print(spam_dataset.groupby("Category")["question_mark_count"].mean())

spam_dataset.groupby("Category")["question_mark_count"].mean().plot(
    kind="bar"
)

plt.title("Average Number of Question Marks per Email")
plt.xlabel("Email Category")
plt.ylabel("Average Number of Question Marks")
plt.xticks(rotation=0)

plt.show()


# Number Analysis

spam_dataset["number_count"] = spam_dataset["Message"].apply(
    lambda x: sum(char.isdigit() for char in x)
)

average_numbers = spam_dataset.groupby("Category")["number_count"].mean()
print(average_numbers)

average_numbers.plot(kind="bar")

plt.title("Average Number of Numerical Digits per Email")
plt.xlabel("Category")
plt.ylabel("Average Number of Digits")

plt.xticks(rotation=0)

plt.show()

# box plot
spam_dataset.boxplot(
    column="number_count",
    by="Category"
)

plt.title("Distribution of Numerical Values by Email Category")
plt.suptitle("")
plt.xlabel("Category")
plt.ylabel("Number Count")

plt.show()

# Length of numerical sequences

def longest_number_length(text):
    numbers = re.findall(r"\d+", text) #\d = any digit and + = one or more
    #consecutive digits

    if not numbers:
        return 0
    return max(len(number)for number in numbers)

spam_dataset["longest_number_length"] = spam_dataset["Message"].apply(longest_number_length)
average_number_length = spam_dataset.groupby("Category")["longest_number_length"].mean()
print(average_number_length)

average_number_length.plot(kind="bar")

plt.title("Average Length of the Longest Numerical Sequence")
plt.xlabel("Category")
plt.ylabel("Average Number of Digits")

plt.xticks(rotation=0)

plt.show()

# Symbol Analysis

spam_dataset["percent_count"] = spam_dataset["Message"].apply(
    lambda x: x.count("%") # Making a count of percentage symbol shown in emails

)

average_percent = spam_dataset.groupby("Category")["percent_count"].mean()

print(average_percent)

# Making a count of dollar symbol shown in emails

spam_dataset["dollar_count"] = spam_dataset["Message"].apply(
    lambda x: x.count("$")

)

average_dollar = spam_dataset.groupby("Category")["dollar_count"].mean()

print(average_dollar)

# Making a count of pound (sterling) symbol shown in emails

spam_dataset["pound_count"] = spam_dataset["Message"].apply(
    lambda x: x.count("£")

)
average_pound = spam_dataset.groupby("Category")["pound_count"].mean()
print(average_pound)

# Plot displaying average symbols including punctuation mark symbols (!, ?)

spam_dataset["exclamation_count"] = spam_dataset["Message"].apply(lambda x: x.count("!"))
spam_dataset["question_mark_count"] = spam_dataset["Message"].apply(lambda x: x.count("?"))
spam_dataset["percent_count"] = spam_dataset["Message"].apply(lambda x: x.count("%"))
spam_dataset["pound_count"] = spam_dataset["Message"].apply(lambda x: x.count("£"))
spam_dataset["dollar_count"] = spam_dataset["Message"].apply(lambda x: x.count("$"))

symbol_analysis = spam_dataset.groupby("Category")[[
    "exclamation_count",
    "question_mark_count",
    "percent_count",
    "pound_count",
    "dollar_count"
]].mean()

symbol_analysis.plot(kind="bar")

plt.title("Average Promotional Symbol Usage by Email Category")
plt.xlabel("Category")
plt.ylabel("Average Count per Email")
plt.xticks(rotation=0)

plt.legend(title="Symbol")

plt.show()


# URL Analysis

def count_urls(text):
    urls = re.findall(r"https?://\S+|www\.\S+", text) # finds different type of url in emails
    return len(urls)

spam_dataset["url_count"] = spam_dataset["Message"].apply(count_urls)

average_urls = spam_dataset.groupby("Category")["url_count"].mean()
print(average_urls)

average_urls.plot(kind = "bar")

plt.title("Average count of URLs within Spam & Ham Emails")
plt.xlabel("Category")
plt.ylabel("Average Count of URLs")
plt.xticks(rotation=0)

plt.show()

# Correlation Analysis

correlation_features = spam_dataset[[
    "word_count",
    "email_length",
    "url_count",
    "exclamation_count",
    "question_mark_count",
    "number_count",
    "longest_number_length",
    "percent_count",
    "dollar_count",
    "pound_count",
]]

# Calculating correlation matrix

correlation_matrix = correlation_features.corr()
print(correlation_matrix)

# Plotting correlation matrix

plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap="coolwarm", interpolation="nearest")
plt.colorbar(label="Correlation")
plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=45,
    ha="right"
)
plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)
for i in range(len(correlation_matrix)):
    for j in range(len(correlation_matrix.columns)):
        plt.text(
            j,
            i,
            f"{correlation_matrix.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            color="black"
        )
plt.title("Correlation Matrix of Email Features")
plt.tight_layout()
plt.show()


##### Machine Learning Phase ########

# Preparing the dataset
X = spam_dataset["Message"]
y = spam_dataset["Category"]

# Turn labels into numerical values

encoder = LabelEncoder()
y = encoder.fit_transform(y)

#print(encoder.classes_)
#print(y[:10]) # checking if encoder code worked supposed to have ham as 0
# and spam as 1

# COnvert text into numbers via tf-idf

tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
X = tfidf.fit_transform(X)

print(spam_dataset["Category"].value_counts())

print(encoder.classes_)



# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y)


# Training the model via naive bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Generating predictions for the model
prediction = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, prediction)
print(f"Accuracy: {accuracy:.2%}")

print(classification_report(y_test, prediction))

### Testing a Second Model using Logisitc Regression

logistic_model = LogisticRegression(max_iter=1000)
# max_iter gives the algorithm 1000 iterations to find
# appropriate solutions

# Train model
logistic_model.fit(X_train,y_train)

# Making predictions for Logistic Regression Model
logistic_predictions = logistic_model.predict(X_test)

# Calculating the Accuracy
logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)
print(f"Logistic Regression Accuracy: {logistic_accuracy:.2%}")

# Classification Report for Logistic Regression
print(classification_report(y_test, logistic_predictions))

#spam_dataset.to_csv("SpamEmail Dataset V2", index=False)

