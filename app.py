import kagglehub
import nltk
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re

# Downloading spam email classification
path = kagglehub.dataset_download("ashfakyeafi/spam-email-classification")

print("Path to dataset files:", path)

# loading spam email dataset
csv_path = path + "/SpamEmail Dataset.csv"

spam_dataset = pd.read_csv(csv_path)
print(spam_dataset.head())

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

#spam_dataset.to_csv("SpamEmail Dataset V2", index=False)
