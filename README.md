titanic-ml-preprocessing/
│
├── data/
│   └── train.csv                # Raw Titanic dataset
│
├── cleaned_data/
│   └── titanic_cleaned.csv     # Final cleaned and processed dataset
│
├── scripts/
│   └── preprocess.py           # Python script to clean and preprocess the data
│
├── requirements.txt            # Python libraries required
└── README.md                   # Project documentation


---

🔧 Technologies Used

Python 🐍

Pandas 📊

NumPy 🔢

Scikit-learn ⚙️



---

📌 Preprocessing Steps

📂 Loaded the dataset using Pandas

🧼 Dropped columns with too much missing or irrelevant data (Cabin, Ticket, Name)

🔄 Filled missing values in Age (with median) and Embarked (with mode)

🔢 Encoded Sex and Embarked columns

⚖️ Scaled Age and Fare using StandardScaler

💾 Saved the cleaned dataset for use in model training



---

▶️ How to Run

1. Clone the repository or download the files.


2. Install required libraries:



pip install -r requirements.txt

3. Run the preprocessing script:



python scripts/preprocess.py

The output will be saved in the cleaned_data/ directory.


---

📘 What I Learned

The importance of clean, structured data before building models

Basic data wrangling with Pandas

How to handle missing values and categorical variables

Real-world steps in preparing data for AI/ML pipelines
# New-repository