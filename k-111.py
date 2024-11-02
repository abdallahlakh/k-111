import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree

# Load the data
df = pd.read_csv('data.csv')

# Preprocess the data (example: encoding categorical variables)
df['previous_job_title'] = df['previous_job_title'].astype('category').cat.codes
df['programming_languages'] = df['programming_languages'].apply(lambda x: len(x.split(',')))
df['database_experience'] = df['database_experience'].astype('category').cat.codes
df['networking_experience'] = df['networking_experience'].map({'Yes': 1, 'No': 0})
df['cybersecurity_experience'] = df['cybersecurity_experience'].map({'Yes': 1, 'No': 0})
df['cloud_computing_experience'] = df['cloud_computing_experience'].astype('category').cat.codes
df['devops_tools_used'] = df['devops_tools_used'].astype('category').cat.codes
df['project_management_experience'] = df['project_management_experience'].astype('category').cat.codes
df['future_interests'] = df['future_interests'].astype('category').cat.codes
df['current_specialty'] = df['current_specialty'].astype('category').cat.codes
df['happy_in_role'] = df['happy_in_role'].map({'Yes': 1, 'No': 0})
df['finds_difficulty'] = df['finds_difficulty'].map({'Yes': 1, 'No': 0})

# Define features and target
X = df.drop(['current_specialty', 'happy_in_role', 'finds_difficulty'], axis=1)
y = df['current_specialty']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Visualize the decision tree
tree.plot_tree(clf, feature_names=X.columns, class_names=df['current_specialty'].astype('category').cat.categories, filled=True)
