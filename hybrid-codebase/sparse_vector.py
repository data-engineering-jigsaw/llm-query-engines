from sklearn.feature_extraction.text import TfidfVectorizer

file_path = './data/ww1.txt'
with open(file_path, 'r') as file:
    content = file.read()

chunks = content.split('.')

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(chunks)


print(vectorizer.get_feature_names_out())