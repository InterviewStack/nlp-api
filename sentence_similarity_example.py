from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

sentence1 = ["Overfitting in machine learning occurs when a model learns the details and noise in the training data to the extent that it performs poorly on new data. It can be prevented by using techniques such as cross-validation, regularization, and pruning. Additionally, using more training data and simplifying the model can help mitigate overfittingOverfitting in machine learning occurs when a model learns the details and noise in the training data to the extent that it performs poorly on new data. It can be prevented by using techniques such as cross-validation, regularization, and pruning. Additionally, using more training data and simplifying the model can help mitigate overfitting"]
sentence2 = ["Overfitting happens when a machine learning model captures noise and details in the training data, leading to poor generalization on unseen data. Preventive measures include cross-validation, regularization techniques, pruning, using more data, and simplifying the model to reduce complexity."]
sentence3 = ["Overfitting is when a machine learning model is too simple and cannot capture the complexity of the data, resulting in poor performance on the training data. It can be prevented by adding more layers to the neural network, increasing the model’s parameters, and using a less complex dataset."]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embedding1 = model.encode(sentence1)
embedding2 = model.encode(sentence2)
embedding3 = model.encode(sentence3)

cos_sim1 = cosine_similarity(embedding1, embedding2)
cos_sim2 = cosine_similarity(embedding1, embedding3)

print(cos_sim1, cos_sim2)
