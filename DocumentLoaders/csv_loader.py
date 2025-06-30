from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='DocumentLoaders/Social_Network_Ads.csv')

docs = loader.load()
print(docs[0])