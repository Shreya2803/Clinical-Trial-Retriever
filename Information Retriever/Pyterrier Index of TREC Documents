indexer = pt.IterDictIndexer('/content/drive/MyDrive/TREC2023/Index', meta=["docno", "filename", "folder"], meta_lengths=[20, 256, 256])
indexref = indexer.index(docs)

# Search the index(Example)/This is not the actual query after summarization, this is just a demo
bm25 = pt.BatchRetrieve(indexref, wmodel="BM25")
query = "Congenital Adrenal Hyperplasia"
results = bm25.search(query)

print("Search Results:")
print(results.head())
