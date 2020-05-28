# import json
# with open('data/data-content/content-data.json') as json_file:
#     data = json.load(json_file)
#
# f = open("sent2vec/data.txt", "a")
#
# for it in data:
#     f.write(it['title'].strip())
#     f.write("\n")
#     f.write(it['description'].strip())
#     f.write("\n")


import sent2vec
model = sent2vec.Sent2vecModel()
model.load_model('sent2vec/sent2vec/model.bin') # The model can be sent2vec or cbow-c+w-ngrams
a = 'asd0-22'.split('-')
print(a[len(a) - 1])
