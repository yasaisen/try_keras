
import os

imdb_dir = os.path.join(os.getcwd(), 'aclImdb')
train_dir = os.path.join(imdb_dir, 'train')

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname), encoding = 'utf8')
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

print(len(labels))
print(len(texts))

with open('11_FileManager_labels.txt', 'a') as f:#找到或創建
    for i in labels:
        f.writelines(str(i))#寫入資訊
        f.writelines('\n')#寫入資訊
    f.close()

with open('11_FileManager_texts.txt', 'a') as f:#找到或創建
    for i in texts:
        f.writelines(str(i))#寫入資訊
        f.writelines('\n')#寫入資訊
    f.close()



