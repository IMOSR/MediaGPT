import os
path = 'data/'
files = os.listdir(path)
for i, file in enumerate(files):
    NewFileName = os.path.join(path, file.strip(".json")+"_1"+'.json')
    OldFileName = os.path.join(path, file)
    os.rename(OldFileName, NewFileName)
