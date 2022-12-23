

train_src_data_file = open('data/ACL2020/train.src', 'r')
sentence = ""

# read and print the first sentence in the file
for line in train_src_data_file:
    for ch in line:
        if ch != '.':
            sentence += ch
        else:
            sentence += ch
            print(sentence)
            sentence = ""
            break
    break



train_src_data_file.close()