from c import *
train_src_data_file = open('train.src', 'r')
sentence = ""
file = open("input.txt",'w')
file_size = os.path.getsize('train.src')
bytes_read = 0
# read and print the first sentence in the file
for line in train_src_data_file:
    file = open("input.txt",'w')
    for ch in line:
        if ch != '.':
            sentence += ch
        else:
            sentence += ch
            bytes_read += len(sentence) 
            file.write(sentence)
            file.close()
            getCandidates()
            print(f'Progress = {bytes_read*100/file_size}\n\n\n')
            sentence = ""
            break
    
    



train_src_data_file.close()