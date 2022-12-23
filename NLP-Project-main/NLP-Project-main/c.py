import time
import generate_candidates
import os

def getCandidates():
    
    os.system("python3 generate_candidates.py --input input.txt --output out.txt")
    file = open('out.txt','r')
    #os.system("cat out.txt")
    x = file.readlines()
    x = str(x)
    candidates = []
    flag = 0
    s = ""
    i = 0
    while i < len(x):
        if i < 2:
            i += 1
            continue
        if x[i] == '<':
            while(x[i] != '>'):
                i += 1
                continue
            i += 1
        if x[i] == '|' and flag == 0:
            flag = 1
            candidates.append(s)
            i = i+1
            continue
        if flag == 0:
            s += x[i]
        #print(x[i])
        #time.sleep(1)
        if flag == 1 and x[i] == '\\' and x[i+1] == 't':
            flag = 0
            i = i+2
            s = ""
            continue
        i += 1
    file.close()
    file = open("candidates.txt",'a')
    for i in candidates:
        file.write(i)
        file.write("<$>\n")
    #file.write(candidates)
    file.write("<#>\n")
    file.close()
    return candidates
    
    
    
    
    
# candidates = getCandidates()
# for i in candidates:
#     print(i)

