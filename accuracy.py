with open("asama1.txt","r") as file:
    lines = file.readlines()
    correct = 0
    for i,line in enumerate(lines[1:]):
        label = line[len(line)-2:len(line)-1]
       

        if line[:-1].split(" ")[0][:2] == "IN":
            if label == "0":
                correct +=1

        else:
            if label == "1":
                correct +=1
    
    
print(correct)
                   
