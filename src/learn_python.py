list = ['eyyo', 'whatup', 'wasup', 'hello']
list2 = [1, 2, 3, 4]
first_dic = {key: value 
    for key in list
    for value in list2}
print(first_dic)