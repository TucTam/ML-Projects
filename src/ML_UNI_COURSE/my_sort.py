def get_min(this_list: list):
    least = this_list[0]
    for x in this_list:
        if x < least: least = x
    return least
    

def sorting_numbers(this_list: list):
    foo = []
    while len(this_list) != 0:
        least = get_min(this_list)
        foo.append(least)
        this_list.pop(this_list.index(least))
    return foo
    
# inbuilt sorting alg.
sorted_list1 = sorted(my_list)
# self built sorting alg.
sorted_list2 = sorting_numbers(my_list)