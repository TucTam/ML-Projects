def selectionSort(list):
    for i in range(len(list)):
        for j in range(i+1, len(list)):
            min_idx = i
            if list[min_idx] > list[j]:
                min_idx = j
                list[i], list[min_idx] = list[min_idx], list[i]
    return list

lists = [5,1,0]

def binaryseach(list,l,r,x):
    if r >= 1:
        mid = l + (r-1) // 2
        if list[mid] == x:
            return mid
        elif list[mid] > x:
            return binaryseach(list,l, mid-1,x)
        else:
            return binaryseach(list, mid+1,r,x)
    else:
        return -1

def partition(l, r, list):
  pivot = list[r]
  ptr = l-1 #O(1)
  for i in range(l, r): #O(n)
    if list[i] <= pivot: #O(1)
      ptr = ptr + 1 #O(1)
      list[ptr], list[i] = list[i], list[ptr]
      
  list[ptr + 1], list[r] = list[r], list[ptr + 1]
  return ptr + 1 #O(1)

def quicksort(l, r, list):
    if l < r:
        # Partition the array and get the pivot index
        pi = partition(l, r, list)
        
        # Recursively sort the elements before the partition and after the partition
        quicksort(l, pi - 1, list)
        quicksort(pi + 1, r, list)
        
    return list
print(quicksort(0, 6, [4, 2, 8, 9, 7, 1, 3]))
print(quicksort(0, 2, [4, 1, 0]))