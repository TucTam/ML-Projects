{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "a4eb39d3-e9aa-43a1-863d-f1d778c78681",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_min(this_list: list):\n",
    "    least = this_list[0]\n",
    "    for x in this_list:\n",
    "        if x < least: least = x\n",
    "    return least\n",
    "\n",
    "\n",
    "def sorting_numbers(this_list: list):\n",
    "    foo = []\n",
    "    while len(this_list) != 0:\n",
    "        least = get_min(this_list)\n",
    "        foo.append(least)\n",
    "        this_list.pop(this_list.index(least))\n",
    "    return foo\n",
    "\n",
    "# inbuilt sorting alg.\n",
    "sorted_list1 = sorted(my_list)\n",
    "# self built sorting alg.\n",
    "sorted_list2 = sorting_numbers(my_list)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
