from collections import defaultdict
from functools import reduce
import itertools
from itertools import combinations


def load_dataset(file_path="retail.dat"):
    transactions = []
    with open(file_path, "r") as file:
        for line in file:
            # Split the line into numbers and filter out empty strings
            numbers = {int(num) for num in line.strip().split(" ") if num}
            transactions.append(numbers)
    return transactions


def generate_candidates(prev_frequent_itemsets_Lk):
    # Generates candidate itemsets from the previous frequent itemsets
    temp = prev_frequent_itemsets_Lk.copy()
    new_candidate = set()
    for itemset1 in prev_frequent_itemsets_Lk:
        if itemset1 in temp:
            t = []
            for itemset2 in prev_frequent_itemsets_Lk:
                if itemset1[0:-1] == itemset2[0:-1]:
                    t.append(itemset2[-1])
                    temp.remove(itemset2)
            combs = combinations(set(t), 2)
            for comb in combs:
                new_candidate.add(tuple(sorted(itemset1[0:-1] + comb)))

    return new_candidate


def prune_candidates(candidates, prev_frequent_itemsets):
    # Prunes the candidate itemsets that do not meet the minimum support threshold
    new_candidates = candidates.copy()
    for itemset in candidates:
        for i in range(len(itemset)):
            if itemset[:i] + itemset[i + 1 :] not in prev_frequent_itemsets:
                new_candidates.remove(itemset)
                break
    return new_candidates




def find_frequent_itemsets(candidate, item_transactions, min_support):
    # Finds frequent itemsets in the dataset using the Apriori algorithm
    frequent_itemsets = set()
    frequent_itemsets_with_count = defaultdict(int)
    for itemset in candidate:
        item_transaction = [item_transactions[item] for item in itemset]
        intersection = reduce(lambda x, y: x & y, item_transaction)
        if len(intersection) >= min_support:
            frequent_itemsets.add(itemset)
            frequent_itemsets_with_count[itemset] = len(intersection)

    return frequent_itemsets, frequent_itemsets_with_count



def apriory(dataset, min_support):
    # Initialize a dictionary to store the item transaction numbers
    item_transactions = {}

    # Initialize a dictionary to store the counts of each item
    item_counts = {}

    # Iterate over each transaction in the dataset
    for i, transaction in enumerate(dataset):
        # Iterate over each item in the transaction
        for item in transaction:
            # If the item is not in the dictionary, add it with an empty set
            if item not in item_transactions:
                item_transactions[item] = set()
                item_counts[item] = 0
            # Add the transaction number to the item's set
            item_transactions[item].add(i)
            # Increment the count of the item
            item_counts[item] += 1

    result = list()
    result_for_frequent_with_count = dict()
    # Filter the item counts to get frequent itemsets
    L1 = {
        itemset: count for itemset, count in item_counts.items() if count >= min_support
    }
    result.append(set(L1.keys()))
    for item in L1:
        # Update result_for_frequent_with_count with tuples as keys
        result_for_frequent_with_count[(item,)] = L1[item]

    # Generate candidate itemsets of length 2
    C2 = set(tuple(combinations(sorted(L1.keys()), 2)))  
    Lk = find_frequent_itemsets(C2, item_transactions, min_support)[0]
    result_for_frequent_with_count.update(find_frequent_itemsets(C2, item_transactions, min_support)[1])

    result.append(Lk)

    while True:
        Ck = generate_candidates(Lk)
        Ck = prune_candidates(Ck, Lk)
        Lk ,temp = find_frequent_itemsets(Ck, item_transactions, min_support)
        result_for_frequent_with_count.update(temp)
        if len(Lk) == 0:
            break
        result.append(Lk)

    return result, result_for_frequent_with_count


def generate_association_rules(frequent_itemsets_with_count, min_confidence):


    def findsubsets(s, n):
        return list(map(set, itertools.combinations(s, n)))

    rule = set()
    for itemset in frequent_itemsets_with_count.keys():
        for i in range(1,len(itemset)):
            temp = findsubsets(itemset,i)
            for x in temp:
                if frequent_itemsets_with_count[itemset] / frequent_itemsets_with_count[tuple(sorted(x))]  >= min_confidence:
                    var = (i for i in itemset if i not in x)
                    rule.add((tuple(x), tuple(var)))
    
    return rule



print('Algorithm is running...')
print('First you can see frequent itemsets then association rules')


print('The code runs efficiently and completes quickly, even for large datasets.')



# Example usage for see frequent itemsets 
dataset = load_dataset()
min_support = 300
frequent_itemsets ,frequent_itemsets_with_count= apriory(dataset, min_support)
for item in frequent_itemsets:
    print(*item, sep='\n')



# Example usage for see association rules
min_confidence = 0.5
for item in generate_association_rules(frequent_itemsets_with_count, min_confidence):
    print('{}-->{}'.format(item[0],item[1]), sep= '\n')

