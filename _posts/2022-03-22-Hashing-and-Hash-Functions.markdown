---
layout: post
title:  "Hashing and Hash Functions"
date:   2022-03-22 15:08:10 +0530
categories: Data structures
---

# Introduction

Let’s talk hashing and hash tables. Array + Linked List advantages.

`Hash Table` is a data structure which organizes data using `hash functions` in order to support quick insertion and search.

# Hashing

In hashing, we convert a value/num/string into key index using hash function. The idea is to uniformly distribute the key to the array.

`Hash_func(value/int/string) = key/index`

`arr[key/index] = value/int/string`

When we insert a key-value pair in a hash table, we store the key along with its value.

Properties of good hash function: -

1. Makes use of all info provided by key
2. Uniformly distributes output across table
3. Maps similar keys to very different hash values
4. Uses only very fast operations.

# Avoiding Collision

It’s very hard to create a perfect hashing function. Rarely or when we use a bad hashing function, it gives same key for 2 different values. Such case is known as Collision.

To avoid collision we apply different techniques, as explained below:

## 1. Separate chaining

In this we use a map of type vector, hence we simply push the next value at the key.

Example →

```cpp
index = hash_func(s)
map[index].push_back(s)
```

Complexity:

Insertion takes O(1) and search takes O(k), k = size of linked list.

**Open addressing** based algorithms. In these algorithms we find the next empty space.

## 2. Linear Probing

```cpp
while not empty:
	index = (index + i)%size
	i++
```

[Implementation of Hash Set with Linear probing](https://leetcode.com/explore/learn/card/hash-table/182/practical-applications/1139/)
    
```python
class MyHashSet {
public:
static const int size = 1000005;
int arr[size];

MyHashSet() {
    memset(arr, -1, sizeof(arr));
}

void add(int key) {
    // hash function
    int indx = key % size;
    
    // linear probing
    int i=1;
    while(arr[indx] != -1) {
        indx = (indx + i) % size;
        i++;
    }
    
    // store
    arr[indx] = key;
}

void remove(int key) {
    int indx = key % size;
    
    int i=1;
    while(arr[indx] != -1) {
        
        // remove only if contains
        if (arr[indx] == key) {
            arr[indx] = -1;
            break;
        }
        
        indx = (indx + i)%size;
        i++;
    }
}

bool contains(int key) {
    int indx = key % size;
    
    // move till empty space
    int i =1;
    while(arr[indx] != -1) {
        
        // contains
        if (arr[indx] == key) 
            return true;
        
        indx = (indx + i)% size;
        i++;
    }
    
    return false;
}
};
```

- [Hash Map Implementation with 2 arrays and Linear probing](https://leetcode.com/explore/learn/card/hash-table/182/practical-applications/1140/)
    
```cpp
class MyHashMap {
public:
    static const int size = 1000005;
    int key[size];
    int value[size];
    
    MyHashMap() {
        memset(key, -1, sizeof(key));
        memset(value, -1, sizeof(value));
    }
    
    void put(int k, int v) {
        int indx = k % size;
        
        int i=1;
        while(key[indx] != -1) {
            // already present
            if (key[indx] == k)
                break;
            
            indx = (indx + i) % size;
            i++;
        }
        
        key[indx] = k;
        value[indx] = v;
    }
    
    int get(int k) {
        int indx = k % size;
        
        int i =1;
        while(key[indx] != -1) {
            
            // contains
            if (key[indx] == k) 
                return value[indx];
            
            indx = (indx + i)% size;
            i++;
        }
        
        return -1;
    }
    
    void remove(int k) {
        int indx = k % size;
        
        int i=1;
        while(key[indx] != -1) {
            
            // remove only if contains
            if (key[indx] == k) {
                key[indx] = -1;
                value[indx] = -1;
                break;
            }
            
            indx = (indx + i)%size;
            i++;
        }
    }
};
```
    
### Complexity: 

Search and insert both takes O(n), n = size of table

## 3. Quadratic Probing

```cpp
while not empty:
	index = (index + i^2) % size
	i++
```

## 4. Double hashing

```cpp
index = hashfunc(s)
index2 = hashfunc2(s)

while not empty:
	index = (index + i * index2) % size
	i++
```

Practically we use separate chaining with height-balanced BST.

### Complexity:-

The average time complexity of both insertion and search is still `O(1)`. And the time complexity in the worst case is `O(logN)` for both insertion and search by using height-balanced BST (at each index, using vector). It is a trade-off between insertion and search.