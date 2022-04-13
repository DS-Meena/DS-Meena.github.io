# Hashing and Hash Functions

## Hash tables 
Hash tables are unordered map, that uses hash functions to store values. 

    Hashing 
    value/number/string ---(applies hash function) ---> key/index ----> arr[index] = value/number/string

## Hash Functions

Hash functions are used to find the key or index of the given element.

Example: -

    int hashfunc(int x) 
    { return x % arbitrary_number; }

    int element = 1e9;
    int index = hashfunc(element);
    arr[index] = element;

## Collision Handling

collision occurs when, the hash function gives same index for two different elements. 

In such a situation their are ways to handle this: -

1. Separate Chaining

    In this we use a linked list, to store all elements corresponding to the index. In this we use array of type linked list.

        index = hashfunc(element);
        arr[index].push(element);


    
