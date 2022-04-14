# Hashing and Hash Functions

## Hash tables

Hash tables are direct access tables, with an optimization of hash functions.

Need:

Since the index value can be large, so we can't use it directly as a index in the table, we have to convert the index value into a smaller key using some hash function. And then we store the data at the calculated index or key.

    Hashing
                    Index value (Integer or string) ---
                    (applies hash function) ---> 
                    key or index ----> 
                    table[index] = data

## Hash Functions

Hash functions are used to convert given index value into a small index or key. A good hash function generates different indeces for different values.

Example: -

    int hashfunc(string s) 
    { 
        int index = 0;
        for (char in s):
            index = (index + (s[i] - 0)) % size;    
        return index;
    }

    string index_value = "DSM";
    int index = hashfunc(element);
    table[index] = data;

## Collision Handling

Collision occurs when, the hash function gives same index for two different index values.

In such a situation their are ways two handle this: -

1. Separate Chaining

    In this we use a linked list, to store all elements corresponding to the index. In this we use array of type linked list.

        index = hashfunc(index_value);
        table[index].append(data);

2. Open Addressing

    In this we find next empty index by simply traversing the array, using some other algorithms.

    A. Linear Probing -> In this we traverse the array linearly.

        Pseudocode:
            while not empty table[index]:
                index = (index + i) % size
                i++

    B. Quadratic Probing -> In this we traverse the array, quadratively.

        Pseudocdoe:
            while not empty table[index]:
                index = (index + i^2) % size
                i++

    C. Double hashing -> In this we find the next empty index using a second hash function.

        Pseudocode:
            index = hashfunc(element)
            index2 = hashfunc2(element)
            
            while not empty table[index]:
                index = (index + i*index2) % size
