# Binary Search Templates

## Binary Search by Index

Binary search can be applied over a sequence by using its indeces. We can fix start and end indeces and on each iteration, check if the mid satisfies our condition.

    PseudoCode:

        l = 1, r = n
        while r > l:
            mid = (l+r)/2

            if check(arr[mid]): // if good fix l
                l = mid
            else 
                r = mid 
        print(l)   

We can modify, the above pseudocode to find the first or last index satisfying a given condition.

## Binary Search by Value

Binary search can also be used with values. If we know the possible range of values, then we can just assign l = start and r= end then iterate over those values.

Their are two versions of this algorithm:

1. **Maximize value satisfying a condition**

        Pseudocode:
            l = 1, r = 1
            while (! good(r)) r*= 2

            while (r > l+1):
                mid = (l+r)/2

                if good(mid):  // if good fix l
                    l = mid
                else 
                    r = mid
            print(l) 

2. **Minimize value satisfying a condition**

        Pseudocode:
            l = 1, r = 1
            while (! good(r)) r*= 2

            while (r > l+1):
                mid = (l+r)/2

                if good(mid):  // if good fix r
                    r = mid
                else 
                    l = mid
            print(r)

If the given values are **real numbers**, then we can fix number of iterations and iterate over the values to find minimum or maximum value satisfying a condition.

In this case we have to use floating point values.

    Pseudocode:
        l = 0, r = 1e9

        for (i = 0; i < 100; i++):
            mid = (l + r)/2

            if (good(mid))  // fix l to maximize value
                l = mid
            else 
                r = mid
        print(l)

Similar, algorithm can be used to minimize value.