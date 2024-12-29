---
layout: post
title:  "Programming Languages"
date:   2024-02-04 15:08:10 +0530
categories: Programming
---

Programming languages are the tools we use to write software, create web applications, manipulate data, and more. They're fundamental to the world of computing, and there's a wide variety to choose from, each with their own strengths and weaknesses.

Let's start off with some common types of programming languages:

## 1. Procedural Programming Languages

This language type is one of the oldest. It revolves around procedures or routines. They're straightforward and efficient, but can be rigid and difficult to manage for larger software projects.
    
They are great for tasks that can be broken down into a series of sequential steps.

Examples include C and Pascal. 
    
### Go

Also known as Golang, Go is a statically-typed compiled language that was developed at Google. It's known for its simplicity and efficiency. It's particularly good for system-level programming, and it's also used in web development. It has a garbage collector, which makes memory management easier. However, it's less flexible than some other languages, due to its emphasis on simplicity.
        
It is primarily a procedural language, but it does offer some support for object-oriented programming.

```go
package main

import "fmt"

func main() {
    fmt.Println("Hello, world!")
}
```
    
    
## 2. Object-Oriented Programming Languages 

These languages organize code into 'objects' that contain both data and the functions that operate on that data. They're great for larger projects and promote reusability, but can be overkill for smaller tasks.
    
Java and Python are examples. 

Now, let’s go in further details of some programming languages:

### Java

It's a general-purpose language known for its 'write once, run anywhere' philosophy. It's widely used for building enterprise-scale applications. However, it requires a lot of memory and its syntax can be complex for beginners.
Other features → memory safe, garbage collection
    
```java
public class HelloWorld {
    private void printHello() {  
        System.out.println("Hello, world!");
    }

    public static void main(String[] args) {
        HelloWorld helloWorld = new HelloWorld();
        helloWorld.printHello();
    }
}

```
    
    
## 3. Functional Programming Languages 

These languages treat computation as the evaluation of mathematical functions and avoid changing-state and mutable data. They're excellent for parallel processing and have no side effects, but their paradigm can be difficult to grasp for newcomers.

They excel at tasks that can be broken down into independent units that can be executed in any order because they don't depend on the state of the program.

Examples include Haskell and Lisp. 

### Haskell (Functional)

Haskell is a statically typed, purely functional programming language with type inference and lazy evaluation. It's used in fields where high-level, declarative code is beneficial, such as data analysis and symbolic computation. Due to its purity, code written in Haskell can be easy to test and reason about.

```haskell
factorial :: Integer -> Integer
factorial 0 = 1
factorial n = n * factorial (n - 1)

main :: IO ()
main = print (factorial 5)  -- Outputs: 120
```
    
    
## 4. Multi-paradigm Programming Languages 

These languages support more than one programming paradigm, offering greater flexibility to developers. They can support procedural, object-oriented, functional programming, and others, depending on the language. They provide a lot of flexibility, but can also be more complex to learn and use because of the multiple paradigms.

Examples include JavaScript and C++.

Some examples are:

### JavaScript (procedural + object-oriented)

It's the language of the web, used for creating interactive web pages. It's flexible and runs directly in the browser. But, it may have security issues and can be inconsistent across different browsers.

```jsx
function greet(name) {
    return "Hello, " + name + "!";
}

var message = greet("world");

console.log(message);
```
    
### C++ (procedural + object-oriented) 

It's a powerful language used for system/software development and game programming. It's fast and flexible. However, it has a steep learning curve and managing memory can be tricky.

### Python 

It's a high-level, interpreted language known for its readability. It's great for beginners and widely used in scientific computing, data analysis, and AI. However, it's slower than some other languages and not ideal for mobile app development.
Other features → Memory safe, garbage collection

Each of these languages has its own use cases, advantages, and disadvantages. The key is to choose the right tool for the job at hand.

## Memory safe Languages

Memory-safe languages are programming languages that include preventive measures to avoid common memory-related errors, such as buffer overflow and null pointer dereferencing. These languages can either prevent such errors at compile-time or runtime. 

Examples include Java, Python, Rust, Swift, Go, JavaScript, Ruby, C#, TypeScript and Kotlin

#### Buffer Overflow

```python
arr = []
for i in range(100):
    arr.append(i)  # This is fine in Python. The array resizes automatically.
print(arr)

```

In this code, Python automatically resizes the array as new elements are added, preventing a buffer overflow from occurring.

#### Null Pointer Dereferencing

```java
public class Main {
    public static void main(String[] args) {
        String str = null;
        System.out.println(str.length());  // This will throw a NullPointerException
    }
}

```

In this code, `str` is null, so trying to call `str.length()` will throw a NullPointerException, preventing the program from continuing to run with invalid data.

### Garbage collection

---

Garbage collection is a form of automatic memory management. Programming languages with garbage collection automatically reclaim memory that the programmer has allocated but is no longer in use. This process helps to eliminate common programming bugs related to memory management, such as memory leaks and dangling pointers (points to invalid memory).

For example, in Java, you do not need to manually deallocate memory once you're done using an object. The Java Virtual Machine (JVM) has a garbage collector that automatically frees up memory that is no longer needed. This greatly simplifies programming and reduces the chance of memory leaks.

```java
public class Main {
    public static void main(String[] args) {
        // Creating a new object
        String str = new String("Hello, world!");

        // The object is now eligible for garbage collection because it's no longer reachable
        str = null;
    }
}
```

Another language that uses garbage collection is Python. Like Java, Python automatically manages memory, meaning that objects are automatically destroyed once they are no longer in use.

Here is an example of garbage collection in Python:

```python
class MyClass:
    def __init__(self, name):
        self.name = name
    def __del__(self):
        print(f"{self.name} has been deleted and is ready for garbage collection")

# Create an object
my_obj = MyClass("Object 1")

# Now, let's delete this object
del my_obj

# Since the object has been deleted, it's now eligible for garbage collection

```

When the `del` keyword is used, it removes a reference to the object.

However, it's important to note that an object can also become eligible for garbage collection without the `del` keyword, as long as there are no more references to it.

However, it's important to note that while garbage collection can make programming easier, it doesn't completely prevent memory-related bugs. For instance, if an object is mistakenly kept alive when it's not needed, this can still lead to memory leaks. Therefore, understanding how garbage collection works in your programming language of choice is still crucial.

Examples of programming languages that use garbage collection include → Java, Python, JavaScript, C#, Ruby, Go and Kotlin.

## Interpreted vs compile time language

Interpreted and compiled languages represent two different ways of translating human-readable source code into machine code that a computer can execute.

### Interpreted Languages

In an interpreted language, the source code is not directly translated into machine code. Either, an interpreter reads the source code line-by-line, and executes each command. (such as PHP or Ruby)

Or, the source code is first translated into this intermediate form like bytecode (such as Python, Java, and C#), which allows for platform independence as the same bytecode can be interpreted on different machines. This bytecode is then interpreted (e.g. Python) or compiled at runtime using JIT (it is an optimization, that makes it hybrid with compiled languages) (e.g. Java and C#).

Note, machine code is not executed line by line but as a whole. This allows for certain optimizations that can make the code run faster.

![Untitled](/assets/2024/September/interpreted.png)

Interpreted languages are generally more flexible, and they can even modify themselves during runtime.  However, they also tend to run slower than compiled languages, as the interpretation process takes time. 

Examples of interpreted languages include Python, Ruby, and PHP.

### Compiled Languages

In a compiled language, the source code is directly translated into machine code by a compiler before it's run. This means that compiled programs generally run faster than interpreted ones, as all the translation work is done beforehand. 

However, they are less flexible than interpreted languages, as the source code cannot be changed during runtime. 

Examples of compiled languages include C, C++, and Go.

![Untitled](/assets/2024/September/compiled.png)

### JIT compilation

Just-In-Time (JIT) compilation is a method of execution where a program is compiled into machine code just before it is run, rather than ahead of time as with traditional compilation. 

JIT compilation is used in several programming languages, most notably Java and JavaScript, but also in Python, Ruby, and .NET languages such as C#. 

Java and C# are often described as a hybrid of compiled and interpreted languages, due to their two-step process. First, source code is compiled into an intermediate form (bytecode for Java, and Intermediate Language for C#), and then this intermediate form is further compiled to machine code at runtime by a Just-In-Time (JIT) compiler. 

![Fig: Flowchart from high level language to Output (Just my understanding) ](/assets/2024/September/flowchart.png)

*Fig: Flowchart from high level language to Output (Just my understanding)*

The main advantage of JIT compilation is that it can optimize the program for the machine's current state, taking into account factors such as the data being processed and the machine's architecture. This can result in more efficient execution than with traditionally compiled code.

However, JIT compilation also has some disadvantages. The compilation process can lead to an initial delay in execution, known as a "warm-up" period. 

Additionally, JIT compilers are complex pieces of software that can introduce their own bugs and security vulnerabilities.

## Statically vs Dynamically Typed languages

### Statically Typed Languages

In statically typed languages, the variable type is checked at compile-time. This means that you must declare the data type of the variable when you define it, and once set, the variable type can't be changed. This approach can help catch errors early in the development process. Statically typed languages include Java, C, C++, Go, Rust, and Swift.

```java
int num = 10; // Declare an integer variable
num = "Hello"; // Error: incompatible types
```

In the above Java code, an error would be thrown at compile time because you cannot assign a string to an integer variable.

### Dynamically Typed Languages

The counterpart to statically typed languages are dynamically typed languages. In these languages, the type is checked at runtime, which means that you can declare a variable without specifying its type, and its type can be changed later in the program. While this provides more flexibility, it can also lead to errors that are only discovered when the code is run. Dynamically typed languages include Python, Ruby, PHP, and JavaScript.

```python
num = 10  # num is an integer
num = "Hello"  # num is now a string
```

In the above Python code, the `num` variable can be reassigned to a string without any errors, because the type check happens at runtime.

It's worth noting that some languages, like Python, are also capable of type hinting. This is a feature that allows developers to specify the expected type of a variable or function return, improving code readability and allowing for better IDE support and error checking, even though the language itself is dynamically typed.

That’s it for this blog. 🦄🦄 I hope this helps you to better understand your favorite programming language. If you find any mistake or have any doubt feel free to contact!