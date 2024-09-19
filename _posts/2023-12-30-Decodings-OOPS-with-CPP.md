---
layout: post
title:  "Decoding OOPs with cpp"
date:   2023-04-09 15:08:10 +0530
categories: Programming
---

Programming is a vast field with myriad techniques and paradigms. Among these, Object-Oriented Programming (OOP) shines as a method that helps us structure our programs using objects. Objects are entities that bundle data and functionality, making our code more intuitive and easier to maintain. C++ is a perfect language to delve into OOP, as it inherently supports this paradigm.

Let’s start our journey with OOP’s (aka Object Oriented Programming). C++ is a OOP’s oriented programming language.

## 1. Class (Logical Entity)

A class serves as a blueprint for creating objects. In technical terms, it's a user-defined data type with its own data members (attributes) and member functions (methods). 

To access a class we then need instance of that class known as object.

Example → Consider a class 'Car', which could have attributes like speed limit and mileage, and methods like applying brakes, increasing speed, and so on.

Class car

Data members → Speed limit, mileage, etc.

Member functions → Apply brakes, increase speed, etc.

## 2. Object (Physical Entity)

An object is an instance of a class. It's a real-world entity that gives us the ability to access and manipulate the data within a class. 

- Class definition (no memory allocated).
- When we create object (memory allocated).

Example →

```cpp
class Person 
{
	char name[20];
	int id;
	public:
		void getdetails() {}
};

int main()
{
	Person pi;     // object
}
```

## 3. Encapsulation (protective shield)

Encapsulation is the practice of binding data and the functions that handle them within a class. It acts as a protective shield, preventing data from being accessed directly by the code outside the class.

In encapsulation we hide the data and implementation details within a class.

- Used in Abstraction
- Binding together the data and function that manipulate them.

Example → a Class.

## 4. Abstraction (Hiding the complexity)

Abstraction means displaying only essential information and hiding the details.

There are 2 ways to do Abstraction: -

#### Using Class → 
    
We use **access specifiers**.  Access specifiers enforce restrictions on class members.

1. Public → We can access from anywhere.
2. Private (Default) → We can access only from within class.
3. Protected → We can also access from subclass.

Example →

```cpp
Class Abstraction
{
    private:
        int a, b;
    public:
        void set(int x, int y) {
            a = x, b = y;
        }
        void dispaly() {
            cout << a << " " << b << "\n";
        }
}

int main()
{
    Abstraction obj;
    obj.set(1, 2);
    obj.display();
}

// OUTPUT
// 1 2
```
    
#### In header Files → 
Example → You can import math.h and use Pow() function without directly knowing algorithm behind it.

The degree of abstraction refers to the level of detail or complexity at which information is presented or understood. 

Completely hidden → highest degree of abstraction.

### 5. Polymorphism (Many forms of an object)

Polymorphism refers to an object's ability to take on many forms. 

- Property of having many forms
- Ability of a message to be displayed in more than one form.

Example → Like man can be husband, father and son.

Polymorphism in C++

#### 1. Operator Overloading
    
Customizes the C++ operators for operands of user-defined types.

Example → “+” operator can be overloaded to do string concatenation and addition.
”a” + “b” = “ab”
4 + 3 = 7

| Overloadable | Non-Overloadable |
| --- | --- |
| +          -           *          /        %          ^          = | ::            .*          .           ?: |

Also, we can define the behavior of operators for objects, be defining them in class.

Example →
    
```cpp
class complex {
    public:
        int real, img;
        complex(int r=0, int i=0) {real=r, img=i;}
        
        // Here, we defined behavior of "+" when we add to complex numbers.
        complex operator + (complex const &obj) {
            complex res;
            res.real = real + obj.real;
            res.img = img + obj.img;
            return res;
        }
}

// an easy example -> used in sorting
struct P {
    int x, y;
    bool operator<(const P &p) {
        if (x != p.x) return x < p.x;
        else return y < p.y;
    }
};
```
    
#### 2. Templates
#### 3. Function Overloading
Use a single function name to perform different tasks.

Example → 

```cpp
int sum(int a, int b) {
    return a+b;	
}
int sum(int a, int b, int c) {
    return a + b + c;
}

int main()
{
    sum1 = sum(2+3);
    sum2 = sum(2 + 3 + 4);
}
```
    

Early binding is when a variable is assigned its value at compile time. Late binding is when a variable is assigned a value at run time.

### Dynamic Binding / Abstract class / virtual method
    
Dynamic binding, also known as late binding or runtime polymorphism, is a mechanism in object-oriented programming where the actual method or function to be executed is determined at runtime based on the type of the object.

In C++, dynamic binding is achieved through the use of virtual functions. A virtual function is a member function declared within a base class and marked with the `virtual` keyword. When a derived class overrides this virtual function, the function call is resolved dynamically at runtime based on the actual type of the object being referenced.

Example:
    
```cpp
class Shape {    // THIS IS NOT AN ABSTRACT CLASS
public:
    virtual void draw() {    // because provides implementation of virtual method
        cout << "Drawing a shape." << endl;
    }
    
};

class Circle : public Shape {
public:
    void draw() override {
        cout << "Drawing a circle." << endl;
    }
};

class Rectangle : public Shape {
};

int main() {
    Shape* shape1 = new Circle();
    Shape* shape2 = new Rectangle();

    shape1->draw(); // Dynamic binding - calls draw() of Circle
    shape2->draw(); // Dynamic binding - calls draw() of Shape
        // FUN NOTE -> A->B IS EQUAL TO (*A).B

    delete shape1;
    delete shape2;

    return 0;
}
```
    
In the above example, the `draw()` function is declared as a virtual function in the base class `Shape`. When we create objects of derived classes (`Circle` and `Rectangle`) and assign them to a pointer of type , the appropriate  function is called based on the actual type of the object being referenced. This is determined dynamically at runtime, allowing for polymorphic behavior.

Yes, if the `Rectangle` class does not override the `draw` method of the `Shape` class, then calling `draw` on an object of the `Rectangle` class will call the `draw` method defined in the `Shape` class.

### Abstract Class

An abstract class is a class that cannot be instantiated, meaning we cannot create objects of that class. It serves as a blueprint for other classes to inherit from. It contains at least one pure virtual function, which is a virtual function that has no implementation in the abstract class and must be overridden by any derived class inheriting from it.

To define an abstract class in C++, we use the `virtual` keyword to declare pure virtual functions. The syntax is as follows:

```cpp
class AbstractClass {   // THIS IS AN ABSTRACT CLASS
public:
    virtual void pureVirtualFunction() = 0; // Pure virtual function
    void nonVirtualFunction() {
        // Non-virtual function
    }
};

```
    
In the example above, `pureVirtualFunction()` is a pure virtual function, and `nonVirtualFunction()` is a normal member function. Any class that inherits from `AbstractClass` must override the pure virtual function, but it can choose whether to override the non-virtual function.
    

## 6. Inheritance

Capability of a child class (Sub-Class) to derive properties and characteristics from parent class (Super Class).

Gives us “Reusability” of methods.

We can access **public, protected** member of super class from subclass.

Example → 

Let’s say there is an Animal class. then we can create subclasses like Dog, Cat and Cow subclasses which inherits the properties of Animal super class.

```cpp
class Animal 
{
		public:
			string species;
}

class Dog:public Animal {
		public: 
			int weight;
}

// Multiple inheritance
class C : public A, public B {
public:
    void displayC() {
        cout << "This is class C" << endl;
    }
};

int main() {
	Dog d;
	d.species = "xyz";
	d.weight = 100;
}
```

Note → The access specifier used while inheriting the parent class, tells the nature of the inherited properties.  By default, it is private. 

In object-oriented programming, an IS-A relationship refers to inheritance or subclassing.

For example, if we have a base class called "Animal" and a derived class called "Dog", we can say that "Dog IS-A Animal". This means that a dog is a type of animal and inherits the characteristics of an animal.

### Types of Inheritance

In C++, inheritance allows a derived class to inherit properties and characteristics from a base class. There are several types of inheritance:

1. Single Inheritance: A derived class inherits properties from a single base class.
2. Multiple Inheritance: A derived class inherits properties from multiple base classes.
A → B
C → B
3. Multilevel Inheritance: A derived class inherits properties from a base class, which in turn inherits properties from another base class.
A → B → C
4. Hierarchical Inheritance: Multiple derived classes inherit properties from a single base class.
A → B
A → C
5. Hybrid Inheritance: It is a combination of multiple and multilevel inheritance.

These types of inheritance provide flexibility in designing class hierarchies, allowing for code reuse and organization of related classes.

In conclusion, Object-Oriented Programming provides a clear structure for our programs, making the code easier to design, understand, and maintain. So, the next time you're coding in C++, keep these OOP concepts in mind as your guiding principles. They will surely make your programming journey smoother and more efficient.