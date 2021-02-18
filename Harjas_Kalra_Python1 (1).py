#!/usr/bin/env python
# coding: utf-8

# <h1>Python 1 - Overview</h1>
# <p>
#   Bootcamp will cover Python fundamentals while making a music playlist program
# </p>
# <ul>
# <li>Evaluating primitive types in python: type()</li>
# <li>Declaring variables and variable declaration conventions: =</li>
# <li>Math Operators and string concatenation: (+ , - , * , /,%)</li>
# <li>IF and WHILE statements with conditional operators: (==, >, >=, break)</li>
# <li>User input: input()</li>
# <li>Data collections - Lists: ([ ], append(), insert(), del, pop(), len(), sort())</li>
# <li>Data collections - Dictionaries: ({ },[ ], insert(), del, clear(), keys(), values())</li>
# <li>Declaring custom functions: def, return</li>
# <li>Classes and object oriented programming: class(), __init__(), methods</li>
# <li>Automating with FOR loops: for, in</li>
# </ul>

# ## Jupyter Notebook 
# 
# This is a web-based application (runs in the browser) that is used to interpret Python code. 
# 
# - To add more code cells (or blocks) click on the **'+'** button in the top left corner
# - There are 3 cell types in Jupyter:
#     - Code: Used to write Python code
#     - Markdown: Used to write texts (can be used to write explanations and other key information)
#     - NBConvert: Used convert Jupyter (.ipynb) files to other formats (HTML, LaTex, etc.) 
#     
# 
# - To run Python code in a specific cell, you can click on the **'Run'** button at the top or press **Shift + Enter**
# - The number sign (#) is used to insert comments when coding to leave messages for yourself or others. These comments will not be interpreted as code and are overlooked by the program
# 

# <h1>Data Types</h1>
# <ul>
#   <li>
#    Four primitive types in Python
#     <ol>
#       <li>
#         Integers
#       </li>
#       <li>
#         Booleans
#       </li>
#       <li>
#         Floats
#       </li>
#       <li>
#         Strings
#       </li>
#     </ol>
#  <li>
#    Types may be changed using int(), str(), float(), and bool() methods    
#  </li>
# </ul>

# In[1]:


# The type() function will return the data type of the data passed to it
type("hello")


# In[2]:


type(True)


# In[3]:


type(3.14)


# In[4]:


type(3)


# In[5]:


# Casting - converting from one data type to another
print(type(float(3)))


# In[6]:


print(int(3.55))
print(bool("hello"))


# <h1>Variables</h1>
# <ul>
#   <li>
#     May consist of letters, numbers, and underscores, but not spaces.
#     <ul>
#       <li>
#         <b>Cannot start with a number.</b>
#       </li>
#     </ul>
#   </li>
#   <li>
#     Avoid using Python keywords (for, if, and, or, etc.)
# </li>
#   <li>
#     Be careful when using 1s and lower case ls, as well as 0s and Os.
# </li>
#   <li>
#     Keep it short.
#   <li>
#     Example: phone_num = 647606
# </li>
#     
# </ul>

# In[7]:


hours_worked = 10


# In[8]:


print(hours_worked)


# <h1>Math Operators</h1>
# <ul>
#   <li>
#     Addition, Subtration, Multiplication and Division may be done using basic math operators (+ , - , * , /,%).
#   </li>
#   <li>
#     Many built-in string methods (title, upper, lower, index, split).
#   </li>
#   <li>
#     Python will also try to interpret your code with other data types
#   </li>
#   <ul><li>(+) may be used with strings!</li></ul>
# </ul>

# In[11]:


# Create two variables, price1 and price2 that have float values representing the respective price of two items
price1 = 3.40
price2 = 2.51
# Create a new variable whose value is the sum of the duration of both songs
tot_price = price1 + price2
diff_price = price1-price2
mult_price = price1 * price2
div_price = price1 / price2
print(tot_price)


# In[12]:


# Define string variables name, job, and tool
name = "Seamus"
job = "works with"
tool = "Python"


# In[13]:


# We can concatenate (combine) strings together using the addition (+) symbol 
employment = name + " " + job + " " + tool
print(employment)


# In[14]:


print(employment.title())
print(employment.lower())

print(employment.index("works"))
print(employment.split(" "))
print(employment.replace("IT", "Finance"))


# In[16]:


# With F strings, variables go directly into a string! Even methods!
print(f"{name} works with {tool.upper()}")


# In[7]:


# A boolean can only have one of two values. Either they are "True" or "False". 


# <h1>IF and WHILE Statements</h1>
# <ul>
#   <li>
#     Will only run indented code if condition is true
#   <li>
#     Make use of <b>conditional operators</b> to create tests
#   </li>
#   <ul><li>(==) will return true if both variables are equal</li>
#   <li>(>) will return true if left variable is larger</li>
#   <li>(>=) will return if left variable is larger or equal to right variable</li></ul>
#   <li>IF will only run indented code once, WHILE will run indented code until condition is no longer true</li>
# </ul>

# In[17]:


# Boolean variables are generally used for conditional statements such as an if statement.
yes = True
no = False

if yes:
    print("true statement!")
    
if no:
    print("this won't print")


# In[18]:


dept_size = 10


# In[27]:


# if else statments can also be used with math or anything really (like strings)!
# if dept_size is less than 14, display the number of employees in the department. 
# Else, display a message saying the department size was exceeded
show_warning = False
if dept_size < 14:
    dept_size += 1
    print(f"New hire. {dept_size} employees in department")
elif dept_size < 18 and show_warning:
    dept_size += 1
    print(f"{dept_size} employees. Office is getting packed!")
else:
    print("size exceeded, new offices needed!")


# In[28]:


# While loops will keep running a loop of code until the intial condition is no longer true
# It is important to always have a breaking condition to stop the loop so it does not run infinitely
dept_size = 0
limit = 10

while dept_size < limit:
    print(dept_size)
    dept_size += 1


# In[29]:


#Give dept_size a value of 0.
dept_size = 0


#WHILE Loop with condition of True will infinitely continue 
while True:
    dept_size += 1
    print(dept_size)
    if dept_size == 8:
        break


# <h1>Lists</h1>
# <ul>
#   <li>
#    Collection of items in a particular order
#   <li>
#    They are used to store data and can be assigned to variables just like integers and strings
#   <li>
#    Indexing (order) starts from <b>0</b>
#   </li>
#   <li>Accessing items in a list can be done with square brackets ([ ])</li>
#   <li>Items can be easily added to lists using append() and insert() methods</li>
# </ul>

# In[30]:


# Lists are a collection of data. List numberings always start from 0.
banks = ['RBC', 'CIBC', 'TD', 'BMO']


# In[34]:


print(banks[0])
print(banks[-1])
print(banks[0:3])
print(banks[1:])


# In[42]:


banks[0] = 'Scotiabank'
print(banks)


# In[43]:


# add value to end of a list - Canadian Western Bank
# The .append() function can be used!
banks.append("CWB")
print(banks)


# In[44]:


# add value to the start of a list  - First Nations Bank of Canada
banks.insert(0,"FNBC")
print(banks)


# In[45]:


# Remove list entries
print(len(banks))
del banks[0]
print(banks)
print(len(banks))


# In[41]:


# lists can contain any type of data. A single list can be a mixture of different data types
mix_list = ['Peter', 31445, True, 'IT']
print(mix_list[0])
print(mix_list[1:])


# In[ ]:


print(f"Name: {mix_list[0]}")


# <h1>Dictionaries</h1>
# <ul>
#   <li>
#    Collection of key-value pairs
#   <li>
#    No positions as with lists, values stored at specific key
#     <ul><li>keys can be of any data type</li></ul>
#   </li>
#   <li>Accessing values in a dictionary can still be done with square brackets ([ ])</li>
#   <li>Declared using braces ({ })</li>
# </ul>

# In[47]:


# collection of "data" which is unordered, changeable, and not indexed. They have keys and values.

employee = {'name': 'Seamus', 'employment_num':31445, 'dept':'IT'}
print(employee)


# In[48]:


print(employee['name'])


# In[49]:


employee['dept'] = 'Finance'
print(employee['dept'])


# In[50]:


employee['management'] = False
print(employee)


# In[51]:


# Can remove a key easily using del
# Other keys are unaffected when you use 'del' to remove a key
del employee['management']
print(employee)


# In[52]:


#Dictionary methods return iterables
print(employee.items())
print(employee.keys())
print(employee.values())
# Cannot do print(employee.keys[0]) because it is not a list
# Iterables are data objects that can be 'interated' over, like in loops 
# Iterables to be used with keyword IN ('IN' example is covered in the next cell blocks and section with 'For' Loops)


# In[53]:


# You can use dictionaries and lists in 'if' statments. 
if "name" in employee:
    print("yes name is one of the keys in this dictionary")

#Will look through keys by default


# <h1>For Loops</h1>
# <ul>
#   <li>
#    Execute a block of code once for each item in collection (List/Dictionary)
#   <li>
#    Declare temporary variable to iterate through collection
#   </li>
#   <li>Can be used in combination with IF statements</li>
# </ul>

# In[54]:


#Loop through banks list
for bank in banks:
    print(bank)


# In[55]:


#Loop through pairs in employee dictionary
for key, val in employee.items():
    print(f"{key}:{val}")


# In[56]:


# Use RANGE to specify a number of iterations 
for i in range(10):
    print(i)


# <h1>Functions</h1>
# <ul>
#   <li>
#    Named blocks of code that do one specific job
#   <li>
#    Functions are also referred to as methods
#   <li>
#    Prevents rewriting of code that accomplishes the same task
#   </li>
#   <li>Keyword <i>def</i> used to declare functions</li>
#   <li>Variables may be passed to functions</li>
# </ul>

# In[24]:


# In this function 'name', 'employee_num', and 'department' are required values that must be passed to the function


# <h1>Classes</h1>
# <ul>
#   <li>
#   Object-orientated programming approach popular and efficient
#   </li>
#   <li>
#    Define classes of real-world things or situations (can be thought of as creating your own data type)
#     <ul>
#       <li>Attributes of various data types</li>
#       <li>Functions inside of a class are the same except called methods</li>
#       <li>Methods may be accessed using the dot operator</li>
#     </ul>
#   </li>
#   <li>Instanciate objects of your classes</li>
#   <li>__init()__ method used to prefill attributes</li>
#   <li>Capitalize class names</li>
# </ul>

# In[1]:


class Employee():    
    pass


# In[ ]:





# <h1>User Input</h1>
# <ul>
#   <li>
#      Pauses your program and waits for the user to enter some text
#   <li>
#     Variable used with Input() will be a <b>string</b> even if user inputs an integer
#   </li>
#   <ul><li>Will need to make use of <b>type casting.<b></li></ul>
# </ul>

# In[ ]:


#Ask user for age


# <h1>Putting it all Together</h1>
# <ul>
#   <li>
#       Let's take user input and create a new <b>Employee</b>
#   <li>
#     We can then use our class methods easily!
#   </li>
# </ul>

# In[ ]:




