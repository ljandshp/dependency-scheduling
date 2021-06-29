'''
Author: 娄炯
Date: 2021-04-27 10:32:43
LastEditors: loujiong
LastEditTime: 2021-04-27 10:46:34
Description: some coding helper
Email:  413012592@qq.com
'''
import time

# example1
def before_after(func):
    def wrapper(*args):
        print("Before")
        func(*args)
        print("After")
    
    return wrapper

class Test:
    @before_after
    def decorated_method(self):
        print("run")

t = Test()
t.decorated_method()


# example2
def timer(func):
    def wrapper():
        before = time.time()
        func()
        print("Function took:", time.time() - before, "seconds")
    
    return wrapper

@timer
def run():
    time.sleep(2)

# run()

#example3
import datetime
def log(func):
    def wrapper(*args,**kwargs):
        with open("log.txt","a") as f:
            f.write("Called function with "+" ".join([str(arg) for arg in args]) +" at "+ datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        val = func(*args,**kwargs)
        return val
    return wrapper

@log
def run(a, b,c=9):
    print(a+b+c)

run (1,3,c=9)