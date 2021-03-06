'''
Author: 娄炯
Date: 2021-04-27 10:32:43
LastEditors: loujiong
LastEditTime: 2021-07-17 13:28:39
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

if __name__ == '__main__':
    # run (1,3,c=9)


    # bisect
    # import bisect
    # a = [1,2,3,4]
    # bisect.bisect(a,2.4)
    # bisect.insort(a,2.5)
    # example 2
    # data = [('red', 5), ('blue', 1), ('yellow', 8), ('black', 0)]
    # data.sort(key=lambda r: r[1])
    # keys = [r[1] for r in data]
    # print(data[bisect_left(keys, 0)])
