"""Utility script to delete id/ids from the redis cache database"""
import redis
from random import randint

def confirm():
    pin = str(randint(1000, 9999))
    inp_pin = input("Please enter '" + pin + "' to confirm: ")
    print("You entered: " + inp_pin)
    return pin == inp_pin

options = "\n\t(1) Raw PMC Articles\n\t(2) Processed PMC Articles\n\t(3) GenBank Object Cache\nOption: "
db = input("Please enter the database number to select for deletion:"+options)
r = redis.Redis(host="localhost", port="6379", db=db)

options = "\n\t'ALL' to delete all keys \n\tor comma separated keys e.g.'AY849090,AY849244'\nKeys: "
inp_opt = input("Please enter deletion option:"+options)
if inp_opt.strip() == "ALL":
    print("You are attempting to delete all entries from database:" + db)
    if confirm():
        print("Deleting database", db)
        r.flushdb()
else:
    keys = [x.strip() for x in inp_opt.split(",")]
    for key in keys:
        if r.get(key):
            r.delete(key)
        else:
            print(key, "doesn't exist in cache")
