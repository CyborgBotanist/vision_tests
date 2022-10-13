import time
from networktables import NetworkTables

import logging

logging.basicConfig(level=logging.DEBUG)

NetworkTables.initialize()
sd = NetworkTables.getTable("SmartDashboard")

i = 0
while True:
    print("dsTime:", sd.getNumber("dsTime", -1))


    sd.putNumber("robotTime", i)
    time.sleep(1)
    i += 1