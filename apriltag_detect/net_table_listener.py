#
# This is a NetworkTables client (eg, the DriverStation/coprocessor side).
# You need to tell it the IP address of the NetworkTables server (the
# robot or simulator).
#
# This shows how to use a listener to listen for changes in NetworkTables
# values. This will print out any changes detected on the SmartDashboard
# table.
#

import numpy as np
import sys
import time
from networktables import NetworkTables

# To see messages from networktables, you must setup logging
import logging

logging.basicConfig(level=logging.DEBUG)

ip = "127.0.0.1"

A = np.zeros( (4,4) )

NetworkTables.initialize(server='127.0.0.2')


def valueChanged(table, key, value, isNew):
    print("valueChanged: key: '%s'; value: %s; isNew: %s" % (key, value, isNew))


def connectionListener(connected, info):
    print(info, "; Connected=%s" % connected)


NetworkTables.addConnectionListener(connectionListener, immediateNotify=True)

sd = NetworkTables.getTable("SmartDashboard")
sd.addEntryListener(valueChanged)
table = NetworkTables.getTable("ArrayVals")

while True:
    table.putNumber("Array1", A[3,3])
    print(table.getNumber("Array1", 10))
    time.sleep(3)