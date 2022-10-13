import numpy as np
import time
from networktables import NetworkTables

import logging

logging.basicConfig(level=logging.DEBUG)

#Array = np.zeros( (4,4) )

#NetworkTables.startClient(server_or_servers="1735")
#NetworkTables.initialize(server='127.0.0.1', )
#sd = NetworkTables.getTable("SmartDashboard")
#NetTable = NetworkTables.getRemoteAddress()

net_table = NetworkTables.create()

net_table.startServer(listenAddress="127.0.0.2")
net_table.setNetworkIdentity("net_table")

table = net_table.getTable('ArrayVals')


i = 0
while True:
    print(table.getNumber('TransX', defaultValue=10))
    print(table.getNumber('TransY', defaultValue=10))
    print(table.getNumber('TransZ', defaultValue=10))
    print(table.getNumber('Array1', defaultValue=10))


    #print(Array[3,3])
    #print("dsTime:", sd.getNumber("dsTime", -1))

    #print(NetTable)

    #print(new_net_table.isServer())

    #net_table.putNumber("Array", A)
    time.sleep(3)
    i += 1