class PulsePalObject(object):
    def __init__(self, port):
        self.port = port

    def programOutputChannelParam(self, *args):
        print(args)

    def sendCustomPulseTrain(self, *args):
        print(args)

    def syncAllParams(self):
        pass

    def triggerOutputChannels(self, *args):
        print(args)

    def abortPulseTrains(self):
        pass

# FIXME - this one currently doesn't work, try patching the object
# def setDisplay(P, row1String, row2String):
#     messageBytes = row1String + chr(254) + row2String
#     messageSize = len(messageBytes)
#     messageBytes = chr(P.OpMenuByte) + chr(78) + chr(messageSize) + messageBytes
#     P.Port.write(messageBytes)
