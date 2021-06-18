import grpc
from protos import object_server_pb2_grpc as service_grpc
from protos import object_server_pb2 as service

# make an RPC request
# channel = grpc.insecure_channel('10.60.108.45:9080')
channel = grpc.insecure_channel('localhost:50051')

stub = service_grpc.ObjectServiceStub(channel)
metadata = [('api_key', '5GpbfutWaHoJllrhsDtE9UJVDpOwlBFW')]

# request = service.ObjectRequest(ws="rtsp://root:1111@117.6.121.13:554/axis-media/media.amp")
# response = stub.ForgotObjectDetect(service.ObjectRequest(ws="rtsp://root:1111@117.6.121.13:554/axis-media/media.amp"), metadata=metadata)
# response = stub.CountObjectVideo(service.ObjectRequest(ws="/home/datdt/Downloads/video/Object_sign.mp4"))
box = service.Box(x=300,y=200,width=450,height=500)
poly = []
p1 = service.Point(x=300, y=200)
p2 = service.Point(x=1000, y = 1000)
p3 = service.Point(x=1000, y = 200)
p4 = service.Point(x=300, y = 1000)
p5 = service.Point(x=800, y = 150)

poly.append(p1)
poly.append(p5)

poly.append(p2)
poly.append(p4)

areas = service.Area(area_id='1', poly=poly)
response = stub.ViolateObjectDetect(service.ObjectRequest(ws="/home/datdt/Videos/crowd.mp4", areas=[areas]), metadata=metadata)
# request = service.StreamingRequest(source_url="rtsp://admin:abcd1234@172.16.10.84:554/Channels/101")
# results = stub.DetectFire(request)
for result in response:
    # for i, _ in enumerate(result.name):
    #     print(result.name[i], " ", result.count[i])
    print("Timestamp: ", result.forgot_obj)
    # for Object in result.counted_Objects:
    #     for detail in Object.detail:
    #         print("Object: ", detail.Object_type , " has ", detail.count)



# import requests
# import json
#
# def sendSMS(msg,numbers):
#     headers = {
#     "authkey": "place AUTH-KEY here",
#     "Content-Type": "application/json"
#     }
#
#     data = "{ \"sender\": \"GTURES\", \"route\": \"4\", \"country\": \"91\", \"sms\": [ { \"message\": \""+msg+"\", \"to\": "+json.dumps(numbers)+" } ] }"
#
#     requests.post("https://api.msg91.com/api/v2/sendsms", headers=headers, data=data)
# sendSMS("hi, i'm detdet",[+84966250499])
