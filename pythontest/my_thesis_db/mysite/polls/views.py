from django.shortcuts import render
import requests
from django.http import JsonResponse, HttpResponse
from .models import Data, Device
import json
import csv
from django.views.decorators.csrf import csrf_exempt
from keras.models import load_model
from .prediction import *
from datetime import datetime
import pickle
import paho.mqtt.client as mqtt
from .trainingmodel import *
import os


@csrf_exempt
def chartdata(request):
    response = HttpResponse()
    if request.method == "POST":
        try:
            device_info = json.loads(request.body)
            desired_date = device_info["date"]
            desired_date_obj = datetime.strptime(desired_date, "%Y-%m-%d").date()
            cdata = list(
                Data.objects.filter(
                    device_id=device_info["device_id"], time__date=desired_date_obj
                )
                .order_by("number")
                .values("x", "y", "z", "time", "ans")
            )
            for data_point in cdata:
                data_point["date"] = data_point["time"].strftime("%Y-%m-%d")
                data_point["time"] = data_point["time"].strftime("%H:%M:%S")
            # print(cdata)
            response = JsonResponse(cdata, safe=False)
            return response
        except Exception as e:
            return JsonResponse({"error": f"Error occurred: {str(e)}"}, status=500)
    else:
        return JsonResponse({"error": "Only GET requests are allowed"}, status=406)


@csrf_exempt
def csvexport(request):
    if request.method == "POST":
        device_info = json.loads(request.body)
        desired_date = device_info["date"]
        desired_date_obj = datetime.strptime(desired_date, "%Y-%m-%d").date()
        csvfile = (
            Data.objects.filter(
                device_id=device_info["device_id"], time__date=desired_date_obj
            )
            .order_by("number")
            .values("x", "y", "z", "time", "ans")
        )
        response = HttpResponse(
            content_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="data1.csv"'},
        )
        writer = csv.writer(response)
        writer.writerow(["time", "x", "y", "z"])
        csvfile_fields = csvfile.values_list("time", "x", "y", "z")
        for data in csvfile_fields:
            writer.writerow(data)
    return response


@csrf_exempt
def getitems(request):
    itemslist = list(Device.objects.all().order_by("name").values())
    try:
        if request.method == "GET":
            response = JsonResponse(itemslist, safe=False)
            return response
        elif request.method == "POST":
            device_info = json.loads(request.body)
            Info = list(
                Device.objects.filter(id=device_info["device_id"]).values(
                    "name", "type", "id"
                )
            )
            return JsonResponse(Info, safe=False)
    except Exception as e:
        return JsonResponse({"error": f"Error occurred: {str(e)}"}, status=500)
    else:
        return JsonResponse({"error": "Only GET requests are allowed"}, status=406)


@csrf_exempt
def MLactive(request):
    try:
        if request.method == "POST":
            global detect
            detect = json.loads(request.body)["active"]
            id = json.loads(request.body)["device_id"]
            hint = loadnewmodel(id)
            print(hint)
            return HttpResponse(detect)
        elif request.method == "GET":
            detect = "0"
            return HttpResponse(detect)
    except Exception as e:
        return HttpResponse(e)


def mqtt_receive_callback(client, userdata, message):
    if detect == "0":
        print("hei")
        try:
            datas = json.loads(message.payload.decode())
            id_value = datas["id"][0]
            if Device.objects.filter(id=id_value).exists():
                if "x" in datas:
                    if Data.objects.filter(device_id=id_value).exists():
                        for i in range(len(datas["x"])):
                            existing_data = Data()
                            existing_data.device_id = id_value
                            # If a record with the id exists, update it with the new values
                            existing_data.x = datas["x"][i]
                            existing_data.y = datas["y"][i]
                            existing_data.z = datas["z"][i]
                            # print(existing_data)
                            # existing_data.time = now #+ timezone.localdate()
                            existing_data.save(force_insert=True)
                    else:

                        inputs = Data.objects.create(
                            device_id=id_value,
                            x=datas["x"][0],
                            y=datas["y"][0],
                            z=datas["z"][0],
                        )
                        # print("1")
                        inputs.save()
                        for i in range(1, len(datas["x"])):
                            existing_data = Data()
                            existing_data.device_id = id_value
                            # If a record with the id exists, update it with the new values
                            existing_data.x = datas["x"][i]
                            existing_data.y = datas["y"][i]
                            existing_data.z = datas["z"][i]
                            # print("2")
                            # existing_data.time = now #+ timezone.localdate()
                            existing_data.save(force_insert=True)
                else:
                    pass
            else:
                new = Device.objects.addnewdevice(
                    datas["name"][0], datas["type"][0], id_value
                )
                existing_data = Data.objects.get(device_id=id_value)
                for i in range(len(datas["x"])):
                    # If a record with the id exists, update it with the new values
                    existing_data.x = datas["x"][i]
                    existing_data.y = datas["y"][i]
                    existing_data.z = datas["z"][i]

                    existing_data.save()
            return HttpResponse({"message": "Data saved successfully"}, status=201)
        except Exception as e:
            return HttpResponse({"error": f"An error occurred: {str(e)}"}, status=500)
    elif detect == "1":
        print("hai")
        datas = json.loads(message.payload.decode())
        ans = predict2(datas)
        print(ans)
        for i in range(len(datas["x"])):
            existing_data = Data()
            existing_data.device_id = datas["id"][i]
            # If a record with the id exists, update it with the new values
            existing_data.x = datas["x"][i]
            existing_data.y = datas["y"][i]
            existing_data.z = datas["z"][i]
            existing_data.ans = ans

            # existing_data.time = now #+ timezone.localdate()
            existing_data.save(force_insert=True)
        return HttpResponse({"message": "Data saved successfully"}, status=201)


def mqtt_connect(request):
    try:
        if request.method == "GET":
            # Create an MQTT client
            global client
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

            # Set up the MQTT callback function
            client.on_message = mqtt_receive_callback
            # Connect to the MQTT broker
            client.connect("192.168.16.163", 1800, 60)
            client.on_connect = print("connected")
            client.on_message = mqtt_receive_callback
            # Start the MQTT client loop
            client.loop_start()
            return HttpResponse("connected")
    except Exception as e:
        return HttpResponse(e)


data = 0


@csrf_exempt
def mqtt_publish_request(request):
    global client
    # Create an MQTT client
    try:
        if request.method == "POST":
            data = json.loads(request.body)["on"]
            res = client.publish("control", data)
            if res.rc == mqtt.MQTT_ERR_SUCCESS:
                # client.disconnect()
                return HttpResponse(f"send {data} to topic control")
            else:
                # client.disconnect()
                return HttpResponse("Failed")
        else:
            return HttpResponse("wrong method")
    except Exception as e:
        return HttpResponse(e)


current_topic = None


@csrf_exempt
def mqtt_subcribe(request):
    try:
        if request.method == "POST":
            id = json.loads(request.body)["id"]
            global current_topic
            if current_topic:
                client.unsubscribe(current_topic)
                print(f"Unsubscribed from {current_topic}")
            client.subscribe(f"data/{id}")
            client.on_subscribe = print(f"subcribed to data/{id}")
            current_topic = f"data/{id}"
            return HttpResponse("connected")
    except Exception as e:
        return HttpResponse(e)


def mqtt_disconnect(request):
    try:
        if request.method == "GET":
            if client and client.is_connected():
                client.disconnect()
                print("disconnect")
                return HttpResponse("Disconnected")
            else:
                print("no new connection")
                return HttpResponse("No new connection")
    except Exception as e:
        return HttpResponse(e)


@csrf_exempt
def resulttraining(request):
    try:
        if request.method == "POST":
            id = json.loads(request.body)["id"]
            file_path = os.path.join(
                "D:/sem232/thesis/pythontest/my_thesis_db/mysite/polls/savedmodels/mini/",
                f"variables_{id}.pkl",
            )

            # Check if the file exists
            if os.path.exists(file_path):
                # If file exists, load its contents
                with open(file_path, "rb") as file:
                    loaded_var1, loaded_var2, loaded_var3, accu, mse = pickle.load(file)
                response_data = {
                    "recommend": loaded_var1,
                    "loss": loaded_var2,
                    "val_loss": loaded_var3,
                    "accu": accu,
                    "mse": list(mse),
                }
                return JsonResponse(response_data)
            else:
                response_data = {
                    "recommend": "Model for this device has not been trained"
                }
                # If file does not exist, return a specific response
                return JsonResponse(response_data)
    except Exception as e:
        # Handle other exceptions and return as HttpResponse
        return HttpResponse(str(e), status=500)


@csrf_exempt
def train(request):
    try:
        if request.method == "POST":
            data = json.loads(request.body)
            ID = data["id"]
            dataset = Data.objects.filter(device_id=data["id"]).values("x", "y", "z")
            training_model(dataset, data["id"])
            response_data = {"res": f"Finished training for {ID}"}
            return JsonResponse(response_data, safe=False)

    except Exception as e:
        return HttpResponse(e)
