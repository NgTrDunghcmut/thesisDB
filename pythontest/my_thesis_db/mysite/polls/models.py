from django.db import models
from django.utils.timezone import now


# Create your models here.
class DevType(models.IntegerChoices):
    Type1 = (0,)
    Type2 = 1


class Devicemanager(models.Manager):
    def addnewdevice(self, name, type, id):
        newdevice = self.create(state=False, name=name, type=type, id=id)
        return newdevice


class Device(models.Model):
    state = models.BooleanField(default=False)
    name = models.CharField(max_length=50)
    type = models.IntegerField(choices=DevType.choices, default=DevType.Type1)
    objects = Devicemanager()


class Data(models.Model):
    number = models.BigAutoField(primary_key=True)
    time = models.DateTimeField(default=now)
    x = models.FloatField()
    y = models.FloatField()
    z = models.FloatField()
    ans = models.FloatField(default=0)
    device = models.ForeignKey(Device, on_delete=models.CASCADE)


# new = Device.objects.addnewdevice()
