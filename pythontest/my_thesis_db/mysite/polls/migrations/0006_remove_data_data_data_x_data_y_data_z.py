# Generated by Django 5.0.1 on 2024-02-17 09:34

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("polls", "0005_data"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="data",
            name="data",
        ),
        migrations.AddField(
            model_name="data",
            name="x",
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name="data",
            name="y",
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name="data",
            name="z",
            field=models.IntegerField(default=0),
        ),
    ]
