# Generated by Django 5.0.1 on 2024-04-04 15:46

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("polls", "0011_anomaly"),
    ]

    operations = [
        migrations.DeleteModel(
            name="Anomaly",
        ),
        migrations.AddField(
            model_name="data",
            name="ans",
            field=models.FloatField(default=0),
        ),
    ]
