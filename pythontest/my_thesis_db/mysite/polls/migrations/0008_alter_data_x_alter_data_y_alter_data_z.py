# Generated by Django 5.0.1 on 2024-02-20 03:28

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("polls", "0007_alter_data_x_alter_data_y_alter_data_z"),
    ]

    operations = [
        migrations.AlterField(
            model_name="data",
            name="x",
            field=models.CharField(),
        ),
        migrations.AlterField(
            model_name="data",
            name="y",
            field=models.CharField(),
        ),
        migrations.AlterField(
            model_name="data",
            name="z",
            field=models.CharField(),
        ),
    ]