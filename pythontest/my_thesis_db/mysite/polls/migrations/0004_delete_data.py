# Generated by Django 5.0.1 on 2024-02-15 07:12

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("polls", "0003_data_rename_thietbi_device"),
    ]

    operations = [
        migrations.DeleteModel(
            name="Data",
        ),
    ]
