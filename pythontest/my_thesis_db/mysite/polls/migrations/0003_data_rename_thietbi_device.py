# Generated by Django 5.0.1 on 2024-02-15 06:30

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("polls", "0002_thietbi_type"),
    ]

    operations = [
        migrations.CreateModel(
            name="Data",
            fields=[
                ("number", models.BigAutoField(primary_key=True, serialize=False)),
                ("time", models.DateTimeField(auto_now_add=True)),
                ("data", models.JSONField()),
            ],
        ),
        migrations.RenameModel(
            old_name="ThietBi",
            new_name="Device",
        ),
    ]
