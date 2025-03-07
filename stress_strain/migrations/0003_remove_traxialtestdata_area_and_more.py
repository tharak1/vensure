# Generated by Django 5.0.4 on 2024-04-15 17:06

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('stress_strain', '0002_traxialtestdata_area_traxialtestdata_diameter_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='traxialtestdata',
            name='area',
        ),
        migrations.RemoveField(
            model_name='traxialtestdata',
            name='diameter',
        ),
        migrations.RemoveField(
            model_name='traxialtestdata',
            name='length',
        ),
        migrations.AddField(
            model_name='traxialtestdata',
            name='area1',
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name='traxialtestdata',
            name='diameter1',
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name='traxialtestdata',
            name='length1',
            field=models.FloatField(default=0.0),
        ),
    ]
