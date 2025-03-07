# Generated by Django 5.0.4 on 2024-04-16 08:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('stress_strain', '0004_traxialtestdata_graph_url'),
    ]

    operations = [
        migrations.AddField(
            model_name='traxialtestdata',
            name='degrees',
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name='traxialtestdata',
            name='radians',
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name='traxialtestdata',
            name='result',
            field=models.FloatField(default=0.0),
        ),
    ]
