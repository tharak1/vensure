# Generated by Django 5.0.4 on 2024-04-15 17:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('stress_strain', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='traxialtestdata',
            name='area',
            field=models.FloatField(default=0.0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='traxialtestdata',
            name='diameter',
            field=models.FloatField(default=0.0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='traxialtestdata',
            name='length',
            field=models.FloatField(default=0.0),
            preserve_default=False,
        ),
    ]
