# Generated by Django 5.0.4 on 2024-04-16 02:49

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('stress_strain', '0003_remove_traxialtestdata_area_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='traxialtestdata',
            name='graph_url',
            field=models.URLField(blank=True, null=True),
        ),
    ]
