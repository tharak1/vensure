import uuid
from django.db import models

class TraxialTestData(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    diameter1 = models.FloatField(default=0.0)
    length1 = models.FloatField(default=0.0)
    area1 = models.FloatField(default=0.0)
    lvdt = models.JSONField()
    load_100 = models.JSONField()
    load_200 = models.JSONField()
    strain = models.JSONField()
    corrected_area = models.JSONField()
    deviator_stress_100 = models.JSONField()
    deviator_stress_200 = models.JSONField()
    graph_url = models.URLField(null=True, blank=True)
    radians = models.FloatField(default=0.0)
    degrees = models.FloatField(default=0.0)
    result = models.FloatField(default=0.0)
    long_text_field = models.TextField(default = "")

