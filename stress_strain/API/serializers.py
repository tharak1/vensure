from rest_framework import serializers
from ..models import TraxialTestData

class TraxialTestDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = TraxialTestData
        fields = '__all__'
