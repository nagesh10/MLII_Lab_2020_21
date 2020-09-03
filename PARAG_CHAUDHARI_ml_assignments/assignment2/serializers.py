from rest_framework import serializers
from .models import Model_image


class MODEL_serializer(serializers.ModelSerializer):
    class Meta:
        model = Model_image
        fields = ['image']

