from django.shortcuts import render
from .models import Model_image
from .serializers import MODEL_serializer
from rest_framework import generics
from rest_framework.parsers import FormParser
from rest_framework.response import Response
from rest_framework import status
from PIL import Image
from .tensorflow_models import cifar,mnist
# Create your views here.


class cifar10(generics.ListCreateAPIView):
    queryset = Model_image.objects.all()
    serializer_class = MODEL_serializer

    def post(self,request,*args,**kwargs):
        if(request.method=="POST"):
                print(request.data)

                image = request.data['image']

                image = Image.open(image)

                print("in this")
                result = cifar(image)


                return Response(status=status.HTTP_200_OK, data={"result": result})
class MNIST(generics.ListCreateAPIView):
    queryset = Model_image.objects.all()
    serializer_class = MODEL_serializer

    def post(self,request,*args,**kwargs):
        if(request.method=="POST"):
                print(request.data)

                image = request.data['image']

                image = Image.open(image)

                print("in this")
                result = mnist(image)


                return Response(status=status.HTTP_200_OK, data={"result": result})












