import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from shapely.geometry import LineString
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
from rest_framework import serializers
from ..models import TraxialTestData 
from .serializers import TraxialTestDataSerializer
import uuid
import math

class DataPointSerializer(serializers.Serializer):
    x = serializers.ListField(child=serializers.FloatField())
    y = serializers.ListField(child=serializers.FloatField())

def longest_linear_segment(x, y):
    longest_segment = (0, 0)
    current_segment = (0, 0)
    
    initial_slope, intercept, _, _, _ = linregress(x[:6], y[:6])  # Calculate slope of the first 6 points
    
    for i in range(1, len(x)):
        slope, intercept, _, _, _ = linregress(x[:i+1], y[:i+1])
        
        if abs(round(slope) - round(initial_slope)) == 0:  # Check if the rounded slope is the same as the rounded initial slope
            current_segment = (current_segment[0], i)
            
            if current_segment[1] - current_segment[0] > longest_segment[1] - longest_segment[0]:
                longest_segment = current_segment
        else:
            current_segment = (i, i)
    
    return longest_segment

@api_view(['POST'])
def stress_strain_analysis_api(request):
    # Deserialize data from the request
    serializer = DataPointSerializer(data=request.data)
    if serializer.is_valid():
        x = serializer.validated_data['x']
        y = serializer.validated_data['y']

        # Find the longest consecutive linear segment with a slope of approximately 1
        start, end = longest_linear_segment(x, y)
        x_linear = np.array(x[start:end+1])
        y_linear = np.array(y[start:end+1])

        # Calculate the linear regression for the identified linear segment
        slope, intercept, r_value, _, _ = linregress(x_linear, y_linear)

        # Calculate Young's Modulus (E) from the linear segment
        youngs_modulus = slope

        x_linear = np.array(x[start:end+2])
        y_linear = np.array(y[start:end+2])

        # Create data2 with offset and multiplied y-axis values
        x_offset = x_linear + 0.002
        y_offset = x_linear * youngs_modulus

        line_1 = LineString(np.column_stack((x,y)))
        line_2 = LineString(np.column_stack((x_offset,y_offset)))
        intersection = line_1.intersection(line_2)

        offset_strain = 0.002
        offset_stress = youngs_modulus * offset_strain + intercept
        yield_strength = np.interp(offset_stress, y, x)

        # Plot the original data, the linear fit, and the offset data
        plt.figure(figsize=(15, 6))
        plt.plot(x, y, label='Original Data', color='blue')
        plt.plot(x_offset, y_offset, '--', color='green', label='Offset Data')
        plt.xlabel('Strain')
        plt.ylabel('Stress (MPa)')
        plt.title('Stress-Strain Curve')
        plt.legend()
        plt.grid(True)

        # Find intersection point and plot it
        plt.plot(*intersection.xy,'ro')

        # Save the plot to a buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()

        # Encode the image to base64
        graphic = base64.b64encode(image_png)
        graphic = graphic.decode('utf-8')

        response_data = {
            'graphic': graphic,
            'youngs_modulus': youngs_modulus,
            'r_squared': r_value ** 2,
            'intersection_point': {'x': intersection.x, 'y': intersection.y},
            'yield_strength': yield_strength,
            'x_offset': list(x_offset),
            'y_offset': list(y_offset)
        }

        return Response(response_data)
    else:
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)




class DataSerializerForTraxial(serializers.Serializer):
    lvdt = serializers.ListField(child=serializers.FloatField())
    load_100 = serializers.ListField(child=serializers.FloatField())
    load_200 = serializers.ListField(child=serializers.FloatField())
    length = serializers.FloatField()
    area = serializers.FloatField()
    diameter = serializers.FloatField()
        


@api_view(['POST'])
def traxial_test(request):
    serializer = DataSerializerForTraxial(data=request.data) 
    if serializer.is_valid():
        lvdt = serializer.validated_data['lvdt']
        load_100 = serializer.validated_data['load_100']
        load_200 = serializer.validated_data['load_200'] 
        length =  serializer.validated_data['length']
        area = serializer.validated_data['area']
        diameter = serializer.validated_data['diameter']

        strain = []
        corrected_area = []
        deviator_stress_100 = []
        deviator_stress_200 = []

        for i in range(len(lvdt)):
            strain.append((lvdt[i]/1000)/length)
            corrected_area.append(area/(1-strain[i]))
            deviator_stress_100.append((load_100[i]-load_100[0])/corrected_area[i])
            deviator_stress_200.append((load_200[i]-load_200[0])/corrected_area[i])
        
        x1 = 100
        x2 = 200

        y1 = round(max(deviator_stress_100)+x1,3)
        y2 = round(max(deviator_stress_200)+x2,3)


        delta_x = x2 - x1
        delta_y = y2 - y1

        radians = math.atan(math.sqrt(delta_y / delta_x))
        degrees = math.degrees(radians)
        radians = math.radians(degrees)
        result = (y1 - x1 * math.pow(math.tan(radians), 2)) / (2 * math.tan(radians))
        x = round((2*degrees)-90)

        test_data = TraxialTestData.objects.create(
            diameter1 = diameter,
            length1 = length,
            area1 = area,
            lvdt=lvdt,
            load_100=load_100,
            load_200=load_200,
            strain=[round(val, 4) for val in strain],
            corrected_area=[round(val, 6) for val in corrected_area],
            deviator_stress_100=[round(val, 3) for val in deviator_stress_100],
            deviator_stress_200=[round(val, 3) for val in deviator_stress_200],
            radians = x,
            degrees = round(degrees,4) ,
            result = round(result,4)
        )



        plt.figure(figsize=(10, 6))
        plt.plot(strain, deviator_stress_100, label='Deviator Stress 100')
        plt.plot(strain, deviator_stress_200, label='Deviator Stress 200')
        plt.xlabel('Strain')
        plt.ylabel('Deviator Stress')
        plt.title('Traxial Test Results')
        plt.legend()

        
        plt.grid(True)


        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        graph_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        graph_url = f"data:image/png;base64,{graph_base64}"

        plt.close()

        test_data.graph_url = graph_url
        test_data.save()

        response_data = {
            'id': test_data.id,
            'graph_url': graph_url
        }
        return Response(response_data)
    else:
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    



@api_view(['GET', 'PUT'])
def get_traxial_test_data(request, unique_id):
    try:
        test_data = TraxialTestData.objects.get(id=unique_id)
    except TraxialTestData.DoesNotExist:
        return Response({"error": "Test data not found"}, status=status.HTTP_404_NOT_FOUND)
    if request.method == 'GET':
        response_data = {
            'id': test_data.id,
            'diameter' : test_data.diameter1,
            'length' : test_data.length1,
            'area' : test_data.area1,
            'lvdt': test_data.lvdt,
            'load_100': test_data.load_100,
            'load_200': test_data.load_200,
            'strain': test_data.strain,
            'correctedArea': test_data.corrected_area,
            'deviatorArea_100': test_data.deviator_stress_100,
            'deviatorArea_200': test_data.deviator_stress_200,
            'graph_url':test_data.graph_url,
            'radians' : test_data.radians,
            'degrees' : test_data.degrees,
            'result' : test_data.result,
            'long_text_field': test_data.long_text_field,
        }

        return Response(response_data)
    
    elif request.method == 'PUT':
        # Assuming the long_text_field data is passed in the request body as 'long_text_field'
        long_text_data = request.data.get('long_text_field')
        if long_text_data is not None:
            test_data.long_text_field = long_text_data
            test_data.save()
            return Response({"message": "Long text field updated successfully"})
        else:
            return Response({"error": "Long text field data not provided"}, status=status.HTTP_400_BAD_REQUEST)


@api_view(['PUT'])
def update_traxial_test(request, unique_id):
    try:
        test_data = TraxialTestData.objects.get(id=unique_id)
    except TraxialTestData.DoesNotExist:
        return Response({"error": "Test data not found"}, status=status.HTTP_404_NOT_FOUND)

    serializer = DataSerializerForTraxial(instance=test_data, data=request.data)
    if serializer.is_valid():
        lvdt = serializer.validated_data.get('lvdt', test_data.lvdt)
        load_100 = serializer.validated_data.get('load_100', test_data.load_100)
        load_200 = serializer.validated_data.get('load_200', test_data.load_200)
        length = serializer.validated_data.get('length', test_data.length1)
        area = serializer.validated_data.get('area', test_data.area1)
        diameter = serializer.validated_data.get('diameter', test_data.diameter1)

        strain = []
        corrected_area = []
        deviator_stress_100 = []
        deviator_stress_200 = []

        for i in range(len(lvdt)):
            strain.append((lvdt[i]/1000)/length)
            corrected_area.append(area/(1-strain[i]))
            deviator_stress_100.append((load_100[i]-load_100[0])/corrected_area[i])
            deviator_stress_200.append((load_200[i]-load_200[0])/corrected_area[i])

        x1 = 100
        x2 = 200

        y1 = round(max(deviator_stress_100)+x1, 3)
        y2 = round(max(deviator_stress_200)+x2, 3)

        delta_x = x2 - x1
        delta_y = y2 - y1

        radians = math.atan(math.sqrt(delta_y / delta_x))
        degrees = math.degrees(radians)
        radians = math.radians(degrees)
        result = (y1 - x1 * math.pow(math.tan(radians), 2)) / (2 * math.tan(radians))
        x = round((2*degrees)-90)

        test_data.diameter1 = diameter
        test_data.length1 = length
        test_data.area1 = area
        test_data.lvdt = lvdt
        test_data.load_100 = load_100
        test_data.load_200 = load_200
        test_data.strain = [round(val, 4) for val in strain]
        test_data.corrected_area = [round(val, 6) for val in corrected_area]
        test_data.deviator_stress_100 = [round(val, 3) for val in deviator_stress_100]
        test_data.deviator_stress_200 = [round(val, 3) for val in deviator_stress_200]
        test_data.radians = x
        test_data.degrees = round(degrees, 4)
        test_data.result = round(result, 4)

        plt.figure(figsize=(10, 6))
        plt.plot(strain, deviator_stress_100, label='Deviator Stress 100')
        plt.plot(strain, deviator_stress_200, label='Deviator Stress 200')
        plt.xlabel('Strain')
        plt.ylabel('Deviator Stress')
        plt.title('Updated Traxial Test Results')
        plt.legend()
        plt.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        graph_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        graph_url = f"data:image/png;base64,{graph_base64}"

        plt.close()

        test_data.graph_url = graph_url
        test_data.save()

        response_data = {
            'id': test_data.id,
            'diameter': test_data.diameter1,
            'length': test_data.length1,
            'area': test_data.area1,
            'lvdt': test_data.lvdt,
            'load_100': test_data.load_100,
            'load_200': test_data.load_200,
            'strain': test_data.strain,
            'correctedArea': test_data.corrected_area,
            'deviatorArea_100': test_data.deviator_stress_100,
            'deviatorArea_200': test_data.deviator_stress_200,
            'graph_url': test_data.graph_url,
            'radians': test_data.radians,
            'degrees': test_data.degrees,
            'result': test_data.result,
        }
        return Response(response_data)
    else:
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)



@api_view(['GET'])
def get_all_traxial_test_data(request):
    try:
        test_data = TraxialTestData.objects.all()
    except TraxialTestData.DoesNotExist:
        return Response({"error": "No test data found"}, status=status.HTTP_404_NOT_FOUND)

    serializer = TraxialTestDataSerializer(test_data, many=True)
    return Response(serializer.data)




from io import BytesIO
from django.http import HttpResponse
from django.template.loader import get_template
from xhtml2pdf import pisa
from ..models import TraxialTestData

def render_to_pdf(template_src, context_dict={}):
    template = get_template(template_src)
    html = template.render(context_dict)
    result = BytesIO()
    pdf = pisa.pisaDocument(BytesIO(html.encode("UTF-8")), result)
    if not pdf.err:
        return result.getvalue()
    return None

def convert_to_pdf(request, unique_id):
    try:
        test_data = TraxialTestData.objects.get(id=unique_id)
    except TraxialTestData.DoesNotExist:
        return HttpResponse("Test data not found", status=404)
    
    data = {
        'id': test_data.id,
        'diameter': test_data.diameter1,
        'length': test_data.length1,
        'area': test_data.area1,
        'lvdt': test_data.lvdt,
        'load_100': test_data.load_100,
        'load_200': test_data.load_200,
        'strain': test_data.strain,
        'correctedArea': test_data.corrected_area,
        'deviatorArea_100': test_data.deviator_stress_100,
        'deviatorArea_200': test_data.deviator_stress_200,
        'graph_url': test_data.graph_url,
        'radians': test_data.radians,
        'degrees': test_data.degrees,
        'result': test_data.result,
        'comments': test_data.long_text_field,
    }

    pdf_content = render_to_pdf('traxial_test_data.html', data)
    if pdf_content:
        response = HttpResponse(pdf_content, content_type='application/pdf')
        filename = f"Invoice_{test_data.id}.pdf"
        response['Content-Disposition'] = f"attachment; filename='{filename}'"
        return response
    return HttpResponse("Error generating PDF", status=500)
