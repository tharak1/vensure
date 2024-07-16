from django.shortcuts import render
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import minimize, root_scalar
import io
import base64
from django.http import HttpResponse
from scipy.interpolate import CubicSpline
from shapely.geometry import LineString

class DataPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

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

def stress_strain_analysis(request):
    # Define the data points
    data = [
        DataPoint(0, 0),
        DataPoint(0.003, 6.2946),
        DataPoint(0.006, 12.5892),
        DataPoint(0.009, 18.8838),
        DataPoint(0.012, 25.1784),
        DataPoint(0.014, 29.3787),
        DataPoint(0.017, 35.6733),
        DataPoint(0.02, 39.0078),
        DataPoint(0.035, 52.4043),
        DataPoint(0.052, 62.3493),
        DataPoint(0.079, 66.7836),
        DataPoint(0.124, 69.9543),
        DataPoint(0.167, 70.317),
        DataPoint(0.212, 69.7086),
        DataPoint(0.264, 67.275),
        DataPoint(0.3, 64.8414),
    ]

    # Extract x and y values
    x = np.array([spot.x for spot in data])
    y = np.array([spot.y for spot in data])

    # Find the longest consecutive linear segment with a slope of approximately 1
    start, end = longest_linear_segment(x, y)
    x_linear = x[start:end+1]
    y_linear = y[start:end+1]

    # Calculate the linear regression for the identified linear segment
    slope, intercept, r_value, _, _ = linregress(x_linear, y_linear)

    # Calculate Young's Modulus (E) from the linear segment
    youngs_modulus = slope

    x_linear = x[start:end+3]
    y_linear = y[start:end+3]

    # Create data2 with offset and multiplied y-axis values
    x_offset = x_linear + 0.002
    y_offset = x_linear * youngs_modulus

# Create data2 with offset and multiplied y-axis values
    # x_offset = np.concatenate((x_linear, x[7]+0.002))
    # y_offset = np.concatenate((x_linear * youngs_modulus, x[7]*youngs_modulus))


    # Find intersection of data2 and original stress-strain curve




    # intersection_point = find_intersection(x, y, x_offset, y_offset)
    # cs = CubicSpline(x, y)

    line_1 = LineString(np.column_stack((x,y)))
    line_2 = LineString(np.column_stack((x_offset,y_offset)))
    intersection = line_1.intersection(line_2)

    offset_strain = 0.002  # 0.2% strain
    offset_stress = youngs_modulus * offset_strain + intercept
    yield_strength = np.interp(offset_stress, y, x) 



    # Plot the original data, the linear fit, and the offset data
    plt.figure(figsize=(15, 6))
    # plt.scatter(x, y, label='Original Data')
    plt.plot(x, y, label='Original Data', color='blue')
    # plt.plot(x, cs(x), label='Original Data', color='blue')
    # plt.plot(x_linear, slope * x_linear + intercept, color='red', label=f'Linear Fit (E={youngs_modulus:.2f} MPa)')
    plt.plot(x_offset, y_offset, '--', color='green', label='Offset Data')



    
    # if intersection_point:
    #     plt.scatter(*intersection_point, color='black', label='Intersection Point')




    plt.xlabel('Strain')
    plt.ylabel('Stress (MPa)')
    plt.title('Stress-Strain Curve')
    plt.legend()
    plt.grid(True)


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

    # Render the Django template with the plot, Young's Modulus value, R-squared value, and intersection point
    return render(request, 'analysis.html', {'graphic': graphic, 'youngs_modulus': youngs_modulus, 'r_squared': r_value ** 2, 'intersection_point': intersection,'yeild_strength':yield_strength})




from django.shortcuts import render
from .models import TraxialTestData  # Import your model

def get_traxial_test_data(request, unique_id):
    try:
        test_data = TraxialTestData.objects.get(id=unique_id)
    except TraxialTestData.DoesNotExist:
        return render(request, 'traxial_test_data.html', {'error': 'Test data not found'})

    return render(request, 'traxial_test_data.html', {'test_data': test_data})





from django.http import HttpResponse
from django.template.loader import get_template
from django.core.files import File
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from io import BytesIO

from .models import TraxialTestData  # Import your model

def render_to_pdf(template_src, context_dict={}):
    template = get_template(template_src)
    html = template.render(context_dict)
    
    # Debugging: Save HTML content to a file
    with open('debug.html', 'w') as f:
        f.write(html)
    
    result = BytesIO()

    pdf = SimpleDocTemplate(
        result,
        pagesize=A4,
        rightMargin=30,
        leftMargin=30,
        topMargin=30,
        bottomMargin=18
    )
    
    styles = getSampleStyleSheet()
    elements = []

    # Parse HTML and add to elements
    table_data = []
    for key, value in context_dict['test_data'].items():
        if isinstance(value, list):
            value_str = ", ".join(map(str, value))
        else:
            value_str = str(value)
        table_data.append([key, value_str])

    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(table)

    pdf.build(elements)
    return result



def get_traxial_test_data_pdf(request, unique_id):
    try:
        test_data = TraxialTestData.objects.get(id=unique_id)
    except TraxialTestData.DoesNotExist:
        return HttpResponse('Test data not found')

    print(test_data.__dict__)  # Debugging line

    pdf = render_to_pdf('traxial_test_data.html', {'test_data': test_data.__dict__})
    response = HttpResponse(pdf, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="traxial_test_data_{unique_id}.pdf"'
    return response






from io import BytesIO
from django.http import HttpResponse
from django.template.loader import get_template
from xhtml2pdf import pisa
from django.template.loader import render_to_string
import os 


def render_to_pdf(template_src, context_dict={}):
    template = get_template(template_src)
    html = template.render(context_dict)
    result = BytesIO()
    pdf = pisa.pisaDocument(BytesIO(html.encode("UTF-8")), result, options={
    'page_breaks': True,  # Enable automatic page breaks
    'enable_meta': True  # Disable meta tags (optional)
        })
    if not pdf.err:
        return result.getvalue()
    return None

def get(request, unique_id):
    try:
        test_data = TraxialTestData.objects.get(id=unique_id)     #you can filter using order_id as well
    except:
        return HttpResponse("505 Not Found")
    data = {
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
        'comments': test_data.long_text_field,
    }


    html_content = render_to_string('traxial_test_data.html', data)
    pdf = render_to_pdf('traxial_test_data.html',data)
    #return HttpResponse(pdf, content_type='application/pdf')

    # force download
    if pdf:
        response = HttpResponse(pdf, content_type='application/pdf')
        response['Content-Disposition'] = 'attachment; filename=traxial_test_data.pdf'
        return response
        # response = HttpResponse(pdf, content_type='application/pdf')
        # filename = "Invoice_%s.pdf" %(data['id'])
        # content = "inline; filename='%s'" %(filename)
        # #download = request.GET.get("download")
        # #if download:
        # content = "attachment; filename=%s" %(filename)
        # response['Content-Disposition'] = content
        # return response
    return HttpResponse("Not found")