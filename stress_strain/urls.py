from django.urls import path
from . import views

urlpatterns = [
    path('gene/',views.stress_strain_analysis, name='stress_strain_analysis'),
    path('test_data/<uuid:unique_id>/',views.get_traxial_test_data, name='strain_analysis_data'),
    path('get_traxial_test_data_pdf/<uuid:unique_id>/',views.get_traxial_test_data_pdf, name='get_traxial_test_data_pdf'),
    path('get111/<uuid:unique_id>/',views.get, name='get111'),


]
