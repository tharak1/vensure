from django.urls import path
from .views import stress_strain_analysis_api,traxial_test,get_traxial_test_data,update_traxial_test,get_all_traxial_test_data,convert_to_pdf

urlpatterns = [
    path('stress_strain_analysis/', stress_strain_analysis_api, name='stress_strain_analysis_api'),
    path('traxial_upload/',traxial_test,name='traxial_test'),
    path('traxial_test/<uuid:unique_id>/', get_traxial_test_data, name='get_traxial_test_data'),
    path('traxial_test_update/<uuid:unique_id>/', update_traxial_test, name='update_traxial_test_data'),
    path('get_all_traxial_test_data/',get_all_traxial_test_data,name='get all data'),
    path('generatePdf/<uuid:unique_id>/',convert_to_pdf,name="pdf_generation")
]