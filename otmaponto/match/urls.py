from django.urls import path 
from . import views 

app_name = 'match'
urlpatterns = [
  path('', views.input, name='input'),
  path('fileupload', views.FileUploadView.as_view(), name='fileupload'),
  path('runmatcher_web_url', views.runmatcher_web_url, name='runmatcher_web_url'),
  path('runmatcher_web_file', views.runmatcher_web_file, name='runmatcher_web_file'),
  path('runmatcher', views.runmatcher, name='runmatcher'),
  path('index', views.IndexView.as_view(), name = 'index'),
  # ex: /polls/5/
  path('<int:pk>/', views.DetailView.as_view(), name='detail'),
  # ex: /polls/5/results/
  path('<int:pk>/results/', views.ResultsView.as_view(), name='results'),
  # ex: /polls/5/vote/
  path('<int:question_id>/vote/', views.vote, name='vote'),
]