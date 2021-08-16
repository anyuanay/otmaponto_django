from django.shortcuts import render, get_object_or_404

# Create your views here.

from django.http import HttpResponse, Http404, HttpResponseRedirect

from .models import Question, Choice, Document
from .forms import OntologyFileForm
from .apps import MatchConfig

from django.urls import reverse

from django.views import generic
from django.conf import settings

import os

from django.views.decorators.csrf import csrf_exempt

from .ontology_mapping.src import OTMapOnto as maponto

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework.parsers import FileUploadParser, MultiPartParser, FormParser

import json
from datetime import datetime

def input(request):
  
  #form = OntologyFileForm(request.POST, request.FILES)
  
  #context = {'form':form}
  context = {}

  return render(request, 'match/input.html', context)

@csrf_exempt
def runmatcher(request):

  if request.method == 'POST':
    
    #form = OntologyFileForm(request.POST, request.FILES)

    #postParams = request.FILES.keys()

    
    sourceFile=request.FILES['source']
    sourceDocument = Document(owlFile=sourceFile)
    sourceDocument.save()

    targetFile=request.FILES['target']
    targetDocument = Document(owlFile=targetFile)
    targetDocument.save()
    
    message = "The source owl {} is stored at {} </br>".format(sourceDocument.owlFile.name, sourceDocument.owlFile.url)
    message = message + "The target owl {} is stored at {} </br>".format(targetDocument.owlFile.name, targetDocument.owlFile.url)

    source_url = sourceDocument.owlFile.url
    target_url = targetDocument.owlFile.url

    source_name = sourceDocument.owlFile.name.replace("uploaded_ontologies/", "")
    target_name = targetDocument.owlFile.name.replace("uploaded_ontologies/", "")
    output_url = "./media/alignments/" + source_name + "_" + target_name + '_alignment.rdf'
    

    #align_url = maponto.match("." + source_url, "." + target_url, output_url, None)

    MatchConfig.mapper.align("." + source_url, "." + target_url, output_url)

    with open(output_url, 'r') as file:
      content = file.read()

    return HttpResponse(content)

#########################

@csrf_exempt
@api_view(['POST'])
def runmatcher_web_file(request):

  if request.method == 'POST':
    
    now_str = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    
    sourceFile = request.data['source']
    source_url = "./media/uploaded_ontologies/source_{}.rdf".format(now_str)
    with open(source_url, 'w') as file:
        file.write(sourceFile)

   
    targetFile = request.data['target']
    #now_str = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    target_url = "./media/uploaded_ontologies/target_{}.rdf".format(now_str)
    with open(target_url, 'w') as file:
        file.write(targetFile)
    
    #now_str = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    output_url = "./media/alignments/source_target_alignment_{}.rdf".format(now_str)
    

    MatchConfig.mapper.align(source_url, target_url, output_url)

    with open(output_url, 'r') as file:
      content = file.read()

    return HttpResponse(content)

##########################


@csrf_exempt
def runmatcher_web_url(request):

  if request.method == 'POST':

    source_url=request.POST.get('source')
    target_url=request.POST.get('target')

    head, source_name = os.path.split(source_url)  
    _, target_name = os.path.split(target_url)  

    output_url = head + "/" + source_name + "_" + target_name + '_alignment.rdf'
    

    #maponto.match(source_url, target_url, output_url[5:], None)

    MatchConfig.mapper.align(source_url, target_url, output_url[5:])

    '''
    with open(output_url, 'r') as file:
      content = file.read()
    '''

    return HttpResponse(output_url)

##############################

class FileUploadView(APIView):
  parser_classes = [FileUploadParser,MultiPartParser]

  def post(self, request):
    file_obj = request.data['source']

    return Response(file_obj)

###############################################

class IndexView(generic.ListView):

  template_name = 'match/index.html'
  context_object_name = 'latest_question_list'

  def get_queryset(self): 
    return Question.objects.order_by('-pub_date')[:5]


class DetailView(generic.DetailView):
  
  model = Question
  template_name = 'match/detail.html'

class ResultsView(generic.DetailView):
  model = Question
  template_name = 'match/results.html'

def vote(request, question_id):
  question = get_object_or_404(Question, pk=question_id)
  try:
    selected_choice = question.choice_set.get(pk=request.POST['choice'])
  except (KeyError, Choice.DoesNotExist):
    # Redisplay the question voting form.
    return render(request, 'match/detail.html', {
            'question': question,
            'error_message': "You didn't select a choice.",
    })
  else:
    selected_choice.votes += 1
    selected_choice.save()
    # Always return an HttpResponseRedirect after successfully dealing
    # with POST data. This prevents data from being posted twice if a
    # user hits the Back button.
    return HttpResponseRedirect(reverse('match:results', args=(question.id,)))