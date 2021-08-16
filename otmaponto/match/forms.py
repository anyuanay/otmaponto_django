from django import forms


class OntologyFileForm(forms.Form):
  
  source = forms.FileField(label='Select source ontology OWL file')
  target = forms.FileField(label='Select target ontology OWL file')
