#from django.shortcuts import render

import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from .services import repondre


@csrf_exempt
@require_POST
def chat_api(request):
    donnees = json.loads(request.body)
    message = donnees.get("message", "")

    resultat = repondre(message)

    return JsonResponse(resultat)
